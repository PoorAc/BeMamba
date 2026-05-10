"""
domain_adapt.py
───────────────
Domain adaptation and cross-scenario fine-tuning for BeMamba.

Supports three adaptation strategies:
  1. Full fine-tune: train all layers on new domain
  2. Adapter-only: freeze extractors, fine-tune adapter + classifier
  3. Progressive: gradually unfreeze layers as training progresses

Usage:
    python domain_adapt.py --checkpoint ./checkpoints/<run-name>/models/best.pt \
        --source-csv ./Scenarios/31/31.csv \
        --target-csv ./Scenarios/32/32.csv \
        --strategy adapter-only \
        --epochs 50 \
        --use-ewc \
        --ewc-lambda 1.0 \
        --run-name domain_adapt_31_to_32_strong_ewc
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

import config
from dataset import build_datasets, DeepSense6GDataset
from models import BeMamba


# ─────────────────────────────────────────────────────────────────────────────
# Layer freezing utilities
# ─────────────────────────────────────────────────────────────────────────────

def freeze_extractor_layers(model: BeMamba):
    """Freeze all modality extractors (domain-invariant feature extraction)."""
    if hasattr(model, 'image_ext'):
        for param in model.image_ext.parameters():
            param.requires_grad = False
        for param in model.image_tsm.parameters():
            param.requires_grad = False
    
    if hasattr(model, 'gps_ext'):
        for param in model.gps_ext.parameters():
            param.requires_grad = False
        for param in model.gps_tsm.parameters():
            param.requires_grad = False
    
    if hasattr(model, 'lidar_ext'):
        for param in model.lidar_ext.parameters():
            param.requires_grad = False
        for param in model.lidar_tsm.parameters():
            param.requires_grad = False
    
    if hasattr(model, 'radar_ext'):
        for param in model.radar_ext.parameters():
            param.requires_grad = False
        for param in model.radar_tsm.parameters():
            param.requires_grad = False


def freeze_msm_layer(model: BeMamba):
    """Freeze cross-modal fusion layer."""
    if hasattr(model, 'msm'):
        for param in model.msm.parameters():
            param.requires_grad = False


def freeze_classifier_layer(model: BeMamba):
    """Freeze the beam classifier head."""
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = False


def unfreeze_all(model: BeMamba):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


def unfreeze_adapter_and_classifier(model: BeMamba):
    """Unfreeze only adapter and classifier (keep extractors frozen)."""
    freeze_extractor_layers(model)
    freeze_msm_layer(model)
    
    if hasattr(model, 'adapter'):
        for param in model.adapter.parameters():
            param.requires_grad = True
    
    for param in model.classifier.parameters():
        param.requires_grad = True


def print_trainable_params(model: BeMamba):
    """Print count of trainable vs frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    print(f"\n[Trainable params] {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print(f"[Frozen params]    {frozen:,} / {total:,} ({100*frozen/total:.1f}%)\n")


# ─────────────────────────────────────────────────────────────────────────────
# Elastic Weight Consolidation (EWC) for catastrophic forgetting prevention
# ─────────────────────────────────────────────────────────────────────────────

class ElasticWeightConsolidation:
    def __init__(self, model: nn.Module, device: str, fisher_lambda: float = 0.4):
        self.device = device
        self.fisher_lambda = fisher_lambda

        # Store source optimal params BEFORE any adaptation
        self.optimal_params = {
            name: param.data.clone().detach()
            for name, param in model.named_parameters()
        }

        self.fisher_matrix = {}

    def compute_fisher_information(self, model, loader, num_batches=None):
        print("\n[EWC] Computing Fisher Information Matrix...")

        model.eval()

        fisher = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
        }

        total_samples = 0

        for batch_idx, batch in enumerate(tqdm(loader, desc="  fisher", leave=False)):
            if num_batches and batch_idx >= num_batches:
                break

            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            labels = batch["label"]
            batch_size = labels.size(0)

            model.zero_grad()
            logits = model(batch)

            # ✅ Correct log-likelihood based loss
            log_probs = torch.log_softmax(logits, dim=1)
            loss = log_probs.gather(1, labels.unsqueeze(1)).mean()

            loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += (param.grad.data ** 2) * batch_size

            total_samples += batch_size

        # Normalize properly
        for name in fisher:
            fisher[name] /= max(total_samples, 1)
            fisher[name] = fisher[name].clamp(min=1e-6)

        self.fisher_matrix = fisher
        print(f"[EWC] Fisher computed over {total_samples} samples")

    def ewc_penalty(self, model):
        penalty = torch.tensor(0.0, device=self.device)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.fisher_matrix:
                continue
            if "classifier" in name:
                continue  # avoid over-regularizing classifier

            fisher = self.fisher_matrix[name]
            optimal = self.optimal_params[name]

            penalty += (fisher * (param - optimal) ** 2).sum()

        return penalty

    def get_config(self):
        return {
            "fisher_lambda": self.fisher_lambda,
            "num_fisher_params": sum(f.numel() for f in self.fisher_matrix.values())
        }



# ─────────────────────────────────────────────────────────────────────────────
# Loading and checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────

def load_source_model(checkpoint_path: str, device: str) -> tuple:
    """Load model from source domain checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    if "modalities" not in ckpt or "model_state" not in ckpt:
        raise ValueError("Checkpoint missing 'modalities' or 'model_state'")
    
    model = BeMamba(
        modalities=ckpt["modalities"],
        d_model=config.D_MODEL,
        num_beams=config.NUM_BEAMS,
        use_adapters=True,
    ).to(device)
    
    # Load state dict with strict=False to handle cases where checkpoint doesn't have adapter weights
    # (adapter will initialize with random weights and be trained on target domain)
    model.load_state_dict(ckpt["model_state"], strict=False)
    
    metadata = {
        "source_checkpoint": checkpoint_path,
        "source_epoch": ckpt.get("epoch", "unknown"),
        "source_scenario": ckpt.get("scenario", "unknown"),
        "source_modalities": ckpt["modalities"],
    }
    
    return model, metadata


def to_device(batch: dict, device: str) -> dict:
    """Move batch tensors to device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# Training and evaluation
# ─────────────────────────────────────────────────────────────────────────────

def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Compute top-k accuracy."""
    _, topk = logits.topk(k, dim=1)
    correct = topk.eq(labels.unsqueeze(1).expand_as(topk))
    return correct.any(dim=1).float().mean().item()


def train_epoch(model, loader, optimizer, criterion, device, ewc=None, ewc_lambda=0.0):
    model.train()
    total_loss = total_acc = 0.0

    for batch in tqdm(loader, desc="  train", leave=False):
        batch = to_device(batch, device)
        labels = batch["label"]

        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, labels)

        if ewc is not None:
            loss = loss + ewc_lambda * ewc.ewc_penalty(model)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += topk_accuracy(logits.detach(), labels, 1)

    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, device, top_k=[1, 3, 5]) -> dict:
    """Evaluate model on a dataset."""
    model.eval()
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc="  eval ", leave=False):
        batch = to_device(batch, device)
        logits = model(batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"].cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    return {f"top{k}": topk_accuracy(all_logits, all_labels, k) for k in top_k}


def plot_adaptation_metrics(history: dict, save_path: str):
    """Plot training metrics during domain adaptation."""
    epochs = [e['epoch'] for e in history['epochs']]
    train_loss = [e['train_loss'] for e in history['epochs']]
    train_top1 = [e['train_top1'] for e in history['epochs']]
    val_top1 = [e['val_top1'] for e in history['epochs']]
    val_top3 = [e['val_top3'] for e in history['epochs']]
    src_top1 = [e['src_top1'] for e in history['epochs']]
    src_top3 = [e['src_top3'] for e in history['epochs']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Domain Adaptation: {history['source_scenario']} → {history['target_scenario']}", fontsize=12)

    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, train_top1, 'g-', linewidth=2, label='Train Top-1')
    ax2.plot(epochs, val_top1, 'g--', linewidth=2, label='Val Top-1')
    ax2.plot(epochs, val_top3, 'orange', linewidth=2, label='Val Top-3')
    ax2.plot(epochs, src_top1, 'purple', linewidth=2, label='Src Top-1')
    ax2.plot(epochs, src_top3, 'purple', linestyle='--', linewidth=2, label='Src Top-3')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved metrics plot to: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main domain adaptation workflow
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Domain adaptation for BeMamba")
    parser.add_argument("--checkpoint", required=True, help="Path to source model checkpoint")
    parser.add_argument("--source-csv", required=True, help="Source domain CSV file")
    parser.add_argument("--target-csv", required=True, help="Target domain CSV file")
    parser.add_argument("--strategy", default="adapter-only", 
                       choices=["full", "adapter-only", "progressive"],
                       help="Adaptation strategy")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--device", default=config.DEVICE)
    parser.add_argument("--run-name", required=True, help="Name for this adaptation run")
    parser.add_argument("--use-ewc", action="store_true", help="Enable Elastic Weight Consolidation")
    parser.add_argument("--ewc-lambda", type=float, default=0.4, 
                       help="EWC regularization strength (default 0.4)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs)")
    
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}\n")

    # ── Load source model ─────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model, src_metadata = load_source_model(args.checkpoint, device)
    
    # ── LOAD SOURCE DATA (for Fisher) ────────────────────────────────────
    src_train, src_val, _, _ = build_datasets(args.source_csv)

    src_loader = DataLoader(
        src_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # ── Load target domain data ───────────────────────────────────────────
    print(f"\nLoading target domain: {args.target_csv}")
    train_ds, val_ds, test_ds, tgt_modalities = build_datasets(args.target_csv)
    
    # Verify modality compatibility
    missing_mods = [m for m, active in src_metadata["source_modalities"].items() 
                    if active and not tgt_modalities.get(m, False)]
    if missing_mods:
        raise RuntimeError(
            f"Target domain missing modalities: {missing_mods}. "
            f"Source requires: {src_metadata['source_modalities']}"
        )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    
    # ── Setup Elastic Weight Consolidation (if enabled) ───────────────────
    ewc = None
    if args.use_ewc:
        print(f"\n[EWC] Initializing (lambda={args.ewc_lambda})")

        ewc = ElasticWeightConsolidation(
            model,
            device,
            fisher_lambda=args.ewc_lambda
        )

        # Optional caching
        fisher_path = os.path.join(models_dir,"fisher_cache.pt")

        if os.path.exists(fisher_path):
            print("[EWC] Loading cached Fisher matrix...")
            ewc.fisher_matrix = torch.load(fisher_path)
        else:
            ewc.compute_fisher_information(model, src_loader)
            torch.save(ewc.fisher_matrix, fisher_path)
    
    # ── Setup training ────────────────────────────────────────────────────
    print(f"\n[Adaptation Strategy: {args.strategy}]")
    
    if args.strategy == "full":
        unfreeze_all(model)
    elif args.strategy == "adapter-only":
        unfreeze_adapter_and_classifier(model)
    elif args.strategy == "progressive":
        # Start with adapter-only; will unfreeze more layers later
        unfreeze_adapter_and_classifier(model)
    
    print_trainable_params(model)
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # ── Initialize history ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.run_name}" if args.run_name else f"adapt_{timestamp}"
    
    run_dir = os.path.join(config.CHECKPOINT_DIR, run_id)
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    history = {
        "run_id": run_id,
        "source_scenario": src_metadata["source_scenario"],
        "target_scenario": os.path.basename(os.path.dirname(args.target_csv)),
        "timestamp": timestamp,
        "strategy": args.strategy,
        "use_ewc": args.use_ewc,
        "ewc_config": ewc.get_config() if ewc else None,
        "hyperparameters": {
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": config.WEIGHT_DECAY,
            "ewc_lambda": args.ewc_lambda if args.use_ewc else None,
            "source_modalities": src_metadata["source_modalities"],
            "target_modalities": tgt_modalities,
        },
        "epochs": []
    }
    
    # ── Training loop ─────────────────────────────────────────────────────
    best_val_top3 = 0.0
    
    patience = args.patience
    no_improve_epochs = 0
    
    for epoch in range(1, args.epochs + 1):
        # Progressive unfreezing: unfreeze MSM at halfway point
        if args.strategy == "progressive" and epoch == args.epochs // 2:
            print(f"\n[Epoch {epoch}] Unfreezing MSM layer for progressive adaptation...")
            for param in model.msm.parameters():
                param.requires_grad = True
            print_trainable_params(model)
        
        train_loss, train_top1 = train_epoch(model, train_loader, optimizer, criterion, device, ewc=ewc)
        val_metrics = evaluate(model, val_loader, device)
        src_metrics = evaluate(model, src_loader, device)
        scheduler.step()
        
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_top1": val_metrics["top1"],
            "val_top3": val_metrics["top3"],
            "val_top5": val_metrics["top5"],
            "src_top1": src_metrics["top1"],
            "src_top3": src_metrics["top3"],
            "src_top5": src_metrics["top5"],
            "lr": optimizer.param_groups[0]["lr"]
        }
        history["epochs"].append(epoch_data)
        
        print(
            f"Epoch {epoch:02d}/{args.epochs}  "
            f"loss={train_loss:.4f}  "
            f"train_top1={train_top1:.4f}  "
            f"val_top1={val_metrics['top1']:.4f}  "
            f"val_top3={val_metrics['top3']:.4f}  "
            f"src_top1={src_metrics['top1']:.4f}"
        )
        
        # Save best checkpoint
        if val_metrics["top3"] > best_val_top3:
            best_val_top3 = val_metrics["top3"]
            best_ckpt = os.path.join(models_dir, "best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "modalities": src_metadata["source_modalities"],
                "epoch": epoch,
                "run_id": run_id,
                "source_scenario": src_metadata["source_scenario"],
                "target_scenario": history["target_scenario"],
                "best_val_top3": best_val_top3,
                "strategy": args.strategy,
            }, best_ckpt)
            print(f"  ✓ Saved best (top-3={best_val_top3:.4f})")
        else:
            no_improve_epochs += 1
            print(f"  No improvement for {no_improve_epochs} epoch(s)")
            
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break
    
    # ── Test evaluation ───────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation...")
    best_ckpt = os.path.join(models_dir, "best.pt")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    
    test_metrics = evaluate(model, test_loader, device)
    
    history["test_results"] = test_metrics
    history["best_epoch"] = ckpt["epoch"]
    history["best_val_top3"] = ckpt["best_val_top3"]
    
    print("\n═══════════ TEST RESULTS ═══════════")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("════════════════════════════════════")
    
    # ── Save results ──────────────────────────────────────────────────────
    metrics_path = os.path.join(run_dir, "adaptation_metrics.png")
    plot_adaptation_metrics(history, metrics_path)
    
    history_path = os.path.join(run_dir, "history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    summary = {
        "run_id": run_id,
        "source_scenario": src_metadata["source_scenario"],
        "target_scenario": history["target_scenario"],
        "strategy": args.strategy,
        "best_val_top3": history["best_val_top3"],
        "test_top1": test_metrics["top1"],
        "test_top3": test_metrics["top3"],
        "test_top5": test_metrics["top5"],
        "hyperparameters": history["hyperparameters"],
    }
    
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDomain adaptation complete! Results saved to: {run_dir}")
    print(f"  Models: {models_dir} (best.pt)")
    print(f"  Metrics: {metrics_path}")
    print(f"  History: {history_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
