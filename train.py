"""
train.py
────────
Training and evaluation loop for BeMamba.

Usage:
    python train.py
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from config import (
    CSV_FILE, NUM_BEAMS, D_MODEL, D_STATE, D_CONV, EXPAND, NUM_LAYERS,
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, TOP_K, CHECKPOINT_DIR,
    MODALITY_DROPOUT, SCHEDULER_T0, SCHEDULER_T_MULT,
    DEVICE, DATASET_ROOT,
)
from dataset import build_datasets
from models import BeMamba


# ─────────────────────────────────────────────────────────────────────────────
# Visualization utilities
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_metrics(training_history: dict, save_path: str):
    """
    Create and save plots of training metrics.

    Args:
        training_history: Dict containing epochs data
        save_path: Path to save the plot image
    """
    epochs = [epoch['epoch'] for epoch in training_history['epochs']]
    train_loss = [epoch['train_loss'] for epoch in training_history['epochs']]
    train_top1 = [epoch['train_top1'] for epoch in training_history['epochs']]
    val_top1 = [epoch['val_top1'] for epoch in training_history['epochs']]
    val_top3 = [epoch['val_top3'] for epoch in training_history['epochs']]
    val_top5 = [epoch['val_top5'] for epoch in training_history['epochs']]
    lr_values = [epoch['lr'] for epoch in training_history['epochs']]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Metrics - {training_history["scenario"]} ({training_history["run_id"]})', fontsize=14)

    # Training Loss
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Training Accuracy
    ax2.plot(epochs, train_top1, 'g-', linewidth=2, label='Train Top-1')
    ax2.plot(epochs, val_top1, 'g--', linewidth=2, label='Val Top-1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Top-1 Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Top-3 and Top-5 Accuracy
    ax3.plot(epochs, val_top3, 'orange', linewidth=2, label='Val Top-3')
    ax3.plot(epochs, val_top5, 'red', linewidth=2, label='Val Top-5')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Top-3 & Top-5 Validation Accuracy')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Learning Rate
    ax4.plot(epochs, lr_values, 'purple', linewidth=2, label='Learning Rate')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Schedule')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Training metrics plot saved to: {save_path}")

def load_model(checkpoint_path: str, device: str = None):
    """
    Load a trained BeMamba model from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load model on (defaults to DEVICE from config)

    Returns:
        model: Loaded BeMamba model
        metadata: Dict with checkpoint metadata
    """
    if device is None:
        device = DEVICE

    ckpt = torch.load(checkpoint_path, map_location=device)

    model = BeMamba(
        modalities=ckpt["modalities"],
        d_model=D_MODEL, d_state=D_STATE, d_conv=D_CONV,
        expand=EXPAND, num_layers=NUM_LAYERS, num_beams=NUM_BEAMS,
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    metadata = {
        "epoch": ckpt.get("epoch", "unknown"),
        "run_id": ckpt.get("run_id", "unknown"),
        "scenario": ckpt.get("scenario", "unknown"),
        "timestamp": ckpt.get("timestamp", "unknown"),
        "modalities": ckpt["modalities"]
    }

    return model, metadata

def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Fraction of samples where the true label appears in the top-k predictions."""
    _, topk   = logits.topk(k, dim=1)
    correct   = topk.eq(labels.unsqueeze(1).expand_as(topk))
    return correct.any(dim=1).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Batch → device helper  (handles variable modality keys gracefully)
# ─────────────────────────────────────────────────────────────────────────────

def to_device(batch: dict) -> dict:
    return {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


# ─────────────────────────────────────────────────────────────────────────────
# One training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = total_acc = 0.0

    for batch in tqdm(loader, desc="  train", leave=False):
        batch  = to_device(batch)
        labels = batch["label"]

        # Random modality dropout for better generalization across gaps.
        if MODALITY_DROPOUT > 0 and len(model.active) > 1:
            if torch.rand(1).item() < MODALITY_DROPOUT:
                drop_mod = model.active[torch.randint(len(model.active), (1,)).item()]
                if drop_mod in batch:
                    batch[drop_mod] = torch.zeros_like(batch[drop_mod])

        optimizer.zero_grad()
        logits = model(batch)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc  += topk_accuracy(logits.detach(), labels, 1)

    n = len(loader)
    return total_loss / n, total_acc / n


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (no gradients)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader) -> dict:
    model.eval()
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc="  eval ", leave=False):
        batch  = to_device(batch)
        logits = model(batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"].cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    return {f"top{k}": topk_accuracy(all_logits, all_labels, k) for k in TOP_K}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train a new BeMamba model on a dataset scenario.")
    parser.add_argument("--run-name", help="Name of the training run", default=None)
    return parser.parse_args()

def main():
    print(f"Device: {DEVICE}\n")
    
    args = parse_args()  # For compatibility with cross_infer.py; not used in this script

    # ── Setup run metadata ─────────────────────────────────────────────────
    scenario_name = os.path.basename(DATASET_ROOT)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_name if args.run_name else f"{scenario_name}_{timestamp}"

    # Create run-specific directories
    run_dir = os.path.join(CHECKPOINT_DIR, run_id)
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Initialize training history
    training_history = {
        "run_id": run_id,
        "scenario": scenario_name,
        "timestamp": timestamp,
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "modality_dropout": MODALITY_DROPOUT,
            "scheduler": "CosineAnnealingWarmRestarts",
            "scheduler_t0": SCHEDULER_T0,
            "scheduler_t_mult": SCHEDULER_T_MULT,
            "d_model": D_MODEL,
            "d_state": D_STATE,
            "d_conv": D_CONV,
            "expand": EXPAND,
            "num_layers": NUM_LAYERS,
            "seq_len": None,  # Will be set from dataset
        },
        "epochs": []
    }

    # ── Datasets ──────────────────────────────────────────────────────────
    print("Building datasets...")
    train_ds, val_ds, test_ds, modalities = build_datasets(CSV_FILE)

    # Update hyperparameters with dataset info
    training_history["hyperparameters"]["seq_len"] = train_ds.seq_len
    training_history["modalities"] = modalities

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = BeMamba(
        modalities=modalities,
        d_model=D_MODEL, d_state=D_STATE, d_conv=D_CONV,
        expand=EXPAND, num_layers=NUM_LAYERS, num_beams=NUM_BEAMS,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nBeMamba — {n_params:,} trainable parameters")
    print(f"Active modalities: {[k for k,v in modalities.items() if v]}\n")

    training_history["hyperparameters"]["num_params"] = n_params

    # ── Optimiser & schedule ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SCHEDULER_T0, T_mult=SCHEDULER_T_MULT)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_top3 = 0.0

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_top1 = train_epoch(model, train_loader, optimizer, criterion)
        val_m = evaluate(model, val_loader)
        scheduler.step()

        # Record epoch metrics (only loss and accuracies, no model saving per epoch)
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_top1": train_top1,
            "val_top1": val_m["top1"],
            "val_top3": val_m["top3"],
            "val_top5": val_m["top5"],
            "lr": optimizer.param_groups[0]["lr"]
        }
        training_history["epochs"].append(epoch_data)

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"loss={train_loss:.4f}  train_top1={train_top1:.4f}  "
            f"val_top1={val_m['top1']:.4f}  "
            f"val_top3={val_m['top3']:.4f}  "
            f"val_top5={val_m['top5']:.4f}"
        )

        # Save best model (only when validation improves)
        if val_m["top3"] > best_top3:
            best_top3 = val_m["top3"]
            best_ckpt_path = os.path.join(models_dir, "best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "modalities": modalities,
                "epoch": epoch,
                "run_id": run_id,
                "scenario": scenario_name,
                "timestamp": timestamp,
                "best_val_top3": best_top3
            }, best_ckpt_path)
            print(f"  ✓ Saved best checkpoint (top-3={best_top3:.4f})")

    # Save final model
    final_ckpt_path = os.path.join(models_dir, "final.pt")
    torch.save({
        "model_state": model.state_dict(),
        "modalities": modalities,
        "epoch": EPOCHS,
        "run_id": run_id,
        "scenario": scenario_name,
        "timestamp": timestamp
    }, final_ckpt_path)

    # ── Test evaluation ───────────────────────────────────────────────────
    print("\nLoading best checkpoint for final test evaluation...")
    best_ckpt_path = os.path.join(models_dir, "best.pt")
    ckpt  = torch.load(best_ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    test_m = evaluate(model, test_loader)

    # Record test results
    training_history["test_results"] = test_m
    training_history["best_epoch"] = ckpt["epoch"]
    training_history["best_val_top3"] = ckpt["best_val_top3"]

    print("\n═══════════ TEST RESULTS ═══════════")
    for k, v in test_m.items():
        print(f"  {k}: {v:.4f}")
    print("════════════════════════════════════")

    # Create training metrics visualization
    plot_path = os.path.join(run_dir, "training_metrics.png")
    plot_training_metrics(training_history, plot_path)

    # Save training history and metadata
    history_path = os.path.join(run_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)

    # Save run summary
    summary = {
        "run_id": run_id,
        "scenario": scenario_name,
        "timestamp": timestamp,
        "best_val_top3": training_history["best_val_top3"],
        "test_top1": test_m["top1"],
        "test_top3": test_m["top3"],
        "test_top5": test_m["top5"],
        "modalities": list(modalities.keys()),
        "active_modalities": [k for k, v in modalities.items() if v],
        "hyperparameters": training_history["hyperparameters"]
    }

    summary_path = os.path.join(run_dir, "run_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete! Results saved to: {run_dir}")
    print(f"- Models: {models_dir} (best.pt, final.pt)")
    print(f"- Metrics plot: {plot_path}")
    print(f"- History: {history_path}")
    print(f"- Summary: {summary_path}")


if __name__ == "__main__":
    main()