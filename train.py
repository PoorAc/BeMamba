"""
train.py
────────
Training and evaluation loop for BeMamba.

Usage:
    python train.py
"""

import os
import json
import random
import numpy as np
from sched import scheduler
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
    DEVICE, DATASET_ROOT, SEED,
)
from dataset import build_datasets
from models import BeMamba

# ─────────────────────────────────────────────────────────────────────────────
# Set random seeds for reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility (can reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
def apply_modality_dropout(batch, active_modalities, p):
    """
    Independently drops each modality with probability p.
    """
    if p <= 0:
        return batch

    for mod in active_modalities:
        if mod in batch and torch.rand(1).item() < p:
            batch[mod] = torch.zeros_like(batch[mod])

    return batch

def train_epoch(model, loader, optimizer, criterion, scheduler, scaler, epoch):
    model.train()
    total_loss = total_acc = 0.0

    for batch_idx, batch in enumerate(tqdm(loader, desc="  train", leave=False)):
        batch  = to_device(batch)
        labels = batch["label"]
        
        # Apply modality dropout to the input batch before feeding it to the model
        batch = apply_modality_dropout(batch, model.active, MODALITY_DROPOUT)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(batch)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()

        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        scheduler.step(epoch + batch_idx / len(loader))

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

    correct = {k: 0 for k in TOP_K}
    total = 0

    for batch in tqdm(loader, desc="  eval ", leave=False):
        batch = to_device(batch)
        logits = model(batch)
        labels = batch["label"]

        for k in TOP_K:
            topk = logits.topk(k, dim=1)[1]
            correct[k] += (topk == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.size(0)

    return {f"top{k}": correct[k] / total for k in TOP_K}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train a new BeMamba model on a dataset scenario.")
    parser.add_argument("--run-name", help="Name of the training run", default=None)
    parser.add_argument("--patience", type=int, default=10, help="Number of epochs with no improvement to wait before early stopping")
    return parser.parse_args()

def main():
    
    set_seed(SEED)
    print(f"Device: {DEVICE}\n")
    
    args = parse_args()

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

    num_workers = min(8, os.cpu_count() or 1)
    pin_memory = DEVICE == "cuda"
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    g = torch.Generator()
    g.manual_seed(SEED)
        
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=g
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker,
        generator=g
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = BeMamba(
        modalities=modalities,
        d_model=D_MODEL, d_state=D_STATE, d_conv=D_CONV,
        expand=EXPAND, num_layers=NUM_LAYERS, num_beams=NUM_BEAMS,
        use_adapters=False  
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nBeMamba — {n_params:,} trainable parameters")
    print(f"Active modalities: {[k for k,v in modalities.items() if v]}\n")

    training_history["hyperparameters"]["num_params"] = n_params

    # ── Optimiser & schedule ──────────────────────────────────────────────
    trainable_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and "adapter" not in name.lower()
    ]

    optimizer = torch.optim.AdamW(trainable_params, lr=LR,
                                weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=SCHEDULER_T0, T_mult=SCHEDULER_T_MULT)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_top3 = 0.0
    
    # ── Training loop ─────────────────────────────────────────────────────
    
    patience = args.patience
    no_improve_epochs = 0
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_top1 = train_epoch(
                model, train_loader, optimizer, criterion, scheduler, scaler, epoch
            )
        val_m = evaluate(model, val_loader)

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
            no_improve_epochs = 0
            best_top3 = val_m["top3"]
            best_ckpt_path = os.path.join(models_dir, "best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "modalities": modalities,
                "epoch": epoch,
                "run_id": run_id,
                "scenario": scenario_name,
                "timestamp": timestamp,
                "best_val_top3": best_top3
            }, best_ckpt_path)
            print(f"  ✓ Saved best checkpoint (top-3={best_top3:.4f})")
        else:
            no_improve_epochs += 1
            print(f"  ✗ No improvement for {no_improve_epochs} epoch(s)")
        
        if no_improve_epochs >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
            break

    # Save final model
    final_ckpt_path = os.path.join(models_dir, "final.pt")
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
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