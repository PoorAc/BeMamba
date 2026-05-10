"""
cross_infer.py
────────────────
Load a saved BeMamba checkpoint and evaluate it on a chosen dataset split.

Usage:
    python cross_infer.py --checkpoint .\checkpoints\31_nocross\models\best.pt \
        --csv .\Scenarios\32\32.csv --dataset-root .\Scenarios\32 --split test

The script updates config paths for the provided dataset root, reloads the
dataset module so file-location constants are refreshed, and evaluates the
saved model on any DeepSense dataset split.
"""

import argparse
import importlib
import json
import os
import sys

import matplotlib.pyplot as plt
import torch

import config


def update_dataset_paths(dataset_root: str, csv_file: str):
    """Update config paths and reload modules so dataset paths are refreshed."""
    if dataset_root:
        config.DATASET_ROOT = os.path.abspath(dataset_root)
        config.IMAGE_DIR = os.path.join(config.DATASET_ROOT, "unit1", "camera_data")
        config.LIDAR_DIR = os.path.join(config.DATASET_ROOT, "unit1", "lidar_data")
        config.RADAR_DIR = os.path.join(config.DATASET_ROOT, "unit1", "mmWave_data")

    if csv_file:
        config.CSV_FILE = os.path.abspath(csv_file)
        if not dataset_root:
            config.DATASET_ROOT = os.path.dirname(config.CSV_FILE)

    # Reload dataset and train modules so they pick up the updated config values.
    import dataset
    import train
    importlib.reload(config)
    importlib.reload(dataset)
    importlib.reload(train)
    return dataset, train


def load_checkpoint(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "modalities" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("Checkpoint must contain 'modalities' and 'model_state'.")

    from models import BeMamba

    model = BeMamba(
        modalities=checkpoint["modalities"],
        d_model=config.D_MODEL,
        num_beams=config.NUM_BEAMS,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def evaluate(model, loader, top_k):
    device = next(model.parameters()).device
    all_logits, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        logits = model(batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"].cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    results = {}
    for k in top_k:
        _, topk = all_logits.topk(k, dim=1)
        correct = topk.eq(all_labels.unsqueeze(1).expand_as(topk))
        results[f"top{k}"] = correct.any(dim=1).float().mean().item()
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved BeMamba model on a dataset split.")
    parser.add_argument("--checkpoint", required=True, help="Path to the saved model checkpoint (.pt)")
    parser.add_argument("--csv", required=True, help="Path to the dataset CSV file to evaluate")
    parser.add_argument("--dataset-root", required=False, help="Root folder of the dataset (used to resolve modality directories)")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--device", default=config.DEVICE, help="Device for evaluation")
    parser.add_argument("--output", help="Optional path to save JSON results")
    return parser.parse_args()

import json
import torch
import numpy as np

def make_json_serializable(obj):
    """Recursively convert tensors/arrays to JSON-safe Python types."""
    
    if isinstance(obj, torch.Tensor):
        # scalar tensor
        if obj.numel() == 1:
            return obj.item()
        # multi-value tensor
        return obj.detach().cpu().tolist()

    elif isinstance(obj, np.ndarray):
        return obj.tolist()

    elif isinstance(obj, np.generic):
        return obj.item()

    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    elif isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]

    return obj

def save_results(output_dir: str, summary: dict, split: str):
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"{split}_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(summary), f, indent=2)

    plt.figure(figsize=(6, 4))
    keys = [k for k in summary["results"] if k.startswith("top")]
    values = [summary["results"][k] for k in keys]
    plt.bar(keys, values, color=["#2a9d8f", "#e9c46a", "#f4a261"])
    plt.ylim(0, 1)
    plt.xlabel("Metric")
    plt.ylabel("Accuracy")
    plt.title(f"Cross Inference: {summary['split'].title()} Accuracy")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{split}_results.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    return json_path, plot_path


def main():
    args = parse_args()

    if args.dataset_root is None:
        args.dataset_root = os.path.dirname(os.path.abspath(args.csv))

    dataset_module, _ = update_dataset_paths(args.dataset_root, args.csv)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)

    train_ds, val_ds, test_ds, detected_modalities = dataset_module.build_datasets(args.csv)
    needed_modalities = checkpoint["modalities"]
    missing_modalities = [m for m, active in needed_modalities.items() if active and not detected_modalities.get(m, False)]
    if missing_modalities:
        raise RuntimeError(
            f"Dataset is missing modalities required by checkpoint: {missing_modalities}. "
            f"Detected modalities: {detected_modalities}"
        )

    split_ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]

    loader = torch.utils.data.DataLoader(
        split_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    results = evaluate(model, loader, config.TOP_K)

    run_id = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint))))
    dataset_name = os.path.basename(os.path.normpath(args.dataset_root))
    output_dir = os.path.join("cross_inference", run_id + "_" + dataset_name)

    summary = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "csv": os.path.abspath(args.csv),
        "dataset_root": os.path.abspath(args.dataset_root),
        "split": args.split,
        "modalities": detected_modalities,
        "results": results,
        "checkpoint_metadata": {k: v for k, v in checkpoint.items() if k != "model_state"},
        "output_dir": os.path.abspath(output_dir),
    }

    json_path, plot_path = save_results(output_dir, summary, args.split)
    print(f"Saved evaluation JSON to: {json_path}")
    print(f"Saved accuracy plot to: {plot_path}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Also saved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
