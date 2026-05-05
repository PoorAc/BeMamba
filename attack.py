"""
attack.py
─────────
Adversarial attacks on BeMamba for multimodal beamforming.

Five attacks are implemented, covering the full threat landscape:

  1. FGSM          — Fast Gradient Sign Method on images (white-box, 1-step)
  2. PGD           — Projected Gradient Descent on images (white-box, iterative)
  3. ModalDrop     — Zero-out one modality at inference (black-box ablation)
  4. GPSSpoof      — Replace GPS coordinates with adversarial values (white-box)
  5. FeatureNoise  — Gaussian noise injected into LiDAR / Radar feature vectors
                     after extraction, before temporal fusion (grey-box)

Usage:
    # Train first, then:
    python attack.py

All results are printed as a comparison table.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import (
    CSV_FILE, NUM_BEAMS, D_MODEL, D_STATE, D_CONV, EXPAND, NUM_LAYERS,
    BATCH_SIZE, TOP_K, BEST_CKPT, DEVICE,
    ATK_EPS, ATK_ALPHA, ATK_STEPS, ATK_RANDOM_START,
    MODAL_DROP_FRAC,
)
from dataset import build_datasets
from models import BeMamba
from train import topk_accuracy, to_device


# ─────────────────────────────────────────────────────────────────────────────
# Utility: load saved model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, modalities: dict) -> BeMamba:
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = BeMamba(
        modalities=modalities,
        d_model=D_MODEL, d_state=D_STATE, d_conv=D_CONV,
        expand=EXPAND, num_layers=NUM_LAYERS, num_beams=NUM_BEAMS,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Utility: evaluate top-k on a loader, optionally applying a perturbation fn
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _clean_eval(model, loader) -> dict:
    """Standard clean evaluation — no gradient needed."""
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc="  clean", leave=False):
        batch  = to_device(batch)
        logits = model(batch)
        all_logits.append(logits.cpu())
        all_labels.append(batch["label"].cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return {f"top{k}": topk_accuracy(logits, labels, k) for k in TOP_K}


# ─────────────────────────────────────────────────────────────────────────────
# Attack 1 — FGSM  (Fast Gradient Sign Method on image modality)
# ─────────────────────────────────────────────────────────────────────────────

def attack_fgsm(model: BeMamba, loader: DataLoader,
                eps: float = ATK_EPS) -> dict:
    """
    Single-step L-inf FGSM attack on the image modality.

    x_adv = x + eps · sign(∇_x L(f(x), y))

    Gradients are computed with respect to the image tensor; all other
    modalities (GPS, LiDAR, Radar) are passed through unperturbed.

    Threat model : white-box, attacker can observe model weights and gradients.
    """
    criterion  = nn.CrossEntropyLoss()
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc="  FGSM ", leave=False):
        batch  = to_device(batch)
        labels = batch["label"]

        if "images" not in batch:
            # Modality absent — fall back to clean forward pass
            with torch.no_grad():
                all_logits.append(model(batch).cpu())
            all_labels.append(labels.cpu())
            continue

        images = batch["images"].clone().requires_grad_(True)

        # Forward with perturb-able images
        adv_batch        = {k: v for k, v in batch.items()}
        adv_batch["images"] = images
        logits           = model(adv_batch)
        loss             = criterion(logits, labels)
        loss.backward()

        # FGSM step — clamp to valid normalised range [-3, 3] (approx)
        grad       = images.grad.detach().sign()
        adv_images = (images.detach() + eps * grad).clamp(-3.0, 3.0)

        with torch.no_grad():
            adv_batch["images"] = adv_images
            adv_logits          = model(adv_batch)

        all_logits.append(adv_logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return {f"top{k}": topk_accuracy(logits, labels, k) for k in TOP_K}


# ─────────────────────────────────────────────────────────────────────────────
# Attack 2 — PGD  (Projected Gradient Descent on image modality)
# ─────────────────────────────────────────────────────────────────────────────

def attack_pgd(model: BeMamba, loader: DataLoader,
               eps: float   = ATK_EPS,
               alpha: float = ATK_ALPHA,
               steps: int   = ATK_STEPS,
               random_start: bool = ATK_RANDOM_START) -> dict:
    """
    Iterative PGD attack (Madry et al., 2018) on the image modality.

    Begins from a random point in the epsilon-ball and takes `steps`
    gradient ascent steps, projecting back after each step.

    x_0    = x + U[-eps, eps]   (if random_start)
    x_{t+1} = Π_{B(x,eps)}( x_t + α · sign(∇_x L) )

    Threat model : white-box, strongest standard first-order attack.
    """
    criterion  = nn.CrossEntropyLoss()
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc="  PGD  ", leave=False):
        batch  = to_device(batch)
        labels = batch["label"]

        if "images" not in batch:
            with torch.no_grad():
                all_logits.append(model(batch).cpu())
            all_labels.append(labels.cpu())
            continue

        orig = batch["images"].clone()

        # Random initialisation inside the epsilon ball
        delta = torch.zeros_like(orig)
        if random_start:
            delta = delta.uniform_(-eps, eps)
        delta.requires_grad_(True)

        for _ in range(steps):
            adv_batch = {k: v for k, v in batch.items()}
            adv_batch["images"] = (orig + delta).clamp(-3.0, 3.0)
            logits = model(adv_batch)
            loss   = criterion(logits, labels)
            loss.backward()

            with torch.no_grad():
                delta  = delta + alpha * delta.grad.sign()
                delta  = delta.clamp(-eps, eps)          # project back to ball
            delta  = delta.detach().requires_grad_(True)

        with torch.no_grad():
            adv_batch = {k: v for k, v in batch.items()}
            adv_batch["images"] = (orig + delta.detach()).clamp(-3.0, 3.0)
            adv_logits          = model(adv_batch)

        all_logits.append(adv_logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return {f"top{k}": topk_accuracy(logits, labels, k) for k in TOP_K}


# ─────────────────────────────────────────────────────────────────────────────
# Attack 3 — ModalDrop  (zero-out one entire modality)
# ─────────────────────────────────────────────────────────────────────────────

def attack_modal_drop(model: BeMamba, loader: DataLoader,
                      drop_modality: str) -> dict:
    """
    Modality suppression attack: zero out all data for `drop_modality`.

    Models the scenario where an adversary physically blocks or jams one
    sensor (e.g., laser jamming of LiDAR, GPS spoofing to all-zeros, spray
    paint on cameras).

    Threat model : black-box physical adversary who can disrupt one sensor.
    If the target modality is not active in this dataset, clean accuracy
    is returned unchanged.
    """
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Drop-{drop_modality:<6}", leave=False):
            batch  = to_device(batch)
            labels = batch["label"]

            adv_batch = {k: v for k, v in batch.items()}
            if drop_modality in adv_batch:
                adv_batch[drop_modality] = torch.zeros_like(adv_batch[drop_modality])

            all_logits.append(model(adv_batch).cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return {f"top{k}": topk_accuracy(logits, labels, k) for k in TOP_K}


# ─────────────────────────────────────────────────────────────────────────────
# Attack 4 — GPSSpoof  (adversarial GPS perturbation)
# ─────────────────────────────────────────────────────────────────────────────

def attack_gps_spoof(model: BeMamba, loader: DataLoader,
                     eps: float = 0.5) -> dict:
    """
    White-box adversarial perturbation of the GPS (latitude, longitude) input.

    The GPS stream is a low-dimensional (T, 2) signal.  We apply a
    single FGSM step directly in the normalised GPS space with a larger
    relative budget (default eps=0.5 standard deviations) to simulate a
    GPS spoofing device that shifts the reported position.

    Threat model : white-box; realistic because GPS signals are unauthenticated
                   and can be spoofed with commodity SDR hardware.
    """
    criterion  = nn.CrossEntropyLoss()
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc="  GPS-spoof", leave=False):
        batch  = to_device(batch)
        labels = batch["label"]

        if "gps" not in batch:
            with torch.no_grad():
                all_logits.append(model(batch).cpu())
            all_labels.append(labels.cpu())
            continue

        gps = batch["gps"].clone().requires_grad_(True)

        adv_batch        = {k: v for k, v in batch.items()}
        adv_batch["gps"] = gps
        logits           = model(adv_batch)
        loss             = criterion(logits, labels)
        loss.backward()

        adv_gps = (gps.detach() + eps * gps.grad.detach().sign())

        with torch.no_grad():
            adv_batch["gps"] = adv_gps
            adv_logits       = model(adv_batch)

        all_logits.append(adv_logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return {f"top{k}": topk_accuracy(logits, labels, k) for k in TOP_K}


# ─────────────────────────────────────────────────────────────────────────────
# Attack 5 — FeatureNoise  (grey-box noise on LiDAR / Radar features)
# ─────────────────────────────────────────────────────────────────────────────

class _HookedBeMamba(nn.Module):
    """
    Wraps BeMamba and injects additive Gaussian noise into the extracted
    features of a target modality BEFORE the TimeSequenceMamba layer.

    This simulates an attack on the feature-level representation (e.g. a
    compromised perception module or adversarial point-cloud perturbations
    that shift the voxelised grid's feature distribution).

    Threat model : grey-box — attacker knows the feature extractor architecture
                   but not the full model weights.
    """

    def __init__(self, model: BeMamba, target: str, noise_std: float):
        super().__init__()
        self.model      = model
        self.target     = target
        self.noise_std  = noise_std

    def forward(self, batch: dict) -> torch.Tensor:
        mod   = self.model
        mods  = mod.modalities
        feats = []

        def _noisy_seq(ext, tsm, data, reshape_fn):
            B, T   = data.shape[:2]
            flat   = reshape_fn(data, B, T)
            f      = ext(flat).view(B, T, -1)
            if self.target in ("lidar", "radar"):
                f  = f + torch.randn_like(f) * self.noise_std
            return tsm(f)

        if mods.get("image") and "images" in batch:
            B, T, C, H, W = batch["images"].shape
            flat  = batch["images"].view(B * T, C, H, W)
            f     = mod.image_ext(flat).view(B, T, -1)
            feats.append(mod.image_tsm(f))

        if mods.get("gps") and "gps" in batch:
            B, T, _ = batch["gps"].shape
            flat    = batch["gps"].view(B * T, 2)
            f       = mod.gps_ext(flat).view(B, T, -1)
            feats.append(mod.gps_tsm(f))

        if mods.get("lidar") and "lidar" in batch:
            B, T, D, H, W = batch["lidar"].shape
            flat  = batch["lidar"].view(B * T, D, H, W)
            f     = mod.lidar_ext(flat).view(B, T, -1)
            if self.target == "lidar":
                f = f + torch.randn_like(f) * self.noise_std
            feats.append(mod.lidar_tsm(f))

        if mods.get("radar") and "radar" in batch:
            B, T, _, H, W = batch["radar"].shape
            flat  = batch["radar"].view(B * T, 1, H, W)
            f     = mod.radar_ext(flat).view(B, T, -1)
            if self.target == "radar":
                f = f + torch.randn_like(f) * self.noise_std
            feats.append(mod.radar_tsm(f))

        fused  = mod.msm(feats)
        return mod.classifier(fused)


def attack_feature_noise(model: BeMamba, loader: DataLoader,
                         target: str = "lidar",
                         noise_std: float = 1.0) -> dict:
    """
    Inject zero-mean Gaussian noise (σ = noise_std) into the extracted feature
    vectors of the `target` modality before temporal fusion.

    If the target modality is not active, clean accuracy is returned.
    """
    if not model.modalities.get(target):
        print(f"  [FeatureNoise] '{target}' not active — returning clean eval.")
        return _clean_eval(model, loader)

    hooked = _HookedBeMamba(model, target=target, noise_std=noise_std)
    hooked.eval()

    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  FeatNoise-{target}", leave=False):
            batch  = to_device(batch)
            logits = hooked(batch)
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return {f"top{k}": topk_accuracy(logits, labels, k) for k in TOP_K}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(metrics: dict) -> str:
    return "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())


def main():
    print(f"Device: {DEVICE}\n")

    # ── Load datasets ─────────────────────────────────────────────────────
    _, _, test_ds, modalities = build_datasets(CSV_FILE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)

    # ── Load model ────────────────────────────────────────────────────────
    model = load_model(BEST_CKPT, modalities)
    active = [k for k, v in modalities.items() if v]
    print(f"Active modalities: {active}\n")

    results = {}

    # ── 0. Clean baseline ─────────────────────────────────────────────────
    print("Running clean evaluation...")
    results["Clean"] = _clean_eval(model, test_loader)
    print(f"  Clean       : {_fmt(results['Clean'])}")

    # ── 1. FGSM ──────────────────────────────────────────────────────────
    if modalities.get("image"):
        print(f"\nAttack 1 — FGSM (eps={ATK_EPS:.4f})")
        results["FGSM"] = attack_fgsm(model, test_loader)
        print(f"  FGSM        : {_fmt(results['FGSM'])}")

    # ── 2. PGD ───────────────────────────────────────────────────────────
    if modalities.get("image"):
        print(f"\nAttack 2 — PGD (eps={ATK_EPS:.4f}, α={ATK_ALPHA:.4f}, steps={ATK_STEPS})")
        results["PGD"] = attack_pgd(model, test_loader)
        print(f"  PGD         : {_fmt(results['PGD'])}")

    # ── 3. ModalDrop — one attack per active modality ─────────────────────
    for mod_name in active:
        print(f"\nAttack 3 — ModalDrop [{mod_name}]")
        key = f"Drop-{mod_name}"
        results[key] = attack_modal_drop(model, test_loader, drop_modality=mod_name)
        print(f"  {key:<14}: {_fmt(results[key])}")

    # ── 4. GPS Spoof ──────────────────────────────────────────────────────
    if modalities.get("gps"):
        print("\nAttack 4 — GPS Spoof (eps=0.5 std)")
        results["GPS-Spoof"] = attack_gps_spoof(model, test_loader)
        print(f"  GPS-Spoof   : {_fmt(results['GPS-Spoof'])}")

    # ── 5. FeatureNoise on LiDAR / Radar ─────────────────────────────────
    for mod_name in ("lidar", "radar"):
        if modalities.get(mod_name):
            print(f"\nAttack 5 — FeatureNoise [{mod_name}] (σ=1.0)")
            key = f"FeatNoise-{mod_name}"
            results[key] = attack_feature_noise(model, test_loader,
                                                 target=mod_name, noise_std=1.0)
            print(f"  {key:<14}: {_fmt(results[key])}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print(f"{'Attack':<20}  {'top1':>8}  {'top3':>8}  {'top5':>8}")
    print("─" * 62)
    clean_top1 = results["Clean"]["top1"]
    for name, m in results.items():
        drop = clean_top1 - m["top1"]
        flag = f"  ▼{drop:.4f}" if drop > 0.005 else ""
        print(f"{name:<20}  {m['top1']:>8.4f}  {m['top3']:>8.4f}  {m['top5']:>8.4f}{flag}")
    print("═" * 62)


if __name__ == "__main__":
    main()