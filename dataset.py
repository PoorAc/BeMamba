"""
dataset.py
──────────
DeepSense 6G dataset loader with automatic modality detection.

Auto-detection logic
────────────────────
The loader inspects the CSV header and the filesystem to decide which
modalities to activate.  Rules per modality:

  Image  — CSV contains column(s) matching  "unit1_rgb"
           AND  IMAGE_DIR exists on disk.

  GPS    — CSV contains "unit2_gps_long" AND "unit2_gps_lat".

  LiDAR  — CSV contains column(s) matching  "unit1_lidar"
           AND  LIDAR_DIR exists on disk.

  Radar  — CSV contains column(s) matching  "unit1_radar"
           AND  RADAR_DIR exists on disk.

Any modality whose conditions are not fully met is silently skipped, so
the same code works unchanged across all DeepSense 6G scenarios regardless
of which sensors were active during collection.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    IMAGE_DIR, LIDAR_DIR, RADAR_DIR,
    SEQ_LEN, VAL_RATIO, TEST_RATIO, SEED,
    NUM_BEAMS,
    GPS_NOISE_STD, RADAR_NOISE_STD,
    LIDAR_VOXEL_H, LIDAR_VOXEL_W, LIDAR_VOXEL_D,
    RADAR_H, RADAR_W,
)

# ── Image pre-processing (ImageNet normalisation) ─────────────────────────────
IMG_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

IMG_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Modality detection helper
# ─────────────────────────────────────────────────────────────────────────────

def detect_modalities(df: pd.DataFrame) -> dict:
    """
    Inspect a loaded CSV DataFrame and the filesystem to determine which
    sensing modalities are available.

    Returns a dict:  {"image": bool, "gps": bool, "lidar": bool, "radar": bool}
    """
    cols = [c.lower() for c in df.columns]

    has_image = (
        any("unit1_rgb"   in c for c in cols) and os.path.isdir(IMAGE_DIR)
    )
    # GPS: check for either the standard column names OR the unit location columns
    has_gps   = (
        ("unit2_gps_long"  in cols and "unit2_gps_lat" in cols) or
        any("unit2_loc" in c for c in cols)
    )
    has_lidar = (
        any("unit1_lidar" in c for c in cols) and os.path.isdir(LIDAR_DIR)
    )
    has_radar = (
        any("unit1_radar" in c for c in cols) or
        any("unit1_pwr" in c for c in cols) or
        any("mmwave" in c for c in cols)
    ) and os.path.isdir(RADAR_DIR)

    detected = {
        "image": has_image,
        "gps":   has_gps,
        "lidar": has_lidar,
        "radar": has_radar,
    }

    active = [k for k, v in detected.items() if v]
    print("=== Modality auto-detection ===")
    for name, present in detected.items():
        status = "[+] active" if present else "[-] not found"
        print(f"  {name:<8} {status}")
    print(f"=== {len(active)} modality/ies active: {active} ===")

    if not active:
        raise RuntimeError(
            "No modalities detected. Check DATASET_ROOT, CSV_FILE, and "
            "directory paths in config.py."
        )

    return detected


# ─────────────────────────────────────────────────────────────────────────────
# LiDAR helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_lidar_file(path: str) -> np.ndarray:
    """
    Load a LiDAR point-cloud file and return a (D, H, W) voxel occupancy grid.

    Supports two common formats:
      .bin  — KITTI-style binary float32 array, shape (N, 4): x, y, z, intensity
      .npy  — numpy array, either (N, 4) point cloud or (D, H, W) pre-voxelised

    If the file is missing, returns a zero grid.
    """
    if not os.path.exists(path):
        return np.zeros((LIDAR_VOXEL_D, LIDAR_VOXEL_H, LIDAR_VOXEL_W),
                        dtype=np.float32)

    if path.endswith(".bin"):
        pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        return _voxelise(pts)

    if path.endswith(".npy"):
        data = np.load(path)
        if data.ndim == 2:              # (N, ≥3) — point cloud
            return _voxelise(data)
        if data.ndim == 3:              # (D, H, W) — pre-voxelised
            return _resize_voxel(data)
        raise ValueError(f"Unexpected LiDAR .npy shape: {data.shape}")

    raise ValueError(f"Unsupported LiDAR file format: {path}")


def _voxelise(pts: np.ndarray,
              x_range=(-50, 50), y_range=(-50, 50), z_range=(-3, 3)) -> np.ndarray:
    """
    Convert a point cloud array (N, ≥3) to a binary occupancy voxel grid.
    Points outside the given ranges are discarded.
    """
    D, H, W = LIDAR_VOXEL_D, LIDAR_VOXEL_H, LIDAR_VOXEL_W
    grid    = np.zeros((D, H, W), dtype=np.float32)

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    mask = (
        (x >= x_range[0]) & (x < x_range[1]) &
        (y >= y_range[0]) & (y < y_range[1]) &
        (z >= z_range[0]) & (z < z_range[1])
    )
    x, y, z = x[mask], y[mask], z[mask]

    xi = ((x - x_range[0]) / (x_range[1] - x_range[0]) * W).astype(int).clip(0, W - 1)
    yi = ((y - y_range[0]) / (y_range[1] - y_range[0]) * H).astype(int).clip(0, H - 1)
    zi = ((z - z_range[0]) / (z_range[1] - z_range[0]) * D).astype(int).clip(0, D - 1)

    grid[zi, yi, xi] = 1.0
    return grid


def _resize_voxel(grid: np.ndarray) -> np.ndarray:
    """Crop or zero-pad a pre-voxelised grid to (D, H, W)."""
    D, H, W  = LIDAR_VOXEL_D, LIDAR_VOXEL_H, LIDAR_VOXEL_W
    out      = np.zeros((D, H, W), dtype=np.float32)
    d = min(grid.shape[0], D)
    h = min(grid.shape[1], H)
    w = min(grid.shape[2], W)
    out[:d, :h, :w] = grid[:d, :h, :w]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Radar helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_radar_file(path: str) -> np.ndarray:
    """
    Load a radar / mmWave power file and return a tensor of shape
    (1, H, W). For 1D beam-power files, reshape to the nearest square.
    """
    if not os.path.exists(path):
        return np.zeros((1, RADAR_H, RADAR_W), dtype=np.float32)

    with open(path, 'r', encoding='utf-8') as f:
        values = [line.strip() for line in f if line.strip()]

    if not values:
        return np.zeros((1, RADAR_H, RADAR_W), dtype=np.float32)

    data = np.array([float(v) for v in values], dtype=np.float32)
    n = data.size

    if n == RADAR_H * RADAR_W:
        data = data.reshape(RADAR_H, RADAR_W)
    elif int(np.sqrt(n))**2 == n:
        side = int(np.sqrt(n))
        data = data.reshape(side, side)
    elif n == RADAR_H:
        data = data.reshape(1, RADAR_H)
    elif n == RADAR_W:
        data = data.reshape(RADAR_W, 1)
    else:
        # If the data length is a perfect square, use the natural square shape.
        side = int(np.sqrt(n))
        if side * side == n:
            data = data.reshape(side, side)
        elif side * (side + 1) == n:
            data = data.reshape(side, side + 1)
        elif (side + 1) * (side + 1) == n:
            data = data.reshape(side + 1, side + 1)
        else:
            padded = np.zeros(RADAR_H * RADAR_W, dtype=np.float32)
            padded[:min(n, RADAR_H * RADAR_W)] = data[:min(n, RADAR_H * RADAR_W)]
            data = padded.reshape(RADAR_H, RADAR_W)

    data = np.log1p(np.abs(data))
    dmin, dmax = data.min(), data.max()
    if dmax > dmin:
        data = (data - dmin) / (dmax - dmin)

    return data[np.newaxis]   # (1, H, W)


def _load_gps_file(path: str) -> np.ndarray:
    """Load a GPS text file and return [longitude, latitude]."""
    if not os.path.exists(path):
        return np.array([0.0, 0.0], dtype=np.float32)

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"Unexpected GPS file format: {path}")

    # Many DeepSense GPS files store latitude then longitude.
    lat = float(lines[0])
    lon = float(lines[1])
    return np.array([lon, lat], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class DeepSense6GDataset(Dataset):
    """
    Sliding-window temporal dataset for DeepSense 6G multi-modal beamforming.

    Each sample is a window of SEQ_LEN consecutive rows from one driving
    sequence.  The label is the optimal beam index at the last timestep.

    Parameters
    ──────────
    csv_path  : path to the scenario CSV file
    split     : "train" | "val" | "test"
    modalities: dict from detect_modalities() — shared across all splits so
                GPS normalisation stats are fit on the whole dataset once.
    norm_stats: dict of normalisation statistics (computed once, passed in)
    """

    def __init__(
        self,
        csv_path:   str,
        split:      str,
        modalities: dict,
        norm_stats: dict,
        seq_len:    int = SEQ_LEN,
    ):
        self.modalities = modalities
        self.norm_stats = norm_stats
        self.seq_len    = seq_len
        self.split      = split
        self.is_train   = split == "train"
        self.csv_path   = csv_path

        df = pd.read_csv(csv_path)

        # Create 'seq' column from 'seq_index' if it doesn't exist
        if "seq" not in df.columns:
            if "seq_index" in df.columns:
                df["seq"] = df["seq_index"]
            else:
                # Fallback: group by time_stamp or create a dummy sequence ID
                df["seq"] = 0

        # Column name caches (original case)
        self._img_col   = next((c for c in df.columns if "unit1_rgb"   in c.lower()), None)
        self._gps_col   = next((c for c in df.columns if "unit2_gps_long" in c.lower()), None)
        self._gps_lat   = next((c for c in df.columns if "unit2_gps_lat"  in c.lower()), None)
        self._gps_path  = None
        self._lidar_col = next((c for c in df.columns if "unit1_lidar" in c.lower()), None)
        self._radar_col = next((c for c in df.columns if "unit1_radar" in c.lower()), None)
        if self._radar_col is None:
            self._radar_col = next((c for c in df.columns if "unit1_pwr" in c.lower()), None)
        if self._radar_col is None:
            self._radar_col = next((c for c in df.columns if "mmwave" in c.lower()), None)

        if self.modalities["gps"] and self._gps_col is None:
            for col in df.columns:
                if col.lower() == "unit2_loc_cal":
                    self._gps_path = col
                    break
            if self._gps_path is None:
                self._gps_path = next((c for c in df.columns if "unit2_loc" in c.lower()), None)

        # ── Build sliding-window samples, grouped by sequence ─────────────
        all_samples = []
        for _, grp in df.groupby("seq"):
            grp = grp.reset_index(drop=True)
            for i in range(seq_len - 1, len(grp)):
                window = grp.iloc[i - seq_len + 1: i + 1]
                # Get beam index: try 'unit1_beam' first (1-based), then 'unit1_beam_index' (already 0-based)
                label = None
                if "unit1_beam" in grp.columns:
                    label = int(grp.iloc[i]["unit1_beam"]) - 1  # Convert 1-based to 0-based
                elif "unit1_beam_index" in grp.columns:
                    label = int(grp.iloc[i]["unit1_beam_index"])  # Already 0-based
                if label is not None and 0 <= label < NUM_BEAMS:
                    all_samples.append((window, label))

        # ── Sequence-level train / val / test split (no data leakage) ─────
        all_seqs  = list(df["seq"].unique())
        train_s, tmp = train_test_split(
            all_seqs, test_size=VAL_RATIO + TEST_RATIO, random_state=SEED
        )
        val_s, test_s = train_test_split(
            tmp,
            test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
            random_state=SEED,
        )
        seq_set = {"train": set(train_s), "val": set(val_s), "test": set(test_s)}[split]

        self.samples = [
            (w, l) for w, l in all_samples if w.iloc[0]["seq"] in seq_set
        ]
        print(f"  [{split:<5}] {len(self.samples):>5} samples")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _load_image_seq(self, window) -> torch.Tensor:
        imgs = []
        transform = IMG_TRAIN_TRANSFORM if self.is_train else IMG_EVAL_TRANSFORM

        for _, row in window.iterrows():
            path = os.path.join(IMAGE_DIR, str(row[self._img_col]))
            img  = Image.open(path).convert("RGB") if os.path.exists(path) \
                   else Image.new("RGB", (224, 224))
            imgs.append(transform(img))
        return torch.stack(imgs)   # (T, 3, 224, 224)

    def _load_gps_seq(self, window) -> torch.Tensor:
        if self._gps_col and self._gps_lat:
            gps = window[[self._gps_col, self._gps_lat]].values.astype(np.float32)
        elif self._gps_path is not None:
            root = os.path.dirname(os.path.abspath(self.csv_path)) if hasattr(self, 'csv_path') else None
            coords = []
            for _, row in window.iterrows():
                path = str(row[self._gps_path])
                full_path = os.path.join(root, path) if root is not None else path
                coords.append(_load_gps_file(full_path))
            gps = np.stack(coords, axis=0)
        else:
            gps = np.zeros((len(window), 2), dtype=np.float32)

        # Disable noise injection during base training (enable only for cross-inference robustness)
        # if self.is_train and GPS_NOISE_STD > 0:
        #     gps += np.random.normal(0, GPS_NOISE_STD, size=gps.shape).astype(np.float32)

        mean = self.norm_stats["gps_mean"]
        std  = self.norm_stats["gps_std"]
        return torch.tensor((gps - mean) / std, dtype=torch.float32)  # (T, 2)

    def _load_lidar_seq(self, window) -> torch.Tensor:
        voxels = []
        for _, row in window.iterrows():
            path   = os.path.join(LIDAR_DIR, str(row[self._lidar_col]))
            voxels.append(_load_lidar_file(path))
        arr = np.stack(voxels)   # (T, D, H, W)
        return torch.tensor(arr, dtype=torch.float32)

    def _load_radar_seq(self, window) -> torch.Tensor:
        maps = []
        for _, row in window.iterrows():
            path = os.path.join(RADAR_DIR, str(row[self._radar_col]))
            maps.append(_load_radar_file(path))
        arr = np.stack(maps)   # (T, 1, H, W)

        # Disable noise injection during base training (enable only for cross-inference robustness)
        # if self.is_train and RADAR_NOISE_STD > 0:
        #     arr = arr + np.random.normal(0, RADAR_NOISE_STD, size=arr.shape).astype(np.float32)
        #     arr = np.clip(arr, 0.0, 1.0)

        return torch.tensor(arr, dtype=torch.float32)

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window, label = self.samples[idx]
        out = {"label": torch.tensor(label, dtype=torch.long)}

        if self.modalities["image"]:
            out["images"] = self._load_image_seq(window)

        if self.modalities["gps"]:
            out["gps"] = self._load_gps_seq(window)

        if self.modalities["lidar"]:
            out["lidar"] = self._load_lidar_seq(window)

        if self.modalities["radar"]:
            out["radar"] = self._load_radar_seq(window)

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Factory — call this once to get all three splits
# ─────────────────────────────────────────────────────────────────────────────

def build_datasets(csv_path: str):
    """
    Load the CSV, detect modalities, compute normalisation stats, and
    return (train_ds, val_ds, test_ds, modalities_dict).

    This is the single entry point used by train.py and attack.py.
    """
    df = pd.read_csv(csv_path)
    modalities = detect_modalities(df)

    # Compute GPS normalisation stats on the whole dataset (no leakage — these
    # are global statistics of the coordinate space, not labels)
    norm_stats = {}
    if modalities["gps"]:
        if "unit2_gps_long" in df.columns and "unit2_gps_lat" in df.columns:
            norm_stats["gps_mean"] = np.array(
                [df["unit2_gps_long"].mean(), df["unit2_gps_lat"].mean()],
                dtype=np.float32,
            )
            norm_stats["gps_std"]  = np.array(
                [df["unit2_gps_long"].std() + 1e-8, df["unit2_gps_lat"].std() + 1e-8],
                dtype=np.float32,
            )
        else:
            gps_path_col = next((c for c in df.columns if c.lower() == "unit2_loc_cal"), None)
            if gps_path_col is None:
                gps_path_col = next((c for c in df.columns if "unit2_loc" in c.lower()), None)

            if gps_path_col is not None:
                root = os.path.dirname(os.path.abspath(csv_path))
                gps_values = []
                for path in df[gps_path_col].dropna().astype(str):
                    full_path = os.path.join(root, path)
                    gps_values.append(_load_gps_file(full_path))
                gps_values = np.stack(gps_values, axis=0)
                norm_stats["gps_mean"] = gps_values.mean(axis=0)
                norm_stats["gps_std"] = gps_values.std(axis=0) + 1e-8
            else:
                norm_stats["gps_mean"] = np.array([0.0, 0.0], dtype=np.float32)
                norm_stats["gps_std"]  = np.array([1.0, 1.0], dtype=np.float32)

    kwargs = dict(csv_path=csv_path, modalities=modalities, norm_stats=norm_stats)
    train_ds = DeepSense6GDataset(split="train", **kwargs)
    val_ds   = DeepSense6GDataset(split="val",   **kwargs)
    test_ds  = DeepSense6GDataset(split="test",  **kwargs)

    return train_ds, val_ds, test_ds, modalities