"""
config.py
─────────
All hyperparameters and path settings for BeMamba.
Edit this file before running train.py or attack.py.
"""

import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
# Root of the extracted DeepSense 6G scenario folder.
# Download any scenario from: https://www.deepsense6g.net
# Recommended: Scenario8 or Scenario9 for multi-modal data
SCENARIO = 31
DATASET_ROOT = "./Scenarios/" + str(SCENARIO)

# CSV file with beam labels, GPS readings, and file-path columns.
CSV_FILE = os.path.join(DATASET_ROOT, str(SCENARIO) + ".csv")

# Directories for file-based modalities.
# The dataset loader checks whether each directory exists AND whether the CSV
# references files in it — modalities absent from the data are skipped silently.
IMAGE_DIR = os.path.join(DATASET_ROOT, "unit1", "camera_data")
LIDAR_DIR = os.path.join(DATASET_ROOT, "unit1", "lidar_data")   # .bin (KITTI) or .npy
RADAR_DIR = os.path.join(DATASET_ROOT, "unit1", "mmWave_data")   # .txt or .npy (beam power / range-Doppler)

# ── Dataset ────────────────────────────────────────────────────────────────────
NUM_BEAMS  = 64   # Beam codebook size (64 for all DeepSense 6G scenarios)
SEQ_LEN    = 5    # Temporal window — how many past frames form one sample
VAL_RATIO  = 0.15
TEST_RATIO = 0.15
SEED       = 42

# ── Model ──────────────────────────────────────────────────────────────────────
D_MODEL    = 128   # Mamba hidden dimension
D_STATE    = 16    # SSM state dimension
D_CONV     = 4     # Depthwise-conv kernel width inside MambaBlock
EXPAND     = 2     # Inner-dimension expansion factor
NUM_LAYERS = 2     # Stacked MambaBlock layers per TSM / MSM module

# LiDAR voxel grid dimensions used by LiDARExtractor
LIDAR_VOXEL_H = 32
LIDAR_VOXEL_W = 32
LIDAR_VOXEL_D = 16   # depth / height bins

# Radar range-Doppler map expected shape (H × W); padded/cropped to this size
RADAR_H = 64
RADAR_W = 64

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE = 24
EPOCHS     = 50
LR         = 1e-3
WEIGHT_DECAY = 1e-4
TOP_K      = [1, 3, 5]
MODALITY_DROPOUT = 0.0
GPS_NOISE_STD = 0.01
RADAR_NOISE_STD = 0.01
SCHEDULER_T0 = 100
SCHEDULER_T_MULT = 2

CHECKPOINT_DIR = "checkpoints"
BEST_CKPT      = os.path.join(CHECKPOINT_DIR, "bemamba_best.pt")

# Modality-dropout attack: fraction of modality tokens to zero out in MSM
MODAL_DROP_FRAC = 0.5

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"