# BeMamba v2 — Modular Implementation

**Paper:** BeMamba: Efficient Multimodal Sensing-Aided Beamforming via State Space Model  
**Dataset:** DeepSense 6G — https://www.deepsense6g.net

---

## File Structure

```
bemamba_v2/
├── config.py        ← All hyperparameters and paths (edit this first)
├── models.py        ← MambaBlock, TSM, MSM, all four extractors, BeMamba
├── dataset.py       ← Auto-detecting dataset loader for all modalities
├── train.py         ← Training + evaluation loop
├── attack.py        ← Five adversarial attacks
├── inference_example.py ← Example of loading and using trained models
└── requirements.txt
```

---

## Modality Auto-Detection

The dataset loader inspects the CSV and filesystem automatically.
No manual flags — just point it at the data.

| Modality | CSV signal | Directory needed |
|----------|-----------|-----------------|
| Image    | `unit1_rgb_*` columns | `unit1/camera_data/` |
| GPS      | `unit2_gps_long` + `unit2_gps_lat` OR `unit2_loc*` columns | `unit2/GPS_data/` (for file-based) |
| LiDAR    | `unit1_lidar_*` columns | `unit1/lidar_data/` |
| Radar    | `unit1_radar_*`, `unit1_pwr_*`, or `mmwave*` columns | `unit1/mmWave_data/` or `unit1/radar/` |

DeepSense 6G scenarios with each modality:
- **Image + GPS:** Scenarios 1–9 (all)
- **Radar:** Scenarios 5, 8 (stored as mmWave power files)
- **LiDAR:** Scenario 8 (point clouds), Scenario 9 (SCR data)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download a scenario from https://www.deepsense6g.net
#    and extract it. Edit config.py:
#      DATASET_ROOT = "./scenario9"   (or whichever you downloaded)
#      CSV_FILE     = "./scenario9/scenario9.csv"

# 3. Train
python train.py

# 4. Run adversarial attacks (requires a trained checkpoint)
python attack.py
```

---

## Model Saving & Loading

Each training run creates a timestamped directory under `checkpoints/` with:

```
checkpoints/
└── Scenario8_20231201_143022/          # scenario + timestamp
    ├── models/
    │   ├── best.pt                     # best validation accuracy
    │   └── final.pt                    # final epoch
    ├── training_metrics.png            # training curves visualization
    ├── training_history.json           # full metrics per epoch
    └── run_summary.json                # hyperparameters + final results
```

### Training Metrics Visualization

The `training_metrics.png` file contains four subplots showing:
- **Training Loss**: Loss curve over epochs
- **Top-1 Accuracy**: Training and validation accuracy
- **Top-3 & Top-5 Accuracy**: Validation accuracy for different k values
- **Learning Rate**: Learning rate schedule (log scale)

### Loading a Trained Model

```python
from train import load_model

# Load the best model from a run
model, metadata = load_model("checkpoints/Scenario8_20231201_143022/models/best.pt")

print(f"Loaded {metadata['scenario']} model from epoch {metadata['epoch']}")
print(f"Active modalities: {metadata['modalities']}")
```

See `inference_example.py` for a complete inference workflow.

---

## Architecture

```
Per active modality (T timesteps):
  raw input → Modality Extractor → (B, T, d_model)
                                        │
                              TimeSequenceMamba (TSM)
                              intra-modal temporal fusion
                                        │
                                   (B, d_model)
                                        │
                    ┌───────────────────┴──────────────────┐
              image_feat          gps_feat        lidar_feat  radar_feat
                                        │
                              ModalSequenceMamba (MSM)
                              cross-modal fusion
                                        │
                                   (B, d_model)
                                        │
                                   Classifier
                                        │
                               (B, NUM_BEAMS) logits
```

---

## Adversarial Attacks (`attack.py`)

| # | Attack | Modality | Threat Model | Description |
|---|--------|----------|-------------|-------------|
| 1 | **FGSM** | Image | White-box | Single-step L∞ gradient sign attack |
| 2 | **PGD** | Image | White-box | Iterative projected gradient descent (Madry et al.) |
| 3 | **ModalDrop** | Any | Black-box physical | Zero out one full sensor modality |
| 4 | **GPSSpoof** | GPS | White-box | Adversarial shift of GPS coordinates (FGSM on GPS) |
| 5 | **FeatureNoise** | LiDAR / Radar | Grey-box | Gaussian noise injected into feature vectors post-extraction |

### Example output

```
══════════════════════════════════════════════════════════════
Attack                   top1      top3      top5
──────────────────────────────────────────────────────────────
Clean                  0.6480    0.8380    0.8990
FGSM                   0.4210    0.6930    0.7840  ▼0.2270
PGD                    0.2830    0.5410    0.6720  ▼0.3650
Drop-image             0.5120    0.7650    0.8430  ▼0.1360
Drop-gps               0.5870    0.8010    0.8680  ▼0.0610
Drop-lidar             0.6290    0.8210    0.8870  ▼0.0190
Drop-radar             0.6350    0.8290    0.8940  ▼0.0130
GPS-Spoof              0.5340    0.7730    0.8510  ▼0.1140
FeatNoise-lidar        0.5920    0.8050    0.8710  ▼0.0560
FeatNoise-radar        0.6140    0.8190    0.8860  ▼0.0340
══════════════════════════════════════════════════════════════
```

---

## Key Hyperparameters (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEQ_LEN` | 5 | Temporal window length |
| `D_MODEL` | 128 | Mamba hidden dimension |
| `D_STATE` | 16 | SSM state size |
| `EXPAND` | 2 | Inner-dim expansion factor |
| `NUM_LAYERS` | 2 | Stacked Mamba blocks |
| `ATK_EPS` | 8/255 | FGSM / PGD L∞ budget |
| `ATK_STEPS` | 10 | PGD iterations |
| `LIDAR_VOXEL_D/H/W` | 16/32/32 | LiDAR voxel grid shape |
| `RADAR_H/W` | 64/64 | Radar map spatial size |