# Cross-Scenario BeMamba

BeMamba is a deep learning framework for multimodal beamforming in wireless communication systems, specifically designed for the DeepSense 6G dataset. It leverages State Space Models (SSMs) implemented via Mamba blocks to efficiently process temporal sequences from multiple sensing modalities, including images, GPS coordinates, LiDAR point clouds, and radar data. The model supports domain adaptation strategies to handle cross-scenario fine-tuning and includes features like Elastic Weight Consolidation (EWC) for preventing catastrophic forgetting.

## Features

- **Multimodal Fusion**: Automatically detects and integrates available modalities (image, GPS, LiDAR, radar) from the dataset.
- **Temporal Modeling**: Uses Time Sequence Mamba (TSM) for intra-modal temporal fusion and Modal Sequence Mamba (MSM) for cross-modal fusion.
- **Domain Adaptation**: Supports strategies like full fine-tuning, adapter-only training, and progressive unfreezing, with optional EWC regularization.
- **Efficient Architecture**: Lightweight Mamba-based SSMs for linear-time sequence processing.
- **Evaluation Tools**: Includes scripts for training, domain adaptation, and cross-inference evaluation.
- **Modality Dropout**: Built-in support for robustness training via modality dropout.

## Requirements

- **Python Version**: 3.10.11 (as specified in .python-version)
- **Dependencies**: 
  - PyTorch (with CUDA support for GPU acceleration)
  - torchvision
  - numpy
  - pandas
  - matplotlib
  - tqdm
  - scikit-learn
  - PIL (Pillow)

Install dependencies using pip:
```bash
pip install -r requirements.txt
```

## Installation

1. Clone or download the project repository.
2. Ensure Python 3.10.11 is installed and set as the active version (use `pyenv` or similar if needed).
3. Install the required packages as listed above.
4. Place the project files in your working directory.

## Dataset

This project uses the DeepSense 6G dataset for training and evaluation. The dataset consists of multimodal sensing data collected in various driving scenarios for beamforming tasks.

### Downloading the Dataset
- Download the desired DeepSense 6G scenarios from the official website: [https://www.deepsense6g.net](https://www.deepsense6g.net).
- Recommended scenarios: Scenario 8 or Scenario 9 for multi-modal data (including image, GPS, LiDAR, and radar).

### Organizing the Dataset
- Extract the downloaded scenarios into a folder named Scenarios in the project root directory.
- Each scenario should be in its own subfolder (e.g., 31, 32, etc.).
- Ensure the following structure for each scenario:
  ```
  Scenarios/
  ├── <scenario_number>/
  │   ├── <scenario_number>.csv  # CSV file with labels and file paths
  │   ├── unit1/
  │   │   ├── camera_data/       # Image files
  │   │   ├── lidar_data/        # LiDAR files (.bin or .npy)
  │   │   └── mmWave_data/       # Radar files (.txt or .npy)
  ```
- The dataset loader automatically detects available modalities based on the CSV file and directory existence.

## Configuration

Edit config.py to customize hyperparameters, paths, and settings:
- Set `DATASET_ROOT` to the path of your scenario folder.
- Adjust model parameters (e.g., `D_MODEL`, `NUM_LAYERS`), training settings (e.g., `BATCH_SIZE`, `EPOCHS`), and device (`DEVICE`).
- For domain adaptation, configure EWC and strategy parameters.

## Usage

### Training a New Model

Run train.py to train BeMamba on a specific scenario:

```bash
python train.py --run-name my_training_run --patience 10
```

- `--run-name`: Optional name for the training run (defaults to timestamp-based).
- `--patience`: Number of epochs to wait for improvement before early stopping.
- Results, models, and plots are saved in `checkpoints/<run_name>/`.

### Domain Adaptation

Use domain_adapt.py to adapt a trained model to a new domain:

```bash
python domain_adapt.py --checkpoint ./checkpoints/<source_run>/models/best.pt \
    --source-csv ./Scenarios/31/31.csv \
    --target-csv ./Scenarios/32/32.csv \
    --strategy adapter-only \
    --epochs 50 \
    --use-ewc \
    --ewc-lambda 1.0 \
    --run-name domain_adapt_31_to_32
```

- `--checkpoint`: Path to the source model checkpoint.
- `--source-csv` and `--target-csv`: Paths to source and target CSV files.
- `--strategy`: Adaptation strategy (`full`, `adapter-only`, `progressive`).
- `--use-ewc`: Enable Elastic Weight Consolidation.
- Adapted models and metrics are saved in `checkpoints/<run_name>/`.

### Cross-Inference Evaluation

Evaluate a trained model on a different dataset split or scenario using cross_infer.py:

```bash
python cross_infer.py --checkpoint ./checkpoints/<run>/models/best.pt \
    --csv ./Scenarios/32/32.csv \
    --dataset-root ./Scenarios/32 \
    --split test \
    --batch-size 24
```

- `--checkpoint`: Path to the model checkpoint.
- `--csv`: Path to the evaluation CSV.
- `--dataset-root`: Root folder of the evaluation dataset.
- `--split`: Dataset split to evaluate (`train`, `val`, `test`).
- Results are saved in `cross_inference/<run>_<dataset>/`.

## File Overview

- **.python-version**: Specifies Python version 3.10.
- **config.py**: Central configuration file for paths, hyperparameters, and settings.
- **dataset.py**: Dataset loading and preprocessing utilities, including modality detection and data augmentation.
- **models.py**: Model definitions, including BeMamba, Mamba blocks, and modality-specific extractors.
- **train.py**: Script for training BeMamba on a dataset scenario.
- **domain_adapt.py**: Script for domain adaptation and cross-scenario fine-tuning.
- **cross_infer.py**: Script for evaluating a model on different datasets or splits.

## Notes

- Ensure GPU availability if using CUDA (set `DEVICE = "cuda"` in config.py).
- The code uses automatic mixed precision (AMP) for faster training on compatible hardware.
- For reproducibility, random seeds are set based on `SEED` in config.py.
- Modality dropout and noise injection can be enabled in config.py for robustness experiments.

## License

This project is provided as-is for research purposes. Please refer to the DeepSense 6G dataset license for data usage terms.

For questions or issues, refer to the code comments or raise an issue in the repository.