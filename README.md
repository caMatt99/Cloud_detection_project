# Cloud Detection – Ceilometer Backscatter Classification

## Overview
This project benchmarks CNN and transformer architectures for binary cloud detection (cloudy / clear) from ceilometer lidar backscatter profile images. It reproduces the baseline from Chisari et al., *"Cloud Detection Challenge – Methods and Results"* (IEEE Access, 2025) and *"On the Cloud Detection from Backscattered Images..."* (ICIP, 2024), then evaluates newer architectures (ConvNeXt-Base, Swin Transformer) against it. The data source is the public Zenodo dataset [record 10616434](https://zenodo.org/records/10616434): 1000×150 backscatter images split into `train`/`val`/`test`, each containing `true` (cloudy) and `false` (clear) subfolders. Training runs and evaluation are driven from Jupyter notebooks and a standalone CLI script; results (confusion matrices, training curves, cross-model comparison charts) are written as PNGs to `results/` for reporting.

---

## System Requirements

- Python 3.9 or later (required by `torch>=2.0`)
- Training/inference: `torch`, `torchvision`, `timm`
- Metrics and plotting: `scikit-learn`, `matplotlib`, `numpy`, `pandas`
- Config and utilities: `pyyaml`, `tqdm`

All are pinned in `requirements.txt`.

---

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

The workflow has two independent entry points: a standalone CLI script (`src/train.py`) and a notebook-driven pipeline (`src/dataset.py` + `src/models.py` + `src/evaluate.py`, configured via `configs/config.yaml`). Both expect the same on-disk dataset layout but build different models internally — see [Design Principles](#design-principles).

### 1. Prepare the dataset

Download the dataset from [Zenodo](https://zenodo.org/records/10616434) and lay it out as `train/{true,false}`, `val/{true,false}`, `test/{true,false}`. If only `train` and `test` exist, run the split cell in `notebooks/00-setup.ipynb`: it moves a seeded random 30% of each class from `train` into `val` (`random.seed(42)`), reproducing the paper's ~49/21/30 train/val/test proportions (expected counts: 770/328/470 images).

```python
# inside notebooks/00-setup.ipynb
VAL_RATIO = 0.30
```

### 2. Train via the CLI script

`src/train.py` is self-contained: it builds its own `torchvision` backbone with a single-logit binary head (`BCEWithLogitsLoss`, sigmoid threshold 0.5), trains with early stopping on validation loss and `ReduceLROnPlateau`, and writes `config.json`, `history.json`, `best_model.pt`, and `test_results.json` to `runs/<model>_<optimizer>_lr<lr>/`. Use this path for a reproducible, resumable-by-checkpoint run outside a notebook.

```bash
python src/train.py --data_dir dataset --model resnet50 \
  --epochs 60 --batch_size 12 --optimizer adam --lr 1e-4
```

### 3. Train via the notebook pipeline

`notebooks/02_baseline.ipynb` and `notebooks/03_new_model.ipynb` are the intended entry points for the baseline (ResNet50) and new architectures (ConvNeXt-Base, Swin Transformer) respectively, using `src/dataset.get_dataloaders()` and `src/models.get_model()` with hyperparameters read from `configs/config.yaml`. This pipeline assumes execution in Google Colab with the dataset under Google Drive, matching the paths hardcoded in `config.yaml` and `notebooks/00-setup.ipynb`. Both notebooks are currently empty scaffolding, as is `notebooks/01_eda.ipynb` (intended for exploratory analysis of the backscatter images).

```yaml
# configs/config.yaml
model:
  name: "convnext_base"
  pretrained: true
```

### 4. Evaluate and compare models

`src/evaluate.py` runs inference on a trained model, computes weighted accuracy/F1/precision/recall, and saves a confusion matrix, training curves, and a grouped bar chart comparing all benchmarked models to `results/`.

```python
from src.evaluate import get_predictions, compute_metrics, compare_models

preds, labels = get_predictions(model, test_loader, device)
metrics = compute_metrics(preds, labels, model_name="convnext_base")
```

---

## Code Reference

### Data Pipeline (`src/dataset.py`)

Builds `DataLoader`s from an `ImageFolder`-structured dataset directory.

| Script | Description |
|---|---|
| `dataset.py` | `get_transforms()` returns train/eval `torchvision` transform pipelines: resize to 224×224, ImageNet normalization, and random horizontal flip (p=0.5) applied only to training. `get_dataloaders()` reads `train/`, `val/`, `test/` from `dataset_path` via `ImageFolder` (labels assigned alphabetically: `false`=0, `true`=1), builds shuffled train / non-shuffled val and test loaders, and prints per-split image counts for validation against the paper's expected splits. |

### Model Definitions (`src/models.py`)

Instantiates pretrained `timm` backbones for the notebook pipeline.

| Script | Description |
|---|---|
| `models.py` | `get_model()` loads a `timm` backbone (`resnet50`, `convnext_base`, `swin_base_patch4_window7_224`, `vgg16`, `efficientnet_b0`, `inception_v3`, `vit_base_patch16_224`), pretrained on ImageNet, with the head replaced for `num_classes` outputs (default 2); raises `ValueError` on an unrecognized name. `get_device()` selects CUDA over CPU and logs GPU name/memory. `count_parameters()` returns total vs. trainable parameter counts. |

### Training CLI (`src/train.py`)

Standalone argparse script, independent of `dataset.py` and `models.py`.

| Script | Description |
|---|---|
| `train.py` | Builds `torchvision` models (`resnet50`, `resnet101`, `vgg16`, `inceptionv3`, `efficientnet_b0`, `vit_b16`) with a single-logit binary head trained via `BCEWithLogitsLoss`, unlike the 2-class softmax heads used elsewhere in `src/`. Supports backbone freezing, gradient clipping, `ReduceLROnPlateau`, early stopping on validation loss, class-weighted loss or `WeightedRandomSampler` for class imbalance, and InceptionV3's auxiliary-logit loss term (weighted 0.4). Persists run config, training history, best checkpoint, and test-set results (including the confusion matrix) under `runs/<model>_<optimizer>_lr<lr>/`. |

### Evaluation & Visualization (`src/evaluate.py`)

Notebook-facing helpers for models produced by `models.py`.

| Script | Description |
|---|---|
| `evaluate.py` | `get_predictions()` runs inference and returns `argmax` class predictions — assumes a 2-logit softmax output, not the single-logit sigmoid head from `train.py`. `compute_metrics()` reports weighted accuracy/F1/precision/recall. `plot_confusion_matrix()` and `plot_training_history()` save per-model PNGs to `results/`. `compare_models()` saves a grouped bar chart (accuracy and F1 per model) across all entries in a `{model_name: metrics_dict}` mapping. |

---

## Repository Structure

```
Cloud_detection_project/
├── configs/
│   └── config.yaml
├── notebooks/
│   ├── 00-setup.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_new_model.ipynb
├── results/
│   └── .gitkeep
├── src/
│   ├── dataset.py
│   ├── evaluate.py
│   ├── models.py
│   └── train.py
├── requirements.txt
└── README.md
```

---

## Design Principles

- **Two independent training pipelines.** `train.py` is a self-contained CLI script with its own `torchvision` model builder and single-logit sigmoid head. The notebook pipeline (`dataset.py` + `models.py` + `evaluate.py` + `config.yaml`) uses `timm` backbones with 2-class softmax heads. Checkpoints and evaluation code are not interchangeable between the two: `evaluate.get_predictions()`'s `argmax` logic will not produce correct labels on a `train.py` checkpoint.
- **Google Drive as the dataset backend for the notebook pipeline.** `config.yaml`, the `dataset.py` docstring, and `00-setup.ipynb` hardcode paths under `/content/drive/MyDrive/cloud_detection/`, so that pipeline assumes execution inside Google Colab with Drive mounted. `train.py` instead takes `--data_dir` as a CLI argument and has no Colab dependency.
- **Paper-derived defaults.** Image size (224), batch size (12), ImageNet normalization, and augmentation limited to horizontal flip match the Chisari et al. setup, so newer architectures are benchmarked under the same conditions as the reported ResNet50 baseline (89.57% accuracy).
- **Fixed seeds for reproducibility.** The train/val split (`00-setup.ipynb`) and training (`train.py`, `torch.manual_seed`) both seed with `42`.
