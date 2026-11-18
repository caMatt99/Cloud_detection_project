# Cloud Detection – Ceilometer Backscatter Classification

## Overview
This project benchmarks CNN and transformer architectures for binary cloud detection (cloudy / clear) from ceilometer lidar backscatter profile images. It reproduces the baseline from Chisari et al., *"Cloud Detection Challenge – Methods and Results"* (IEEE Access, 2025) and *"On the Cloud Detection from Backscattered Images..."* (ICIP, 2024), then evaluates newer architectures (ConvNeXt-Base, Swin Transformer) against it. The data source is the public Zenodo dataset [record 10616434](https://zenodo.org/records/10616434): 1000×150 backscatter images split into `train`/`val`/`test`, each containing `true` (cloudy) and `false` (clear) subfolders. Every experiment is a YAML config run over several seeds by `src/experiment.py`, from the notebooks or the command line. Per-seed results and aggregates are written to `paths.results/<experiment.name>/`.

---

## System Requirements

- Python 3.9 or later (required by `torch>=2.0`)
- Training/inference: `torch`, `torchvision`, `timm`
- Metrics, statistics and plotting: `scikit-learn`, `scipy`, `matplotlib`, `numpy`, `pandas`, `pillow`
- Config and notebooks: `pyyaml`, `jupyterlab`

Direct dependencies are pinned in `requirements.txt`.

---

## Installation

Local (code editing, EDA, CPU smoke tests):

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Training runs on Kaggle (GPU T4 x2). Each notebook's first cell clones the repo into `/tmp`, so `src/` and `configs/` always come from the pushed `main`; results and checkpoints go to `/kaggle/working`. Once per notebook:

1. Settings → Accelerator *GPU T4 x2*, Internet *On*.
2. *Add Input* → the dataset, **already split** into `train/val/test` (`/kaggle/input` is read-only, so the split cannot be created there). `dataset.path.kaggle: auto` finds the folder under `/kaggle/input`, ignoring `__MACOSX`; if several candidates exist, set the path explicitly in the YAML.
3. Add-ons → Secrets → `GITHUB_TOKEN` only if the repo is private, enabled for the notebook.
4. For multi-seed runs use *Save Version → Save & Run All*: it runs unattended and keeps the outputs.

Notebook cells are not synced from GitHub: after changing a notebook in the repo, re-import it on Kaggle. The setup cell prints the commit in use.

---

## Usage

### 1. Prepare the dataset

Download the dataset from [Zenodo](https://zenodo.org/records/10616434) and lay it out as `train/{true,false}`, `val/{true,false}`, `test/{true,false}`. If only `train` and `test` exist, create the validation split once on a fresh copy of the data:

```python
from dataset import make_val_split
make_val_split("/path/to/dataset", val_ratio=0.30, seed=42)   # expected: 770 / 328 / 470
```

Every run calls `check_split()`, which fails if the on-disk counts differ from the expected split, and logs a fingerprint of the file lists: runs with different fingerprints used different data and are not comparable.

### 2. Run an experiment

An experiment is a YAML file in `configs/`. It is run once per seed.

```bash
python src/experiment.py --config configs/resnet50_paper.yaml          # seeds from the config
python src/experiment.py --config configs/convnext_legacy.yaml --seeds 0 1 2
```

or from `notebooks/02_baseline.ipynb` / `notebooks/03_new_model.ipynb` (set `CONFIG_PATH`), which call `run_seeds_parallel()`: with two GPUs it starts one worker process per GPU and assigns seeds round-robin (seeds 0 1 2 → GPU0: 0, 2 · GPU1: 1), logging to `<results>/<experiment>/logs/gpu<i>.log`; with one GPU it runs the seeds sequentially. The CLI equivalent is `--parallel`. Per-seed results and a `summary.csv` (mean / std / min / max) are written to `paths.results/<experiment.name>/`, the val-selected weights to `paths.checkpoints/<experiment.name>/seed<k>.pt`.

### 3. Evaluation protocol

- the checkpoint is selected on validation (`selection.monitor`: `val_loss` or `val_acc`), never on test;
- test is reported both with the val-selected weights and with the last-epoch weights (`test_best_*` / `test_last_*` in the per-seed JSON);
- precision / recall / F1 are binary with positive class `true`, as in the paper (weighted averages are logged too);
- results are compared as mean ± std over seeds. The test set has 470 images, so one image is 0.21 pp.

---

## Configs

| Config | Purpose |
|---|---|
| `resnet50_paper.yaml` | Paper replica: torchvision ResNet50, SGD (lr 1e-4, momentum 0.8, wd 1e-5), flip only, ImageNet normalization, paper early stopping (\|Δval_loss\| < 0.05 for 3 epochs). |
| `resnet50_paper_full.yaml` | Same as `resnet50_paper.yaml` without early stopping: 60 epochs, epoch selected on validation. Isolates the effect of the paper's stopping criterion, which halts training at epoch 5–6. |
| `convnext_legacy.yaml` | The ConvNeXt-Base recipe that produced the earlier single-run 92.77%: gradual unfreezing with the optimizer rebuilt every epoch, train-split normalization, RandAugment + Cutout, dropout 0.3 on every `nn.Dropout`, no early stopping. Kept to re-measure that result over seeds. |

---

## Results

| Model | Protocol | Test accuracy |
|---|---|---|
| ResNet50 (paper, Chisari et al.) | single best run | 89.57% |
| ConvNeXt-Base, legacy recipe | single unseeded run, last epoch | 92.77% |

The 92.77% run was neither seeded nor selected on validation, so it is being re-measured with `convnext_legacy.yaml` under the protocol above before being compared with the paper.

---

## Code Reference

| Module | Contents |
|---|---|
| `src/utils.py` | `seed_everything()`, `seed_worker()`, environment detection (`kaggle` / `local`), `print_env()` (GPU / torch-build check), `find_dataset_root()`, `load_config()` (resolves per-environment entries and `dataset.path: auto`), `flatten_dict()`. |
| `src/dataset.py` | Split utilities (`check_split`, `split_fingerprint`, `make_val_split`), transforms (`paper` / `randaugment`, `imagenet` / `local` normalization), seeded `get_dataloaders()`. |
| `src/models.py` | `get_model()` with pinned timm tags and torchvision `resnet50_tv`; `drop_rate` / `drop_path_rate`; legacy helpers `apply_legacy_dropout()` and `get_unfrozen_params()`. |
| `src/engine.py` | `train_one_epoch()` / `evaluate()` with optional AMP and gradient clipping, `build_optimizer()`, `BestCheckpoint`, `EarlyStopping` (`none` / `paper` / `standard`). |
| `src/evaluate.py` | Binary + weighted metrics, confusion matrix, training curves, model comparison chart. |
| `src/experiment.py` | `run_experiment(cfg, seed)`, `run_seeds(cfg)`, `run_seeds_parallel(config_path)` (one process per GPU), `report()`, CLI entry point. |

---

## Repository Structure

```
Cloud_detection_project/
├── configs/
│   ├── convnext_legacy.yaml
│   ├── resnet50_paper.yaml
│   └── resnet50_paper_full.yaml
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_new_model.ipynb
├── results/
├── src/
│   ├── dataset.py
│   ├── engine.py
│   ├── evaluate.py
│   ├── experiment.py
│   ├── models.py
│   └── utils.py
├── requirements.txt
└── README.md
```
