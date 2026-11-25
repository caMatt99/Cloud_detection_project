# Cloud Detection – Ceilometer Backscatter Classification

## Overview
This project replicates the ResNet50 baseline of Chisari et al. for binary cloud detection (cloudy / clear) from ceilometer backscatter images, *"Cloud Detection Challenge – Methods and Results"* (IEEE Access, 2025) and *"On the Cloud Detection from Backscattered Images..."* (ICIP, 2024), and compares it with ConvNeXt-Base under a reproducible protocol: every model is trained over three seeds, the checkpoint is selected on validation, and results are reported as mean ± std on the official test set. The data is the public Zenodo dataset [record 10616434](https://zenodo.org/records/10616434): 1000×150 backscatter images, one per hour of measurement, split into `train`/`val`/`test`, each with `true` (cloudy) and `false` (clear) subfolders.

**Main result:** ConvNeXt-Base reaches **89.15 ± 0.56%** test accuracy, +2.6 pp over the ResNet50 replica (86.52 ± 0.33%) and in line with the paper's best single run (89.57%). The error analysis shows that the remaining errors are almost all clear-sky images predicted as cloudy and concentrate on a few days of the test period (see [Analysis](#analysis)).

---

## System Requirements

- Python 3.11 or later (required by the pinned `numpy` / `pandas`)
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

or from `notebooks/02_baseline.ipynb` / `notebooks/03_new_model.ipynb` (set `CONFIG_PATH`), which call `run_seeds_parallel()`: with two GPUs it starts one worker process per GPU and assigns seeds round-robin (seeds 0 1 2 → GPU0: 0, 2 · GPU1: 1), logging to `<results>/<experiment>/logs/gpu<i>.log`; with one GPU it runs the seeds sequentially. The CLI equivalent is `--parallel`. Per-seed results, per-image predictions (`predictions_seed<k>.csv`: split, file, label, P(true) with the last-epoch and val-selected weights, for val and test) and a `summary.csv` (mean / std / min / max) are written to `paths.results/<experiment.name>/` and zipped to `paths.results/<experiment.name>.zip` (the notebooks' last cell shows a download link on Kaggle), the val-selected weights to `paths.checkpoints/<experiment.name>/seed<k>.pt`.

### 3. Evaluation protocol

- the checkpoint is selected on validation (`selection.monitor`: `val_loss` or `val_acc`), never on test;
- test is reported both with the val-selected weights and with the last-epoch weights (`test_best_*` / `test_last_*` in the per-seed JSON);
- precision / recall / F1 are binary with positive class `true`, as in the paper (weighted averages are logged too); ROC-AUC is reported as a threshold-free metric;
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

Test set: 470 images (315 cloudy, 155 clear); one image is 0.21 pp. Mean ± std over seeds 0, 1, 2, weights selected on validation; precision / recall / F1 with positive class `true`.

| Model | Config | Accuracy (%) | Precision | Recall | F1 | ROC-AUC | Best epoch |
|---|---|---|---|---|---|---|---|
| ResNet50, paper (Chisari et al.) | single best run | 89.57 | 0.90 | 0.96 | 0.93 | — | — |
| ResNet50, paper replica | `resnet50_paper.yaml` | 86.52 ± 0.33 | 0.846 ± 0.002 | 0.977 ± 0.004 | 0.907 ± 0.002 | — | 5–6 |
| ResNet50, no early stopping | `resnet50_paper_full.yaml` | 85.74 ± 0.21 | 0.835 ± 0.001 | 0.982 ± 0.002 | 0.902 ± 0.002 | — | 17–27 |
| **ConvNeXt-Base, legacy recipe** | `convnext_legacy.yaml` | **89.15 ± 0.56** | 0.882 ± 0.023 | 0.968 ± 0.028 | 0.923 ± 0.003 | 0.927 ± 0.004 | 57–60 |

ROC-AUC was added to the pipeline after the ResNet50 runs.

- **ConvNeXt vs ResNet50.** +2.6 pp on the same protocol; the worst ConvNeXt seed (88.51%) is above the best ResNet50 seed (86.81%).
- **vs the paper.** ConvNeXt matches the published ResNet50 figure but does not beat it; the paper reports a single run without variance. The ResNet50 replica is 3 pp below it.
- **The earlier 92.77%** (single unseeded ConvNeXt run, last epoch) does not reproduce: the same recipe over three seeds gives 89.15 ± 0.56%.
- **Training longer does not help ResNet50.** Without the paper's early stopping, validation accuracy rises by ~1 pp while test accuracy drops by ~0.8 pp: after epoch ~20 the model overfits (train accuracy 99.6%, validation loss rising).
- **ConvNeXt was still improving at epoch 60** (best epoch 57–60, train accuracy ~77% under gradual unfreezing and heavy augmentation), so the legacy recipe is not converged.

---

## Analysis

**Error profile.** All models trade precision for recall: almost every error is a clear-sky image predicted as cloudy. With ConvNeXt, specificity on the 155 clear test images is only 0.74 on average, against a recall of 0.97. Of the 33 test images misclassified by all three seeds, 30 are false positives, many predicted with P(cloudy) > 0.95.

**Errors cluster by day.** File names encode date and hour (`YYYYMMDD-HH.png`). The persistent false positives concentrate on a few days, e.g. 23 Feb 2023: 15 images, all labelled clear, 6 of them misclassified by every seed. Labels are not derived from the ceilometer itself but from the WRF weather model (1×1 km grid, hourly cloud cover), so confident, day-level clusters of errors are consistent with label noise or with atmospheric conditions (e.g. aerosol layers) that look like cloud in the backscatter signal. They have not been inspected individually.

**Class imbalance is not the cause of the val/test gap.** The cloudy/clear ratio is ~2:1 and identical across train, val and test (2.02 / 2.04 / 2.03, χ² p = 0.998).

**The split is temporal, the validation set is not.** The validation set is a random 30% of the training images (its images span 1 Jan – 25 Feb 2023), while the test set covers a later period (15 Feb – 14 Mar 2023); validation and test share only 7 days. Consecutive hours of the same day are nearly identical images, so a random validation split contains near-duplicates of training images. Consequences observed in the runs:
- validation accuracy overestimates test accuracy (ResNet50 without early stopping: 90.3% val vs 85.7% test);
- selecting epochs on validation improves validation but not test;
- a decision threshold tuned on validation lowers test accuracy for every ConvNeXt seed (86.2–87.2% instead of 88.5–89.6%).

The random validation split is kept because it is the paper's protocol, so the replica stays comparable.

---

## Limitations and future work

- **Temporal validation split.** Hold out the last days of the training period as validation, whole days only, so that model selection reflects the test distribution. This changes the protocol, so baselines must be re-run on it.
- **Clean ConvNeXt recipe.** Full fine-tuning, ImageNet normalization and early stopping, developed on the temporal split; the legacy recipe was not converged at 60 epochs.
- **Label audit.** Inspect the persistent false positives (`predictions_seed<k>.csv`) against the WRF labels to quantify label noise in the test set.
- **Input resolution.** Images are resized from 1000×150 to 224×224, which compresses the height axis 4.5× and can thin out weak cloud layers.
- Three seeds give a usable mean but a coarse estimate of the std.

---

## Code Reference

| Module | Contents |
|---|---|
| `src/utils.py` | `seed_everything()`, `seed_worker()`, environment detection (`kaggle` / `local`), `print_env()` (GPU / torch-build check), `find_dataset_root()`, `load_config()` (resolves per-environment entries and `dataset.path: auto`). |
| `src/dataset.py` | Split utilities (`check_split`, `split_fingerprint`, `make_val_split`), transforms (`paper` / `randaugment`, `imagenet` / `local` normalization), seeded `get_dataloaders()`. |
| `src/models.py` | `get_model()` with pinned timm tags and torchvision `resnet50_tv`; `drop_rate` / `drop_path_rate`; legacy helpers `apply_legacy_dropout()` and `get_unfrozen_params()`. |
| `src/engine.py` | `train_one_epoch()` / `evaluate()` with optional AMP and gradient clipping, `build_optimizer()`, `BestCheckpoint`, `EarlyStopping` (`none` / `paper` / `standard`). |
| `src/evaluate.py` | Binary + weighted metrics, confusion matrix, training curves, model comparison chart. |
| `src/experiment.py` | `run_experiment(cfg, seed)`, `run_seeds(cfg)`, `run_seeds_parallel(config_path)` (one process per GPU), `report()`, `archive_results()` / `download_link()`, CLI entry point. |

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
