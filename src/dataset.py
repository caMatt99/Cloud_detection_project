"""
PyTorch data pipeline for ceilometer backscatter profiles.

Images are 1000x150 colormap-encoded backscatter plots, stored as
dataset/{train,val,test}/{true,false}/. ImageFolder assigns labels
alphabetically: false=0 (clear), true=1 (cloudy).
"""

import hashlib
import os
import random
import shutil
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import seed_worker

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLITS = ("train", "val", "test")
CLASSES = ("false", "true")

# Paper split: 770 / 328 / 470 (train / val / test).
EXPECTED_SPLIT: Dict[str, Dict[str, int]] = {
    "train": {"true": 515, "false": 255},
    "val":   {"true": 220, "false": 108},
    "test":  {"true": 315, "false": 155},
}


# ─────────────────────────────────────────────
# Split integrity
# ─────────────────────────────────────────────

def _list_images(directory: str) -> List[str]:
    """Sorted, hidden-file-free listing — os.listdir order is not stable on Drive."""
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if not f.startswith("."))


def count_split(dataset_path: str) -> Dict[str, Dict[str, int]]:
    return {
        split: {c: len(_list_images(os.path.join(dataset_path, split, c))) for c in CLASSES}
        for split in SPLITS
    }


def split_fingerprint(dataset_path: str) -> str:
    """
    SHA-1 over the sorted relative paths of every image in every split.
    Logged with each run: two runs with different fingerprints were trained
    or evaluated on different data and are not comparable.
    """
    h = hashlib.sha1()
    for split in SPLITS:
        for c in CLASSES:
            for f in _list_images(os.path.join(dataset_path, split, c)):
                h.update(f"{split}/{c}/{f}\n".encode())
    return h.hexdigest()[:12]


def check_split(
    dataset_path: str,
    expected: Dict[str, Dict[str, int]] = EXPECTED_SPLIT,
) -> str:
    """
    Verifies on-disk counts against `expected` and returns the split
    fingerprint. Raises instead of silently re-splitting: changing the data
    between runs is exactly what makes results incomparable.
    """
    actual = count_split(dataset_path)
    if actual != expected:
        lines = [
            f"  {s}/{c}: {actual[s][c]} (expected {expected[s][c]})"
            for s in SPLITS for c in CLASSES if actual[s][c] != expected[s][c]
        ]
        raise RuntimeError(
            "Dataset split does not match the expected one:\n" + "\n".join(lines)
            + "\nRun make_val_split() once on a fresh copy of the original data."
        )
    return split_fingerprint(dataset_path)


def make_val_split(dataset_path: str, val_ratio: float = 0.30, seed: int = 42) -> None:
    """
    One-off: moves a seeded `val_ratio` of each class from train/ to val/.
    Deterministic given the original train/ content (sorted listing + local RNG).
    Only meant for the original Zenodo layout (train/ + test/, no val/).
    """
    rng = random.Random(seed)
    for c in CLASSES:
        src = os.path.join(dataset_path, "train", c)
        dst = os.path.join(dataset_path, "val", c)
        if _list_images(dst):
            raise RuntimeError(f"{dst} is not empty — refusing to re-split.")
        os.makedirs(dst, exist_ok=True)
        files = _list_images(src)
        rng.shuffle(files)
        n_val = int(len(files) * val_ratio)
        for f in files[:n_val]:
            shutil.move(os.path.join(src, f), os.path.join(dst, f))
        print(f"[make_val_split] {c}: {len(files)} -> {n_val} val, {len(files) - n_val} train")


# ─────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────

class Cutout:
    """
    Masks `num_holes` random rectangles of a normalized tensor with 0
    (= the dataset mean after Normalize), as in DeVries & Taylor (2017).
    """

    def __init__(self, num_holes: int = 1, max_h_size: int = 32, max_w_size: int = 32) -> None:
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2:]
        for _ in range(self.num_holes):
            hole_h = random.randint(1, self.max_h_size)
            hole_w = random.randint(1, self.max_w_size)
            cy, cx = random.randint(0, h - 1), random.randint(0, w - 1)
            y1, y2 = max(cy - hole_h // 2, 0), min(cy + hole_h // 2, h)
            x1, x2 = max(cx - hole_w // 2, 0), min(cx + hole_w // 2, w)
            img[..., y1:y2, x1:x2] = 0.0
        return img


@lru_cache(maxsize=4)
def compute_dataset_stats(dataset_path: str, split: str = "train") -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    Per-channel mean/std on the [0, 1] scale over `split` (train only, to
    avoid leaking val/test). Cached: multi-seed runs read Drive once.
    """
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    for c in CLASSES:
        cls_dir = os.path.join(dataset_path, split, c)
        for fname in _list_images(cls_dir):
            img = np.asarray(Image.open(os.path.join(cls_dir, fname)).convert("RGB"), dtype=np.float64) / 255.0
            flat = img.reshape(-1, 3)
            channel_sum += flat.sum(axis=0)
            channel_sq_sum += (flat ** 2).sum(axis=0)
            n_pixels += flat.shape[0]
    means = channel_sum / n_pixels
    stds = np.sqrt(channel_sq_sum / n_pixels - means ** 2)
    return tuple(means.tolist()), tuple(stds.tolist())


def get_normalization(normalization: str, dataset_path: Optional[str] = None) -> transforms.Normalize:
    """
    'imagenet' (default): matches the statistics the pretrained backbone saw.
    'local'           : train-split statistics (legacy recipe, see 01_eda).
    """
    if normalization == "imagenet":
        return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if normalization == "local":
        if dataset_path is None:
            raise ValueError("normalization='local' requires dataset_path")
        means, stds = compute_dataset_stats(dataset_path)
        return transforms.Normalize(means, stds)
    raise ValueError(f"Unknown normalization '{normalization}' (use 'imagenet' or 'local')")


def get_transforms(
    image_size: int = 224,
    normalization: str = "imagenet",
    augmentation: str = "paper",
    aug_cfg: Optional[dict] = None,
    dataset_path: Optional[str] = None,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns (train_transform, eval_transform).

    augmentation:
        'paper'       : resize + horizontal flip (time reversal) — paper baseline.
        'randaugment' : paper + RandAugment + Cutout (legacy ConvNeXt recipe).

    The 1000x150 images are resized to a square image_size, as in the paper.
    """
    normalize = get_normalization(normalization, dataset_path)
    aug = {
        "randaugment_num_ops": 2,
        "randaugment_magnitude": 9,
        "cutout_num_holes": 1,
        "cutout_max_h_size": 32,
        "cutout_max_w_size": 32,
        **(aug_cfg or {}),
    }

    train_ops = [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if augmentation == "paper":
        train_ops += [transforms.ToTensor(), normalize]
    elif augmentation == "randaugment":
        train_ops += [
            transforms.RandAugment(num_ops=aug["randaugment_num_ops"], magnitude=aug["randaugment_magnitude"]),
            transforms.ToTensor(),
            normalize,
            Cutout(aug["cutout_num_holes"], aug["cutout_max_h_size"], aug["cutout_max_w_size"]),
        ]
    else:
        raise ValueError(f"Unknown augmentation '{augmentation}' (use 'paper' or 'randaugment')")

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return transforms.Compose(train_ops), eval_transform


# ─────────────────────────────────────────────
# DataLoaders
# ─────────────────────────────────────────────

def get_dataloaders(
    dataset_path: str,
    batch_size: int = 12,
    image_size: int = 224,
    num_workers: int = 2,
    normalization: str = "imagenet",
    augmentation: str = "paper",
    aug_cfg: Optional[dict] = None,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Returns (train_loader, val_loader, test_loader, class_names).
    The train loader's shuffling and per-worker augmentation RNG are derived
    from `seed`, so the same seed gives the same batches and augmentations.
    """
    train_tf, eval_tf = get_transforms(image_size, normalization, augmentation, aug_cfg, dataset_path)

    train_ds = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(dataset_path, "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(dataset_path, "test"), transform=eval_tf)

    generator = torch.Generator()
    generator.manual_seed(seed)
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  "
          f"classes={train_ds.classes}  norm={normalization}  aug={augmentation}")
    return train_loader, val_loader, test_loader, train_ds.classes
