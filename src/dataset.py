"""
PyTorch Dataset module for ceilometer backscatter profiles.
Reads images from Google Drive and returns DataLoaders ready for training.
"""

import os
import random
from typing import Optional, Tuple, List
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Cutout:
    """
    Randomly masks out square patches of a tensor image.

    Applied after normalization, as in the original Cutout paper
    (DeVries & Taylor, 2017), so the mask value is 0 (the mean
    of the normalized distribution) rather than raw pixel black.

    Args:
        num_holes (int): number of patches to cut out.
        max_h_size (int): maximum height of each patch, in pixels.
        max_w_size (int): maximum width of each patch, in pixels.
    """

    def __init__(self, num_holes: int = 1, max_h_size: int = 32, max_w_size: int = 32) -> None:
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2], img.shape[-1]

        for _ in range(self.num_holes):
            hole_h = random.randint(1, self.max_h_size)
            hole_w = random.randint(1, self.max_w_size)

            cy = random.randint(0, h - 1)
            cx = random.randint(0, w - 1)

            y1 = max(cy - hole_h // 2, 0)
            y2 = min(cy + hole_h // 2, h)
            x1 = max(cx - hole_w // 2, 0)
            x2 = min(cx + hole_w // 2, w)

            img[..., y1:y2, x1:x2] = 0.0

        return img


def compute_dataset_stats(
    dataset_path: str,
    split: str = "train"
) -> Tuple[List[float], List[float]]:
    """
    Computes per-channel (RGB) mean and std over every image in
    dataset_path/split, on the [0, 1] pixel scale (i.e. before Normalize()).

    ImageNet stats (mean~0.485/0.456/0.406) assume natural RGB photographs.
    The ceilometer backscatter images are instead a colormap-encoded signal:
    EDA (notebooks/01_eda.ipynb, section 3) measured a much higher, colormap-skewed
    mean (~0.618) and non-trivial cross-channel differences (R != G != B, i.e. NOT
    grayscale), so normalizing with the dataset's own stats is a better match for
    this domain than reusing ImageNet's.

    Args:
        dataset_path (str): base dataset path (contains train/val/test folders).
        split        (str): which split to compute stats over. Default "train" —
                            val/test must never influence normalization stats,
                            or it would leak information from held-out data.

    Returns:
        Tuple[List[float], List[float]]: (means, stds), one value per RGB channel.
    """

    split_dir = os.path.join(dataset_path, split)
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for cls in sorted(os.listdir(split_dir)):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if fname.startswith('.'):
                continue
            img = np.asarray(
                Image.open(os.path.join(cls_dir, fname)).convert("RGB"), dtype=np.float64
            ) / 255.0
            channel_sum += img.reshape(-1, 3).sum(axis=0)
            channel_sq_sum += (img.reshape(-1, 3) ** 2).sum(axis=0)
            n_pixels += img.shape[0] * img.shape[1]

    means = channel_sum / n_pixels
    stds = np.sqrt(channel_sq_sum / n_pixels - means ** 2)

    return means.tolist(), stds.tolist()


def get_transforms(
    image_size: int = 224,
    dataset_path: Optional[str] = None,
    compute_local_stats: bool = True,
    aug_cfg: Optional[dict] = None
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns image transformations for training and evaluation.

    Training uses random horizontal flip (as described in the paper)
    plus RandAugment and Cutout as additional augmentation.
    Validation and test use only resize and normalization.

    Normalization: by default (compute_local_stats=True) mean/std are computed
    from dataset_path's train split via compute_dataset_stats() instead of the
    hardcoded ImageNet stats — see compute_dataset_stats()'s docstring for why.
    Falls back to ImageNet stats if compute_local_stats=False, or if dataset_path
    isn't provided (local stats can't be computed without it) — so existing
    callers that only pass image_size keep the original ImageNet-normalized
    behavior unchanged.

    Augmentation strength (RandAugment magnitude, Cutout hole size) is read from
    aug_cfg (typically config["augmentation"] from configs/config.yaml); any key
    missing from aug_cfg (or aug_cfg=None entirely) falls back to the original
    hardcoded defaults (num_ops=2, magnitude=9, 1 hole up to 32x32).

    Args:
        image_size          (int)          : target size for resizing images.
                                             Default 224 as in the paper.
        dataset_path         (str, optional): base dataset path, used only to
                                             compute local normalization stats.
        compute_local_stats  (bool)        : if True (default) and dataset_path
                                             is given, normalize with stats
                                             computed from its train split
                                             instead of ImageNet's.
        aug_cfg              (dict, optional): augmentation hyperparameters —
                                             randaugment_num_ops, randaugment_magnitude,
                                             cutout_num_holes, cutout_max_h_size,
                                             cutout_max_w_size. Missing keys fall
                                             back to the original hardcoded values.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]:
            - train_transform: with data augmentation
            - eval_transform:  without data augmentation
    """

    if compute_local_stats and dataset_path is not None:
        means, stds = compute_dataset_stats(dataset_path)
        print(
            f"[get_transforms] Normalizzazione con stats locali (train): "
            f"mean={[round(m, 3) for m in means]}  std={[round(s, 3) for s in stds]}"
        )
    else:
        if compute_local_stats and dataset_path is None:
            print(
                "[get_transforms] compute_local_stats=True ma dataset_path non fornito: "
                "fallback a ImageNet stats."
            )
        # ImageNet normalization values — original default, kept as fallback
        means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    normalize: transforms.Normalize = transforms.Normalize(mean=means, std=stds)

    default_aug = {
        "randaugment_num_ops": 2,
        "randaugment_magnitude": 9,
        "cutout_num_holes": 1,
        "cutout_max_h_size": 32,
        "cutout_max_w_size": 32,
    }
    if aug_cfg:
        default_aug.update(aug_cfg)
    aug = default_aug

    # Original images are 1000x150 pixels.
    # Resizing to 224x224 distorts the aspect ratio but follows
    # the same approach used in the paper.
    train_transform: transforms.Compose = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),        # augmentation as in paper
        transforms.RandAugment(
            num_ops=aug["randaugment_num_ops"],
            magnitude=aug["randaugment_magnitude"],
        ),                                              # operates on PIL image
        transforms.ToTensor(),                          # converts pixels 0-255 to tensor 0-1
        normalize,
        Cutout(
            num_holes=aug["cutout_num_holes"],
            max_h_size=aug["cutout_max_h_size"],
            max_w_size=aug["cutout_max_w_size"],
        )                                                # operates on normalized tensor
    ])

    eval_transform: transforms.Compose = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

    return train_transform, eval_transform


def get_dataloaders(
    dataset_path: str,
    batch_size: int = 12,
    image_size: int = 224,
    num_workers: int = 2,
    compute_local_stats: bool = True,
    aug_cfg: Optional[dict] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Reads train/, val/, test/ from dataset_path and returns DataLoaders.

    ImageFolder automatically reads the folder structure:
        train/true/  → label 1
        train/false/ → label 0
    Labels are assigned alphabetically so false=0 and true=1.

    Args:
        dataset_path (str) : base path of the dataset on Google Drive.
                             e.g. '/content/drive/MyDrive/cloud_detection/dataset'
        batch_size   (int) : images per batch. Default 12 as in the paper.
        image_size   (int) : resize dimension. Default 224 as in the paper.
        num_workers  (int) : parallel processes for data loading.
                             2 is a reasonable value for Colab.
        compute_local_stats (bool)        : passed to get_transforms() — normalize
                             with stats computed from dataset_path's train split
                             instead of ImageNet's. Default True.
        aug_cfg              (dict, optional): passed to get_transforms() —
                             augmentation hyperparameters, typically
                             config["augmentation"] from configs/config.yaml.

    Returns:
        Tuple containing:
            - train_loader (DataLoader) : training data with augmentation
            - val_loader   (DataLoader) : validation data without augmentation
            - test_loader  (DataLoader) : test data without augmentation
            - class_names  (List[str])  : class names ['false', 'true']
    """

    train_transform, eval_transform = get_transforms(
        image_size,
        dataset_path=dataset_path,
        compute_local_stats=compute_local_stats,
        aug_cfg=aug_cfg,
    )

    # ImageFolder expects this structure inside dataset_path:
    # train/
    #   true/   <- images with clouds
    #   false/  <- images without clouds
    train_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=os.path.join(dataset_path, 'train'),
        transform=train_transform
    )
    val_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=os.path.join(dataset_path, 'val'),
        transform=eval_transform
    )
    test_dataset: datasets.ImageFolder = datasets.ImageFolder(
        root=os.path.join(dataset_path, 'test'),
        transform=eval_transform
    )

    # shuffle=True in training randomizes image order each epoch
    # so the model learns features rather than memorizing order
    train_loader: DataLoader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    # shuffle=False for val and test — ensures reproducible results
    val_loader: DataLoader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader: DataLoader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # summary to verify correct loading
    # expected numbers from paper: train 770, val 328, test 470
    print(f"Train:      {len(train_dataset)} images")
    print(f"Validation: {len(val_dataset)} images")
    print(f"Test:       {len(test_dataset)} images")
    print(f"Classes:    {train_dataset.classes}")

    return train_loader, val_loader, test_loader, train_dataset.classes