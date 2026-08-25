"""
PyTorch Dataset module for ceilometer backscatter profiles.
Reads images from Google Drive and returns DataLoaders ready for training.
"""

import os
import random
from typing import Tuple, List
import torch
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


def get_transforms(
    image_size: int = 224
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns image transformations for training and evaluation.

    Training uses random horizontal flip (as described in the paper)
    plus RandAugment and Cutout as additional augmentation.
    Validation and test use only resize and normalization.

    Args:
        image_size (int): target size for resizing images.
                          Default 224 as in the paper.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]:
            - train_transform: with data augmentation
            - eval_transform:  without data augmentation
    """

    # ImageNet normalization values — compatible with all pretrained
    # models used in this project: ResNet50, ConvNeXt, Swin Transformer
    normalize: transforms.Normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Original images are 1000x150 pixels.
    # Resizing to 224x224 distorts the aspect ratio but follows
    # the same approach used in the paper.
    train_transform: transforms.Compose = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),        # augmentation as in paper
        transforms.RandAugment(num_ops=2, magnitude=9),  # operates on PIL image
        transforms.ToTensor(),                          # converts pixels 0-255 to tensor 0-1
        normalize,
        Cutout(num_holes=1, max_h_size=32, max_w_size=32)  # operates on normalized tensor
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
    num_workers: int = 2
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

    Returns:
        Tuple containing:
            - train_loader (DataLoader) : training data with augmentation
            - val_loader   (DataLoader) : validation data without augmentation
            - test_loader  (DataLoader) : test data without augmentation
            - class_names  (List[str])  : class names ['false', 'true']
    """

    train_transform, eval_transform = get_transforms(image_size)

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