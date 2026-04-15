"""
train.py — Cloud Detection Binary Classifier
=============================================
Binary classification (cloudy / clear) from ceilometer backscatter images.
Supported architectures: ResNet50 (baseline), ResNet101, VGG16,
                         InceptionV3, EfficientNet-B0, ViT-B/16

Expected dataset structure:
    dataset/
        train/
            true/    <- cloudy
            false/   <- clear
        val/
            true/
            false/
        test/        <- optional
            true/
            false/

References:
    - Chisari et al., "Cloud Detection Challenge – Methods and Results", IEEE Access 2025
    - Chisari et al., "On the Cloud Detection from Backscattered Images...", ICIP 2024
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)


# ─────────────────────────────────────────────
# 1. ARGUMENT PARSING
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Cloud Detection – Binary Classifier")

    # Paths
    parser.add_argument("--data_dir",         type=str,   default="dataset",
                        help="Root folder containing train/val/test splits")
    parser.add_argument("--output_dir",        type=str,   default="runs",
                        help="Folder for checkpoints, logs, and results")

    # Architecture
    parser.add_argument("--model",             type=str,   default="resnet50",
                        choices=["resnet50", "resnet101", "vgg16",
                                 "inceptionv3", "efficientnet_b0", "vit_b16"],
                        help="Backbone to use")
    parser.add_argument("--pretrained",        action="store_true", default=True,
                        help="Use ImageNet pretrained weights (default: True)")
    parser.add_argument("--freeze_backbone",   action="store_true", default=False,
                        help="Freeze backbone weights, train the head only")

    # Hyperparameters
    parser.add_argument("--epochs",            type=int,   default=60)
    parser.add_argument("--batch_size",        type=int,   default=12)
    parser.add_argument("--lr",                type=float, default=1e-4)
    parser.add_argument("--weight_decay",      type=float, default=1e-5)
    parser.add_argument("--optimizer",         type=str,   default="adam",
                        choices=["adam", "sgd"])
    parser.add_argument("--momentum",          type=float, default=0.7,
                        help="Momentum (SGD only)")
    parser.add_argument("--img_size",          type=int,   default=224,
                        help="Resize target (224 for CNNs, 384 for ViT)")

    # Regularisation
    parser.add_argument("--dropout",           type=float, default=0.5)
    parser.add_argument("--grad_clip",         type=float, default=1.0,
                        help="Max-norm for gradient clipping (0 = disabled)")

    # Early stopping
    parser.add_argument("--patience",          type=int,   default=10,
                        help="Epochs without improvement before stopping")
    parser.add_argument("--min_delta",         type=float, default=1e-4,
                        help="Minimum improvement considered significant")

    # Class imbalance
    parser.add_argument("--class_weight",      action="store_true", default=False,
                        help="Apply class weighting to the loss function")
    parser.add_argument("--weighted_sampler",  action="store_true", default=False,
                        help="Use WeightedRandomSampler to balance batches")

    # Misc
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--num_workers",       type=int,   default=4)
    parser.add_argument("--device",            type=str,   default="auto",
                        choices=["auto", "cuda", "cpu", "mps"])

    return parser.parse_args()


# ─────────────────────────────────────────────
# 2. TRANSFORMS / AUGMENTATION
# ─────────────────────────────────────────────

# ImageNet statistics used by the paper baseline
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(img_size: int, phase: str) -> transforms.Compose:
    """
    Build transforms for each dataset phase.
    Training: resize, random horizontal flip (p=0.5), color jitter, normalise.
    Val/Test: resize and normalise only.
    """
    if phase == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ─────────────────────────────────────────────
# 3. DATASET AND DATALOADERS
# ─────────────────────────────────────────────

def get_sample_weights(dataset) -> torch.Tensor:
    """Return per-sample weights inversely proportional to class frequency."""
    targets = [label for _, label in dataset.samples]
    counts  = np.bincount(targets)
    class_w = 1.0 / counts
    return torch.tensor([class_w[t] for t in targets], dtype=torch.float32)


def build_dataloaders(args):
    data_dir  = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir   = data_dir / "val"
    test_dir  = data_dir / "test"

    train_dataset = datasets.ImageFolder(
        train_dir, transform=get_transforms(args.img_size, "train"))
    val_dataset   = datasets.ImageFolder(
        val_dir,   transform=get_transforms(args.img_size, "val"))

    test_dataset = None
    if test_dir.exists():
        test_dataset = datasets.ImageFolder(
            test_dir, transform=get_transforms(args.img_size, "test"))

    # ImageFolder sorts classes alphabetically: "false"=0 (clear), "true"=1 (cloudy)
    print(f"Detected classes : {train_dataset.class_to_idx}")
    print(f"Train: {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples")
    if test_dataset:
        print(f"Test : {len(test_dataset)} samples")

    sampler = None
    if args.weighted_sampler:
        sample_w = get_sample_weights(train_dataset)
        sampler  = WeightedRandomSampler(sample_w, num_samples=len(sample_w),
                                         replacement=True)
        print("WeightedRandomSampler enabled.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader, train_dataset.class_to_idx


# ─────────────────────────────────────────────
# 4. MODEL BUILDING
# ─────────────────────────────────────────────

def build_model(args) -> nn.Module:
    """
    Instantiate the selected backbone with a binary classification head.
    Outputs a single logit; sigmoid + threshold=0.5 gives the final label.
    """
    weights_flag = "DEFAULT" if args.pretrained else None

    # ── ResNet50 (paper baseline, 89.57% accuracy) ──────────────────
    if args.model == "resnet50":
        model = models.resnet50(weights=weights_flag)
        if args.freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(model.fc.in_features, 1),
        )

    # ── ResNet101 ───────────────────────────────────────────────────
    elif args.model == "resnet101":
        model = models.resnet101(weights=weights_flag)
        if args.freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        model.fc = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(model.fc.in_features, 1),
        )

    # ── VGG16 ───────────────────────────────────────────────────────
    elif args.model == "vgg16":
        model = models.vgg16(weights=weights_flag)
        if args.freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(args.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(args.dropout),
            nn.Linear(4096, 1),
        )

    # ── InceptionV3 ─────────────────────────────────────────────────
    elif args.model == "inceptionv3":
        model = models.inception_v3(weights=weights_flag, aux_logits=True)
        if args.freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        model.AuxLogits.fc = nn.Linear(768, 1)
        model.fc = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(model.fc.in_features, 1),
        )

    # ── EfficientNet-B0 ─────────────────────────────────────────────
    elif args.model == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights_flag)
        if args.freeze_backbone:
            for p in model.features.parameters():
                p.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(in_features, 1),
        )

    # ── ViT-B/16 (Alpha Research Group approach) ────────────────────
    elif args.model == "vit_b16":
        model = models.vit_b_16(weights=weights_flag)
        if args.freeze_backbone:
            for p in model.parameters():
                p.requires_grad = False
        in_features = model.heads.head.in_features
        model.heads = nn.Sequential(
            nn.Dropout(args.dropout),
            nn.Linear(in_features, 1),
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Total params: {total:,}  |  Trainable: {trainable:,}")
    return model


# ─────────────────────────────────────────────
# 5. EARLY STOPPING
# ─────────────────────────────────────────────

class EarlyStopping:
    """
    Monitors val_loss. Saves the best model state and stops training
    when no improvement is seen for `patience` consecutive epochs.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 checkpoint_path: str = "best_model.pt"):
        self.patience         = patience
        self.min_delta        = min_delta
        self.checkpoint_path  = checkpoint_path
        self.best_loss        = float("inf")
        self.counter          = 0
        self.best_model_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss        = val_loss
            self.counter          = 0
            self.best_model_state = copy.deepcopy(model.state_dict())
            torch.save(self.best_model_state, self.checkpoint_path)
        else:
            self.counter += 1
        return self.counter >= self.patience

    def load_best(self, model: nn.Module):
        """Restore the best checkpoint into `model`."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


# ─────────────────────────────────────────────
# 6. EPOCH RUNNER
# ─────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, phase, grad_clip):
    """
    Run one full training or evaluation epoch.
    Returns: (average_loss, predictions_array, targets_array).
    """
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_preds    = []
    all_targets  = []

    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            inputs  = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True).float().unsqueeze(1)

            logits = model(inputs)

            # InceptionV3 returns a namedtuple (logits, aux_logits) during training
            if isinstance(logits, tuple):
                logits, aux_logits = logits
                loss = criterion(logits, targets)
                if aux_logits is not None:
                    loss = loss + 0.4 * criterion(aux_logits, targets)
            else:
                loss = criterion(logits, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long().squeeze(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.long().squeeze(1).cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, np.array(all_preds), np.array(all_targets)


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    return {
        "accuracy":  accuracy_score(targets, preds),
        "f1":        f1_score(targets, preds, zero_division=0),
        "precision": precision_score(targets, preds, zero_division=0),
        "recall":    recall_score(targets, preds, zero_division=0),
    }


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Output directory
    run_name   = f"{args.model}_{args.optimizer}_lr{args.lr}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(output_dir / "best_model.pt")

    # Persist run configuration
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(args)

    # ── Model ─────────────────────────────────────────────────────
    model = build_model(args).to(device)

    # ── Loss ──────────────────────────────────────────────────────
    if args.class_weight:
        train_dataset = train_loader.dataset
        counts     = np.bincount([lbl for _, lbl in train_dataset.samples])
        pos_weight = torch.tensor([counts[0] / counts[1]], dtype=torch.float32).to(device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Class weighting enabled  (pos_weight = {pos_weight.item():.3f})")
    else:
        criterion = nn.BCEWithLogitsLoss()

    # ── Optimiser ─────────────────────────────────────────────────
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == "adam":
        optimizer = optim.Adam(
            trainable_params, lr=args.lr,
            weight_decay=args.weight_decay, betas=(0.9, 0.999),
        )
    else:
        optimizer = optim.SGD(
            trainable_params, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay,
        )

    # ── LR Scheduler ──────────────────────────────────────────────
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7, verbose=True,
    )

    # ── Early stopping ────────────────────────────────────────────
    early_stopper = EarlyStopping(
        patience=args.patience,
        min_delta=args.min_delta,
        checkpoint_path=checkpoint_path,
    )

    # ── Training loop ─────────────────────────────────────────────
    history: dict = {
        "train_loss": [], "val_loss": [],
        "train_f1":   [], "val_f1":   [],
        "train_acc":  [], "val_acc":  [],
    }

    print("\n" + "=" * 65)
    print(f"  Starting training — max {args.epochs} epochs")
    print("=" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_preds, train_targets = run_epoch(
            model, train_loader, criterion, optimizer, device, "train", args.grad_clip)
        val_loss, val_preds, val_targets = run_epoch(
            model, val_loader, criterion, optimizer, device, "val", args.grad_clip)

        train_m = compute_metrics(train_preds, train_targets)
        val_m   = compute_metrics(val_preds,   val_targets)

        scheduler.step(val_loss)

        for key, val in [("train_loss", train_loss), ("val_loss", val_loss),
                         ("train_f1",   train_m["f1"]), ("val_f1",  val_m["f1"]),
                         ("train_acc",  train_m["accuracy"]), ("val_acc", val_m["accuracy"])]:
            history[key].append(val)

        print(
            f"Epoch {epoch:3d}/{args.epochs} [{time.time()-t0:.0f}s]  "
            f"Train — loss: {train_loss:.4f}  acc: {train_m['accuracy']:.4f}  "
            f"F1: {train_m['f1']:.4f}  |  "
            f"Val — loss: {val_loss:.4f}  acc: {val_m['accuracy']:.4f}  "
            f"F1: {val_m['f1']:.4f}"
        )

        if early_stopper.step(val_loss, model):
            print(f"\nEarly stopping triggered at epoch {epoch}. "
                  f"Best val_loss: {early_stopper.best_loss:.4f}")
            break

    # Restore best weights
    early_stopper.load_best(model)

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ── Final evaluation on test set ──────────────────────────────
    if test_loader is not None:
        print("\n" + "=" * 65)
        print("  Evaluating on test set")
        print("=" * 65)

        test_loss, test_preds, test_targets = run_epoch(
            model, test_loader, criterion, optimizer, device, "val", 0.0)
        test_m = compute_metrics(test_preds, test_targets)
        cm     = confusion_matrix(test_targets, test_preds)

        print(f"\nTest loss : {test_loss:.4f}")
        print(f"Accuracy  : {test_m['accuracy']*100:.2f}%")
        print(f"F1-score  : {test_m['f1']*100:.2f}%")
        print(f"Precision : {test_m['precision']*100:.2f}%")
        print(f"Recall    : {test_m['recall']*100:.2f}%")
        print(f"\nConfusion matrix:\n{cm}")
        print("\n" + classification_report(
            test_targets, test_preds,
            target_names=["false (clear)", "true (cloudy)"],
        ))

        results = {
            "test_loss": test_loss,
            **{f"test_{k}": v for k, v in test_m.items()},
            "confusion_matrix": cm.tolist(),
        }
        with open(output_dir / "test_results.json", "w") as f:
            json.dump(results, f, indent=2)
    else:
        print("\nNo 'test' folder found — skipping final evaluation.")

    print(f"\nBest checkpoint : {checkpoint_path}")
    print(f"Run artifacts   : {output_dir}/")


if __name__ == "__main__":
    main()