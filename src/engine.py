"""
Shared training engine for the notebook experiments (02_baseline, 03_new_model).
One epoch of training/evaluation plus the per-epoch W&B logging, so both
notebooks import the same code instead of duplicating it.
"""

import numpy as np
import torch
import wandb


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return epoch_loss, epoch_acc, all_preds, all_labels


@torch.no_grad()
def evaluate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    return epoch_loss, epoch_acc, all_preds, all_labels


def log_epoch_to_wandb(
    epoch, train_loss, train_acc, val_loss, val_acc, val_preds, val_labels, class_names
):
    """Logs per-epoch train/val loss, accuracy, and the val confusion matrix to W&B."""
    wandb.log(
        {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/confusion_matrix": wandb.plot.confusion_matrix(
                preds=val_preds,
                y_true=val_labels,
                class_names=class_names,
            ),
        },
        step=epoch,
    )
