"""
Shared training engine for the notebook experiments (02_baseline, 03_new_model).
One epoch of training/evaluation plus the per-epoch W&B logging, so both
notebooks import the same code instead of duplicating it.
"""

import numpy as np
import torch
import torch.optim as optim
import wandb

from models import get_unfrozen_params


def compute_class_weights(train_dataset) -> torch.Tensor:
    """
    Computes inverse-frequency class weights from a train ImageFolder dataset,
    to pass to nn.CrossEntropyLoss(weight=...).

    The ceilometer train set is imbalanced ~2:1 true:false (515 vs 255 images —
    see notebooks/01_eda.ipynb, sections 2 and 7), which the paper doesn't mention
    or correct for. Weighting the minority class ("false") more heavily keeps the
    loss from being dominated by the majority class.

    Args:
        train_dataset: torchvision.datasets.ImageFolder (e.g. train_loader.dataset),
                       whose `.targets` is a list of int labels (false=0, true=1,
                       per ImageFolder's alphabetical class ordering).

    Returns:
        torch.Tensor: [weight_false, weight_true], in ImageFolder label order,
        ready to pass as nn.CrossEntropyLoss(weight=...).
    """

    targets = train_dataset.targets
    true_count = sum(1 for t in targets if t == 1)
    false_count = sum(1 for t in targets if t == 0)
    total = true_count + false_count

    weight_true = false_count / total
    weight_false = true_count / total

    print(
        f"[compute_class_weights] true: {true_count} (weight={weight_true:.3f})  "
        f"false: {false_count} (weight={weight_false:.3f})"
    )

    return torch.tensor([weight_false, weight_true], dtype=torch.float32)


def build_phase_optimizer(
    model,
    epoch,
    total_epochs,
    lr_head,
    lr_backbone,
    weight_decay=1e-5,
    optimizer_name="adam",
    momentum=0.9,
):
    """
    Rebuilds the optimizer for the current epoch according to the gradual
    unfreezing schedule in models.get_unfrozen_params(). Call this once per
    epoch, before train_one_epoch, so newly-unfrozen parameters are picked up
    by the optimizer (a plain PyTorch optimizer will otherwise ignore params
    that were frozen when it was constructed).

    Args:
        model         (nn.Module) : model returned by get_model().
        epoch         (int)       : current epoch (1-indexed).
        total_epochs  (int)       : total planned epochs.
        lr_head       (float)     : learning rate for the classification head.
                                    Typically config["lr"].
        lr_backbone   (float)     : learning rate for unfrozen backbone params.
                                    Typically config["lr"] / 10.
        weight_decay  (float)     : weight decay applied to both param groups.
        optimizer_name(str)       : "adam" or "sgd".
        momentum      (float)     : momentum, used only when optimizer_name="sgd".

    Returns:
        torch.optim.Optimizer: freshly built optimizer over the currently
        unfrozen parameters, with head/backbone learning rates set via
        param_groups.
    """

    param_groups = get_unfrozen_params(
        model, epoch, total_epochs, lr_head=lr_head, lr_backbone=lr_backbone
    )

    if optimizer_name == "sgd":
        return optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
    return optim.Adam(param_groups, weight_decay=weight_decay)


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
