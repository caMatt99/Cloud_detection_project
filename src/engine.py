"""
Model-agnostic training primitives: one epoch of train/eval, optimizer
factory, early stopping and best-checkpoint tracking.
"""

from contextlib import nullcontext
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def autocast_ctx(device: torch.device, enabled: bool):
    """fp16 autocast on CUDA only; no-op elsewhere."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_optimizer(params: Iterable, name: str, lr: float, weight_decay: float, momentum: float = 0.9):
    """`params` can be a parameter iterable or a list of param-group dicts."""
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'")


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_clip: float = 0.0,
) -> Tuple[float, float]:
    """Returns (mean loss, accuracy in [0, 1])."""
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()
    total_loss, correct, n = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        n += labels.size(0)

    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    amp: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Returns (mean loss, accuracy, P(class=1) per sample, labels)."""
    model.eval()
    total_loss, probs, labels_all = 0.0, [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_ctx(device, amp):
            logits = model(images)
        logits = logits.float()
        total_loss += criterion(logits, labels).item() * labels.size(0)
        probs.append(torch.softmax(logits, 1)[:, 1].cpu())
        labels_all.append(labels.cpu())

    probs = torch.cat(probs).numpy()
    labels_np = torch.cat(labels_all).numpy()
    acc = float(((probs >= 0.5).astype(int) == labels_np).mean())
    return total_loss / len(labels_np), acc, probs, labels_np


class BestCheckpoint:
    """
    Keeps an in-memory (CPU) copy of the weights at the best epoch according
    to `monitor` ('val_loss' -> lower is better, 'val_acc' -> higher is
    better; ties on val_acc are broken by lower val_loss).
    """

    def __init__(self, monitor: str = "val_loss"):
        if monitor not in ("val_loss", "val_acc"):
            raise ValueError("monitor must be 'val_loss' or 'val_acc'")
        self.monitor = monitor
        self.best_key = None
        self.best_epoch = 0
        self.state_dict = None

    def _key(self, val_loss: float, val_acc: float):
        return (val_loss,) if self.monitor == "val_loss" else (-val_acc, val_loss)

    def update(self, model: nn.Module, epoch: int, val_loss: float, val_acc: float) -> bool:
        key = self._key(val_loss, val_acc)
        if self.best_key is None or key < self.best_key:
            self.best_key, self.best_epoch = key, epoch
            self.state_dict = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            return True
        return False

    def restore(self, model: nn.Module) -> None:
        model.load_state_dict(self.state_dict)


class EarlyStopping:
    """
    mode='none'    : never stops.
    mode='paper'   : stops when |val_loss - previous val_loss| < delta for
                     `patience` consecutive epochs (Chisari et al., Sec. 4.2).
    mode='standard': stops when val_loss has not improved on its best by more
                     than `delta` for `patience` consecutive epochs.
    """

    def __init__(self, mode: str = "none", patience: int = 10, delta: float = 0.0):
        if mode not in ("none", "paper", "standard"):
            raise ValueError("mode must be 'none', 'paper' or 'standard'")
        self.mode, self.patience, self.delta = mode, patience, delta
        self.counter = 0
        self.prev = None
        self.best = float("inf")

    def step(self, val_loss: float) -> bool:
        if self.mode == "none":
            return False
        if self.mode == "paper":
            if self.prev is not None and abs(val_loss - self.prev) < self.delta:
                self.counter += 1
            else:
                self.counter = 0
            self.prev = val_loss
        else:
            if val_loss < self.best - self.delta:
                self.best, self.counter = val_loss, 0
            else:
                self.counter += 1
        return self.counter >= self.patience
