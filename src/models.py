"""
Model factory for ceilometer cloud detection.

Every model is ImageNet-pretrained with a 2-logit head (CrossEntropyLoss).
timm models are referenced by an explicit pretrained tag, so a timm upgrade
cannot silently swap the weights under a run.
"""

from typing import Dict, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

# name -> pinned timm model id
TIMM_MODELS: Dict[str, str] = {
    "convnext_tiny":  "convnext_tiny.fb_in22k_ft_in1k",
    "convnext_small": "convnext_small.fb_in22k_ft_in1k",
    "convnext_base":  "convnext_base.fb_in22k_ft_in1k",
    "swin_base":      "swin_base_patch4_window7_224.ms_in22k_ft_in1k",
    "vit":            "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    "efficientnet":   "efficientnet_b0.ra_in1k",
}


def get_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """
    Supported:
        'resnet50_tv'   : torchvision ResNet50 IMAGENET1K_V1 — the paper baseline.
        TIMM_MODELS keys: pinned timm checkpoints.

    drop_rate      : dropout before the classifier (timm head).
    drop_path_rate : stochastic depth — the intended regularizer for ConvNeXt/Swin/ViT.
    """
    if model_name == "resnet50_tv":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        if drop_rate > 0:
            model.fc = nn.Sequential(nn.Dropout(drop_rate), model.fc)
        return model

    if model_name not in TIMM_MODELS:
        raise ValueError(f"Model '{model_name}' not supported. Choose from: resnet50_tv, {', '.join(TIMM_MODELS)}")

    kwargs = {}
    if drop_rate > 0:
        kwargs["drop_rate"] = drop_rate
    if drop_path_rate > 0:
        kwargs["drop_path_rate"] = drop_path_rate
    return timm.create_model(TIMM_MODELS[model_name], pretrained=pretrained, num_classes=num_classes, **kwargs)


def apply_legacy_dropout(model: nn.Module, p: float) -> int:
    """
    LEGACY (ConvNeXt recipe up to commit 89cf8be): sets p on EVERY nn.Dropout
    in the model. For timm ConvNeXt this hits the head dropout *and* the two
    MLP dropouts in each of the 36 blocks (pretrained with p=0). Kept only to
    re-measure that recipe; new recipes use drop_rate / drop_path_rate.
    Returns the number of modified layers.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p
            n += 1
    return n


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        print(f"Device: cuda ({props.name}, {props.total_memory / 1e9:.1f} GB)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: mps")
    else:
        device = torch.device("cpu")
        print("Device: cpu")
    return device


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────
# Gradual unfreezing (legacy recipe)
# ─────────────────────────────────────────────

def _split_backbone_head(model: nn.Module) -> Tuple[Optional[nn.Module], List[nn.Module], nn.Module]:
    """(stem, ordered stages, head) for the supported architectures."""
    if hasattr(model, "stages") and hasattr(model, "head"):          # ConvNeXt (timm)
        return getattr(model, "stem", None), list(model.stages), model.head
    if hasattr(model, "blocks") and hasattr(model, "head"):          # ViT (timm)
        return getattr(model, "patch_embed", None), list(model.blocks), model.head
    if hasattr(model, "layers") and hasattr(model, "head"):          # Swin (timm)
        return getattr(model, "patch_embed", None), list(model.layers), model.head
    if hasattr(model, "fc") and all(hasattr(model, f"layer{i}") for i in range(1, 5)):  # ResNet
        stem = nn.ModuleList([model.conv1, model.bn1])
        return stem, [getattr(model, f"layer{i}") for i in range(1, 5)], model.fc
    raise ValueError(f"Don't know how to split backbone/head for {type(model).__name__}")


def get_unfrozen_params(
    model: nn.Module,
    epoch: int,
    lr_head: float,
    lr_backbone: float,
) -> Tuple[List[Dict], int]:
    """
    LEGACY gradual-unfreezing schedule (1-indexed epochs):
        phase 1 (epoch <= 10): head only
        phase 2 (epoch <= 30): head + last 2 stages
        phase 3              : everything
    Sets requires_grad accordingly and returns (param_groups, phase).
    """
    stem, stages, head = _split_backbone_head(model)
    phase = 1 if epoch <= 10 else 2 if epoch <= 30 else 3
    n_trainable_stages = {1: 0, 2: min(2, len(stages)), 3: len(stages)}[phase]

    for p in model.parameters():
        p.requires_grad = False
    trainable_backbone = list(stages[len(stages) - n_trainable_stages:]) if n_trainable_stages else []
    if phase == 3 and stem is not None:
        trainable_backbone.insert(0, stem)
    for module in [head, *trainable_backbone]:
        for p in module.parameters():
            p.requires_grad = True

    groups = [{"params": list(head.parameters()), "lr": lr_head}]
    backbone_params = [p for m in trainable_backbone for p in m.parameters()]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone})
    return groups, phase
