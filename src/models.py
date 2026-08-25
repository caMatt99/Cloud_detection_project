"""
Model definitions for ceilometer backscatter cloud detection.
Supports baseline (ResNet50) and new architectures (ConvNeXt, Swin Transformer).
All models are pretrained on ImageNet and adapted for binary classification.
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, List, Optional, Tuple


def get_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True
) -> nn.Module:
    """
    Returns a pretrained model adapted for binary cloud classification.

    Supported models:
        - 'resnet50'       : baseline from the paper (89.57% accuracy)
        - 'convnext_base'  : new architecture to benchmark
        - 'swin_base'      : alternative new architecture to benchmark
        - 'vgg16'          : from the paper
        - 'efficientnet'   : from the paper
        - 'inceptionv3'    : from the paper
        - 'vit'            : Vision Transformer from the paper

    Args:
        model_name  (str)  : name of the architecture to load.
        num_classes (int)  : number of output classes. Default 2 (true/false).
        pretrained  (bool) : whether to load ImageNet pretrained weights.
                             Default True as in the paper.

    Returns:
        nn.Module: model ready for training on the ceilometer dataset.
    
    Raises:
        ValueError: if model_name is not supported.
    """

    if model_name == 'resnet50':
        # baseline del paper — residual connections, 89.57% accuracy
        model: nn.Module = timm.create_model(
            'resnet50',
            pretrained=pretrained,
            num_classes=num_classes
        )

    elif model_name == 'convnext_base':
        # architettura moderna CNN (2022) — non testata nel paper
        # combina i principi dei transformer con struttura convoluzionale
        model = timm.create_model(
            'convnext_base',
            pretrained=pretrained,
            num_classes=num_classes
        )

    elif model_name == 'swin_base':
        # Swin Transformer — transformer gerarchico (2021)
        # processa patch locali prima di espandere il contesto globale
        # potenzialmente più adatto dei ViT plain per pattern spaziali
        model = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            num_classes=num_classes
        )

    elif model_name == 'vgg16':
        # dal paper — architettura classica, strong baseline
        model = timm.create_model(
            'vgg16',
            pretrained=pretrained,
            num_classes=num_classes
        )

    elif model_name == 'efficientnet':
        # dal paper — scala width/depth/resolution in modo bilanciato
        model = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=num_classes
        )

    elif model_name == 'inceptionv3':
        # dal paper — inception modules per feature multi-scala
        model = timm.create_model(
            'inception_v3',
            pretrained=pretrained,
            num_classes=num_classes
        )

    elif model_name == 'vit':
        # dal paper — Vision Transformer, 89.36% accuracy
        # opera direttamente su patch dell'immagine
        model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=pretrained,
            num_classes=num_classes
        )

    else:
        raise ValueError(
            f"Model '{model_name}' not supported. "
            f"Choose from: resnet50, convnext_base, swin_base, "
            f"vgg16, efficientnet, inceptionv3, vit"
        )

    return model


def get_device() -> torch.device:
    """
    Returns the available device (GPU if available, otherwise CPU).

    Returns:
        torch.device: 'cuda' if GPU is available, 'cpu' otherwise.
    """

    device: torch.device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return device


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Counts total and trainable parameters of a model.
    Useful to compare model complexity across architectures.

    Args:
        model (nn.Module): the PyTorch model to inspect.

    Returns:
        Tuple[int, int]:
            - total_params     : all parameters
            - trainable_params : only parameters that will be updated during training
    """

    total_params: int = sum(p.numel() for p in model.parameters())
    trainable_params: int = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return total_params, trainable_params


def _split_backbone_head(
    model: nn.Module
) -> Tuple[Optional[nn.Module], List[nn.Module], nn.Module]:
    """
    Splits a timm model (as returned by get_model()) into its stem,
    an ordered list of backbone stages, and its classification head.

    The "stages" list is what gradual unfreezing operates on:
        - ConvNeXt / timm hierarchical CNNs (`model.stages`, 4 stages: 0-3)
        - Swin Transformer                  (`model.layers`)
        - ViT                                (`model.blocks`)
        - ResNet                             (`model.layer1..layer4`)
        - VGG / EfficientNet (features-based) (`model.features`)

    Args:
        model (nn.Module): model returned by get_model().

    Returns:
        Tuple[Optional[nn.Module], List[nn.Module], nn.Module]:
            - stem   : early feature-extraction layers before the first stage
                       (None if the architecture has no separate stem).
            - stages : ordered list of backbone blocks, one entry per stage.
            - head   : the classification head module.
    """

    if hasattr(model, 'stages') and hasattr(model, 'head'):
        # ConvNeXt (timm): model.stages has exactly 4 stages (0, 1, 2, 3),
        # what the project refers to as "model.features".
        return getattr(model, 'stem', None), list(model.stages), model.head

    if hasattr(model, 'blocks') and hasattr(model, 'head'):
        # ViT (timm): transformer blocks are the "stages".
        return getattr(model, 'patch_embed', None), list(model.blocks), model.head

    if hasattr(model, 'layers') and hasattr(model, 'head'):
        # Swin Transformer (timm)
        return getattr(model, 'patch_embed', None), list(model.layers), model.head

    if hasattr(model, 'fc') and all(hasattr(model, f'layer{i}') for i in range(1, 5)):
        # ResNet (timm)
        stem = nn.ModuleList(
            m for m in (getattr(model, 'conv1', None), getattr(model, 'bn1', None))
            if m is not None
        )
        stages = [getattr(model, f'layer{i}') for i in range(1, 5)]
        return stem, stages, model.fc

    if hasattr(model, 'features'):
        # VGG / EfficientNet (timm) — features-based backbone
        head = getattr(model, 'head', None) or getattr(model, 'classifier', None)
        return None, list(model.features), head

    raise ValueError(
        f"Don't know how to split backbone/head for model of type {type(model)}."
    )


def get_unfrozen_params(
    model: nn.Module,
    epoch: int,
    total_epochs: int,
    lr_head: float = 1e-4,
    lr_backbone: float = 1e-5,
) -> List[Dict]:
    """
    Gradual unfreezing schedule. Sets requires_grad on model parameters
    according to which training phase `epoch` falls into, then returns
    param_groups ready to hand to an optimizer (e.g. optim.Adam(param_groups)).

    Phases (thresholds are inclusive of the phase's last epoch):
        - Phase 1 (epoch <= 10)                 : backbone frozen, head only.
        - Phase 2 (10 < epoch <= 30)             : last 2 backbone stages
                                                    unfrozen (e.g. ConvNeXt
                                                    model.stages[2:4]), head
                                                    stays unfrozen.
        - Phase 3 (epoch > 30)                   : everything unfrozen.

    Args:
        model        (nn.Module): model returned by get_model().
        epoch        (int)      : current epoch (1-indexed).
        total_epochs (int)      : total planned epochs (used only for logging).
        lr_head      (float)    : learning rate for the classification head.
        lr_backbone  (float)    : learning rate for unfrozen backbone params.
                                  Typically args.lr / 10.

    Returns:
        List[Dict]: param_groups, e.g.
            [{'params': [...head params...], 'lr': lr_head},
             {'params': [...backbone params...], 'lr': lr_backbone}]
    """

    stem, stages, head = _split_backbone_head(model)

    def set_requires_grad(module: Optional[nn.Module], flag: bool) -> None:
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = flag

    num_stages = len(stages)
    unfrozen_stage_count = min(2, num_stages)
    late_stages = stages[num_stages - unfrozen_stage_count:]
    early_stages = stages[:num_stages - unfrozen_stage_count]

    if epoch <= 10:
        phase = 1
        set_requires_grad(stem, False)
        for stage in stages:
            set_requires_grad(stage, False)
        set_requires_grad(head, True)

    elif epoch <= 30:
        phase = 2
        set_requires_grad(stem, False)
        for stage in early_stages:
            set_requires_grad(stage, False)
        for stage in late_stages:
            set_requires_grad(stage, True)
        set_requires_grad(head, True)

    else:
        phase = 3
        set_requires_grad(stem, True)
        for stage in stages:
            set_requires_grad(stage, True)
        set_requires_grad(head, True)

    head_params = [p for p in head.parameters() if p.requires_grad]
    backbone_modules = ([stem] if stem is not None else []) + stages
    backbone_params = [
        p for module in backbone_modules for p in module.parameters() if p.requires_grad
    ]

    param_groups: List[Dict] = [{'params': head_params, 'lr': lr_head}]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': lr_backbone})

    print(
        f"[get_unfrozen_params] epoch {epoch}/{total_epochs} -> phase {phase}  "
        f"head: {len(head_params)} tensors @ lr={lr_head}  |  "
        f"backbone: {len(backbone_params)} tensors @ lr={lr_backbone}"
    )

    return param_groups


def apply_dropout(model: nn.Module, model_name: str, dropout_rate: float) -> nn.Module:
    """
    Applies dropout_rate to a model's existing nn.Dropout layers, in an
    architecture-aware way, and returns the modified model.

    - ConvNeXt ('convnext_base'): sets dropout_rate on every nn.Dropout found
      inside the backbone stages (model.stages) and inside the head.
    - ResNet ('resnet50'): timm's default resnet50 head has no dropout, so a
      nn.Dropout(dropout_rate) is inserted right before the final fc layer.
    - ViT ('vit'): sets dropout_rate on every nn.Dropout found inside the
      transformer blocks (model.blocks).
    - Any other architecture: best-effort fallback that sets dropout_rate on
      every nn.Dropout module found anywhere in the model.

    Args:
        model        (nn.Module): model returned by get_model(), modified in place.
        model_name   (str)      : the model_name passed to get_model().
        dropout_rate (float)    : dropout probability to apply.

    Returns:
        nn.Module: the same model instance, with dropout applied.
    """

    def set_dropout(module: nn.Module, rate: float) -> int:
        count = 0
        for submodule in module.modules():
            if isinstance(submodule, nn.Dropout):
                submodule.p = rate
                count += 1
        return count

    if model_name == 'convnext_base':
        n = set_dropout(model.stages, dropout_rate) + set_dropout(model.head, dropout_rate)
        print(f"[apply_dropout] convnext_base: set p={dropout_rate} on {n} Dropout layers")

    elif model_name == 'resnet50':
        if not isinstance(model.fc, nn.Sequential):
            model.fc = nn.Sequential(nn.Dropout(p=dropout_rate), model.fc)
            print(f"[apply_dropout] resnet50: inserted nn.Dropout(p={dropout_rate}) before fc")
        else:
            n = set_dropout(model.fc, dropout_rate)
            print(f"[apply_dropout] resnet50: set p={dropout_rate} on {n} Dropout layers")

    elif model_name == 'vit':
        n = set_dropout(model.blocks, dropout_rate)
        print(f"[apply_dropout] vit: set p={dropout_rate} on {n} Dropout layers")

    else:
        n = set_dropout(model, dropout_rate)
        print(f"[apply_dropout] {model_name}: set p={dropout_rate} on {n} Dropout layers (fallback)")

    return model