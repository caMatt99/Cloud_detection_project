"""
Model definitions for ceilometer backscatter cloud detection.
Supports baseline (ResNet50) and new architectures (ConvNeXt, Swin Transformer).
All models are pretrained on ImageNet and adapted for binary classification.
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple


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