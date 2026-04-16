from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from dense_net.common import NUM_CLASSES


def configure_torch_home(torch_home: Path | None) -> None:
    if torch_home is None:
        return

    torch_home.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_home))


def build_densenet_model(
    model_name: str = "densenet121",
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    pretrained: bool = True,
    torch_home: Path | None = None,
) -> nn.Module:
    configure_torch_home(torch_home)

    if model_name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
    elif model_name == "densenet169":
        weights = models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet169(weights=weights)
    else:
        raise ValueError(f"Unsupported DenseNet model: {model_name}")

    if freeze_backbone:
        set_feature_extractor_trainable(model, trainable=False)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def set_feature_extractor_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.features.parameters():
        param.requires_grad = trainable


def get_head_parameters(model: nn.Module):
    return model.classifier.parameters()
