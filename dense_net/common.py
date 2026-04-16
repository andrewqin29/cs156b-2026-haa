from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

LABEL_COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

NUM_CLASSES = len(LABEL_COLS)
MISSING_VALUE = -999.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def masked_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    element_loss = loss_fn(logits, labels)
    masked_loss = element_loss * mask
    denom = mask.sum().clamp(min=1)
    return masked_loss.sum() / denom


def compute_multilabel_auc(
    probabilities: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
) -> tuple[float, dict[str, float]]:
    per_label_auc: dict[str, float] = {}

    for index, label_name in enumerate(LABEL_COLS):
        valid_mask = masks[:, index].astype(bool)
        if valid_mask.sum() < 2:
            continue

        y_true = labels[valid_mask, index]
        if len(np.unique(y_true)) < 2:
            continue

        auc = roc_auc_score(y_true, probabilities[valid_mask, index])
        if not np.isnan(auc):
            per_label_auc[label_name] = float(auc)

    mean_auc = float(np.mean(list(per_label_auc.values()))) if per_label_auc else float("nan")
    return mean_auc, per_label_auc


def compute_pos_weight_from_df(
    df,
    missing_value: float = MISSING_VALUE,
    max_pos_weight: float = 20.0,
) -> torch.Tensor:
    weights: list[float] = []

    for label in LABEL_COLS:
        values = df[label].astype(np.float32).to_numpy()
        valid = values != missing_value
        valid_values = values[valid]

        positives = float((valid_values == 1.0).sum())
        negatives = float((valid_values == 0.0).sum())

        if positives <= 0.0 or negatives <= 0.0:
            weights.append(1.0)
            continue

        weights.append(min(negatives / positives, max_pos_weight))

    return torch.tensor(weights, dtype=torch.float32)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
