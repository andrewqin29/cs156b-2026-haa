"""
Train EfficientNet on full-view CheXpert manifests with scaled MSE.

Label convention:
  1.0    positive
 -1.0    negative
  0.0    unknown / missing label, masked out
-999.0   missing sentinel from preprocessing, masked out
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

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
POSITIVE = 1.0
NEGATIVE = -1.0
SENTINEL = -999.0

FULL_512_ROOT = Path("/resnick/groups/CS156b/from_central/2026/haa/full_512_nans_-999_JPG")
EFFICIENT_NET_RESULTS = Path("/resnick/groups/CS156b/from_central/2026/haa/askumar/efficient_net_results")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_path_column(df: pd.DataFrame) -> str:
    if "preprocessed_path" in df.columns:
        return "preprocessed_path"
    if "abs_path" in df.columns:
        return "abs_path"
    raise ValueError("Manifest must contain either 'preprocessed_path' or 'abs_path'.")


def resolve_existing_image_path(row: pd.Series) -> str | None:
    for column in ("preprocessed_path", "abs_path"):
        if column not in row.index:
            continue
        candidate = row[column]
        if pd.isna(candidate):
            continue
        candidate = str(candidate)
        if Path(candidate).exists():
            return candidate
    return None


def _validate_manifest(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing_cols = [column for column in LABEL_COLS if column not in df.columns]
    if missing_cols:
        raise ValueError(f"{name}: missing label columns: {missing_cols}")

    _pick_path_column(df)
    df = df.copy()
    for column in LABEL_COLS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[LABEL_COLS] = df[LABEL_COLS].where(~df[LABEL_COLS].isna(), other=SENTINEL)
    df["_image_path"] = df.apply(resolve_existing_image_path, axis=1)

    before = len(df)
    df = df[df["_image_path"].notna()].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] {name}: dropped {dropped} rows with missing/nonexistent image files")

    if len(df) == 0:
        raise ValueError(f"{name}: no usable rows after validation")
    return df


def _known_mask(raw_labels: np.ndarray) -> np.ndarray:
    return (raw_labels == POSITIVE) | (raw_labels == NEGATIVE)


def _binary_from_raw(raw_labels: np.ndarray) -> np.ndarray:
    return (raw_labels == POSITIVE).astype(np.float32)


def compute_label_variance(train_df: pd.DataFrame) -> torch.Tensor:
    variances: list[float] = []
    print("Per-class label variance over known {-1, 1} labels:")
    for column in LABEL_COLS:
        vals = train_df[column].to_numpy(dtype=np.float32)
        known = vals[_known_mask(vals)]
        variance = float(np.var(known, ddof=1)) if len(known) >= 2 else 1.0
        variance = max(variance, 1e-6)
        variances.append(variance)
        print(f"  {column:<35} variance={variance:.4f} known={len(known):>7}")
    return torch.tensor(variances, dtype=torch.float32)


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.zeros_like(sorted_scores, dtype=np.float64)
    idx = 0
    while idx < len(sorted_scores):
        end = idx
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[idx]:
            end += 1
        ranks[idx : end + 1] = (idx + end + 2) / 2.0
        idx = end + 1

    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks
    return float((full_ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def compute_auc(scores: np.ndarray, raw_labels: np.ndarray) -> tuple[float, dict[str, float]]:
    known = _known_mask(raw_labels)
    binary = _binary_from_raw(raw_labels)
    per_label: dict[str, float] = {}
    for idx, label in enumerate(LABEL_COLS):
        mask = known[:, idx]
        if mask.sum() < 2 or len(np.unique(binary[mask, idx])) < 2:
            per_label[label] = float("nan")
            continue
        per_label[label] = _binary_auc(binary[mask, idx], scores[mask, idx])
    valid = [value for value in per_label.values() if not np.isnan(value)]
    return float(np.mean(valid)) if valid else float("nan"), per_label


def compute_scaled_mse(scores: np.ndarray, raw_labels: np.ndarray, variance: torch.Tensor) -> tuple[float, dict[str, float]]:
    variance_np = variance.cpu().numpy()
    per_label: dict[str, float] = {}
    scaled_values: list[float] = []
    for idx, label in enumerate(LABEL_COLS):
        mask = _known_mask(raw_labels[:, idx])
        if mask.sum() < 1:
            per_label[label] = float("nan")
            continue
        mse = float(((scores[mask, idx] - raw_labels[mask, idx]) ** 2).mean())
        scaled = mse / max(float(variance_np[idx]), 1e-6)
        per_label[label] = scaled
        scaled_values.append(scaled)
    return float(np.mean(scaled_values)) if scaled_values else float("nan"), per_label


class XrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(str(row["_image_path"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        raw_labels = row[LABEL_COLS].to_numpy(dtype=np.float32)
        mask = _known_mask(raw_labels).astype(np.float32)
        safe_labels = np.where(mask.astype(bool), raw_labels, 0.0).astype(np.float32)
        return img, torch.tensor(safe_labels), torch.tensor(mask), torch.tensor(raw_labels)


def get_transforms(image_size: int, train: bool):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(7),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def configure_torch_home(torch_home: Path | None) -> None:
    if torch_home is None:
        return
    torch_home.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_home))


def set_feature_extractor_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.features.parameters():
        param.requires_grad = trainable


def get_head_parameters(model: nn.Module):
    return model.classifier.parameters()


def build_model(
    model_name: str = "efficientnet_b0",
    dropout: float = 0.3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    torch_home: Path | None = None,
) -> nn.Module:
    configure_torch_home(torch_home)
    try:
        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b0(weights=weights)
        elif model_name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.efficientnet_b3(weights=weights)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
    except Exception as exc:
        warnings.warn(f"Could not load pretrained weights ({exc}); falling back to random init.")
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
        elif model_name == "efficientnet_b3":
            model = models.efficientnet_b3(weights=None)
        else:
            raise

    if freeze_backbone:
        set_feature_extractor_trainable(model, trainable=False)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, NUM_CLASSES),
        nn.Tanh(),
    )
    return model


def scaled_mse_loss(preds, labels, mask, variance):
    scaled = ((preds - labels) ** 2) / variance.to(preds.device).view(1, -1).clamp(min=1e-6)
    masked = scaled * mask
    per_class_counts = mask.sum(dim=0)
    per_class_loss = masked.sum(dim=0) / per_class_counts.clamp(min=1)
    valid_classes = per_class_counts > 0
    if valid_classes.any():
        return per_class_loss[valid_classes].mean()
    return masked.sum() * 0.0


def run_epoch(model, loader, optimizer, device, training: bool, variance: torch.Tensor):
    model.train() if training else model.eval()
    total_loss = 0.0
    all_scores, all_raw = [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels, masks, raw_labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            preds = model(imgs)
            loss = scaled_mse_loss(preds, labels, masks, variance)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            all_scores.append(preds.detach().cpu())
            all_raw.append(raw_labels.detach().cpu())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    scores = torch.cat(all_scores).numpy()
    raw = torch.cat(all_raw).numpy()
    auc, per_label_auc = compute_auc(scores, raw)
    scaled_mse, per_label_scaled_mse = compute_scaled_mse(scores, raw, variance)
    return avg_loss, scaled_mse, auc, per_label_scaled_mse, per_label_auc


def parse_args() -> argparse.Namespace:
    root = FULL_512_ROOT / "manifests_preprocessed"
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=Path, default=root / "train_manifest_preprocessed.csv")
    parser.add_argument("--val_csv", type=Path, default=root / "val_manifest_preprocessed.csv")
    parser.add_argument("--model_name", choices=["efficientnet_b0", "efficientnet_b3"], default="efficientnet_b0")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head_lr_multiplier", type=float, default=10.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--torch_home", type=Path, default=Path("/resnick/groups/CS156b/from_central/2026/haa/askumar/.torch_cache"))
    parser.add_argument("--runs_base", type=Path, default=EFFICIENT_NET_RESULTS / "runs")
    parser.add_argument("--run_name", type=str, default="efficientnet_b0_full512_scaled_mse_mask0_baseline")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--limit_train_rows", type=int, default=0, help="Optional smoke-test row limit.")
    parser.add_argument("--limit_val_rows", type=int, default=0, help="Optional smoke-test row limit.")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    args.runs_base.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        return args.runs_base / args.run_name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.runs_base / f"{args.model_name}_scaled_mse_{args.image_size}px_{ts}"


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def checkpoint_payload(args, epoch, model, val_scaled_mse, val_auc, per_label_scaled_mse, per_label_auc, variance):
    return {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "val_scaled_mse": val_scaled_mse,
        "val_auc": val_auc,
        "per_label_scaled_mse": per_label_scaled_mse,
        "per_label_auc": per_label_auc,
        "label_variance": variance.cpu().tolist(),
        "label_cols": LABEL_COLS,
        "model_name": args.model_name,
        "image_size": args.image_size,
        "dropout": args.dropout,
        "output_activation": "tanh",
        "label_policy": "mask values 0 and -999; train/evaluate known -1/1 labels only",
        "args": vars(args),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = resolve_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in (args.train_csv, args.val_csv):
        if not path.exists():
            raise FileNotFoundError(f"Missing input CSV: {path}")

    train_df = _validate_manifest(pd.read_csv(args.train_csv), "train_csv")
    val_df = _validate_manifest(pd.read_csv(args.val_csv), "val_csv")
    if args.limit_train_rows > 0:
        train_df = train_df.head(args.limit_train_rows).copy()
    if args.limit_val_rows > 0:
        val_df = val_df.head(args.limit_val_rows).copy()
    variance = compute_label_variance(train_df)

    with open(out_dir / "run_config.txt", "w", encoding="utf-8") as handle:
        for key, value in sorted(vars(args).items()):
            handle.write(f"{key}={value!r}\n")
        handle.write("label_policy='mask 0 and -999; known labels are -1 and 1'\n")
        handle.write(f"label_variance={variance.tolist()!r}\n")

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        handle.write("epoch,train_loss,train_scaled_mse,train_auc,val_loss,val_scaled_mse,val_auc\n")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Output directory: {out_dir}")
    print(f"Device: {device}")
    print(f"Usable rows -> train: {len(train_df):,}, val: {len(val_df):,}")

    train_ds = XrayDataset(train_df, transform=get_transforms(args.image_size, train=True))
    val_ds = XrayDataset(val_df, transform=get_transforms(args.image_size, train=False))
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    model = build_model(
        model_name=args.model_name,
        dropout=args.dropout,
        pretrained=True,
        freeze_backbone=args.warmup_epochs > 0,
        torch_home=args.torch_home,
    ).to(device)

    if args.warmup_epochs > 0:
        optimizer = torch.optim.AdamW(get_head_parameters(model), lr=args.lr * args.head_lr_multiplier, weight_decay=args.weight_decay)
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_scaled_mse = float("inf")
    epochs_no_improve = 0
    best_path = out_dir / "best_model.pt"
    final_path = out_dir / "final_model.pt"
    best_metrics_path = out_dir / "best_metrics.json"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if args.warmup_epochs > 0 and epoch == args.warmup_epochs + 1:
            print("\nUnfreezing EfficientNet backbone")
            set_feature_extractor_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs - args.warmup_epochs, 1))

        train_loss, train_scaled_mse, train_auc, train_per_scaled_mse, train_per_auc = run_epoch(
            model, train_loader, optimizer, device, training=True, variance=variance
        )
        val_loss, val_scaled_mse, val_auc, val_per_scaled_mse, val_per_auc = run_epoch(
            model, val_loader, optimizer, device, training=False, variance=variance
        )
        if scheduler is not None and (args.warmup_epochs == 0 or epoch > args.warmup_epochs):
            scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_scaled_mse={train_scaled_mse:.4f} train_auc={train_auc:.4f} "
            f"val_scaled_mse={val_scaled_mse:.4f} val_auc={val_auc:.4f} [{elapsed:.0f}s]"
        )
        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch},{train_loss:.6f},{train_scaled_mse:.6f},{train_auc:.6f},"
                f"{val_loss:.6f},{val_scaled_mse:.6f},{val_auc:.6f}\n"
            )

        torch.save(checkpoint_payload(args, epoch, model, val_scaled_mse, val_auc, val_per_scaled_mse, val_per_auc, variance), final_path)
        if val_scaled_mse < best_scaled_mse:
            best_scaled_mse = val_scaled_mse
            epochs_no_improve = 0
            torch.save(checkpoint_payload(args, epoch, model, val_scaled_mse, val_auc, val_per_scaled_mse, val_per_auc, variance), best_path)
            save_json(
                {
                    "best_epoch": epoch,
                    "best_val_scaled_mse": val_scaled_mse,
                    "best_val_auc": val_auc,
                    "val_per_label_scaled_mse": val_per_scaled_mse,
                    "val_per_label_auc": val_per_auc,
                    "train_per_label_scaled_mse": train_per_scaled_mse,
                    "train_per_label_auc": train_per_auc,
                    "label_policy": "mask 0 and -999; known labels are -1 and 1",
                },
                best_metrics_path,
            )
            print(f"  Saved new best checkpoint: {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"Training complete. Best val scaled MSE: {best_scaled_mse:.4f}")


if __name__ == "__main__":
    main()
