"""
Train EfficientNet (multi-label) on CheXpert manifests.

Expected input manifests are produced by efficient_net/data_preprocessing scripts.
Uses `preprocessed_path` if present, otherwise falls back to `abs_path`.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
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

def _progress(iterable, leave=False):
    return iterable


def _pick_path_column(df: pd.DataFrame) -> str:
    if "preprocessed_path" in df.columns:
        return "preprocessed_path"
    if "abs_path" in df.columns:
        return "abs_path"
    raise ValueError("Manifest must contain either 'preprocessed_path' or 'abs_path'.")


def _validate_manifest(df: pd.DataFrame, missing_value: float, name: str) -> pd.DataFrame:
    missing_cols = [c for c in LABEL_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{name}: missing label columns: {missing_cols}")

    path_col = _pick_path_column(df)

    for c in LABEL_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[LABEL_COLS] = df[LABEL_COLS].fillna(missing_value)

    before = len(df)
    df = df[df[path_col].notna()].copy()
    df[path_col] = df[path_col].astype(str)

    exists = df[path_col].map(lambda p: Path(p).exists())
    df = df[exists].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] {name}: dropped {dropped} rows with missing/nonexistent image files")

    if len(df) == 0:
        raise ValueError(f"{name}: no usable rows after validation")

    return df


def compute_pos_weight_tensor(
    train_df: pd.DataFrame, uncertain_policy: str, missing_value: float
) -> torch.Tensor:
    """Per-class neg/pos ratio for BCEWithLogitsLoss(pos_weight=...)."""
    arr = train_df[LABEL_COLS].values.astype(np.float64)
    mv = missing_value
    if uncertain_policy == "mask":
        labels = np.where(arr == mv, 0.0, arr)
        m = arr != mv
    elif uncertain_policy == "uzeros":
        labels = np.where(arr == mv, 0.0, arr)
        m = np.ones_like(arr, dtype=bool)
    elif uncertain_policy == "uones":
        labels = np.where(arr == mv, 1.0, arr)
        m = np.ones_like(arr, dtype=bool)
    else:
        raise ValueError(f"Unknown uncertain_policy: {uncertain_policy}")

    weights: list[float] = []
    for c in range(NUM_CLASSES):
        mc = m[:, c]
        if mc.sum() < 1:
            weights.append(1.0)
            continue
        pos = float(((labels[:, c] == 1.0) & mc).sum())
        neg = float(((labels[:, c] == 0.0) & mc).sum())
        weights.append(neg / max(pos, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC without sklearn using rank statistic."""
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


class XrayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        missing_value: float = -999.0,
        uncertain_policy: str = "mask",
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.missing_value = missing_value
        self.uncertain_policy = uncertain_policy

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.get("preprocessed_path", row.get("abs_path"))

        if pd.isna(img_path):
            raise ValueError(f"Row {idx} has no usable image path.")

        img = Image.open(str(img_path)).convert("RGB")
        if self.transform:
            img = self.transform(img)

        raw_labels = row[LABEL_COLS].values.astype(np.float32)
        mv = self.missing_value

        if self.uncertain_policy == "mask":
            mask = (raw_labels != mv).astype(np.float32)
            labels = np.where(raw_labels == mv, 0.0, raw_labels).astype(np.float32)
        elif self.uncertain_policy == "uzeros":
            # U-zeros: treat uncertain/missing sentinel as negative (0), supervise all classes.
            # Note: manifests store both original NaN and -1 as the same sentinel.
            labels = np.where(raw_labels == mv, 0.0, raw_labels).astype(np.float32)
            mask = np.ones_like(raw_labels, dtype=np.float32)
        elif self.uncertain_policy == "uones":
            labels = np.where(raw_labels == mv, 1.0, raw_labels).astype(np.float32)
            mask = np.ones_like(raw_labels, dtype=np.float32)
        else:
            raise ValueError(f"Unknown uncertain_policy: {self.uncertain_policy}")

        return img, torch.tensor(labels), torch.tensor(mask)


def get_transforms(image_size: int, train: bool):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(7),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
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


def build_model(model_name: str = "efficientnet_b0", dropout: float = 0.3) -> nn.Module:
    try:
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif model_name == "efficientnet_b3":
            model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
    except Exception as e:
        warnings.warn(f"Could not load pretrained weights ({e}); falling back to random init.")
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
        elif model_name == "efficientnet_b3":
            model = models.efficientnet_b3(weights=None)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def masked_bce_loss(logits, labels, mask, pos_weight=None):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    element_loss = loss_fn(logits, labels)
    masked = element_loss * mask
    denom = mask.sum().clamp(min=1)
    return masked.sum() / denom


def run_epoch(model, loader, optimizer, device, training: bool, pos_weight=None):
    model.train() if training else model.eval()
    total_loss = 0.0
    all_logits, all_labels, all_masks = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels, masks in _progress(loader, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)
            pw = pos_weight
            if pw is not None and training:
                pw = pw.to(device)
            elif not training:
                pw = None
            loss = masked_bce_loss(logits, labels, masks, pos_weight=pw)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_masks.append(masks.detach().cpu())

    avg_loss = total_loss / max(len(loader.dataset), 1)

    if len(all_logits) == 0:
        return avg_loss, float("nan")

    probs = torch.cat(all_logits).sigmoid().numpy()
    labels = torch.cat(all_labels).numpy()
    masks = torch.cat(all_masks).numpy().astype(bool)

    aucs = []
    for c in range(NUM_CLASSES):
        mask_c = masks[:, c]
        if mask_c.sum() < 2:
            continue
        y_true = labels[mask_c, c]
        if len(np.unique(y_true)) < 2:
            continue
        auc = _binary_auc(y_true, probs[mask_c, c])
        if not np.isnan(auc):
            aucs.append(auc)

    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    return avg_loss, mean_auc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train_csv",
        type=Path,
        default=Path("/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests_preprocessed/train_manifest_preprocessed.csv"),
    )
    p.add_argument(
        "--val_csv",
        type=Path,
        default=Path("/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests_preprocessed/val_manifest_preprocessed.csv"),
    )
    p.add_argument("--model_name", default="efficientnet_b3", choices=["efficientnet_b0", "efficientnet_b3"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--missing_value", type=float, default=-999.0)
    p.add_argument(
        "--uncertain_policy",
        default="uzeros",
        choices=["mask", "uzeros", "uones"],
        help="mask: ignore sentinel labels. uzeros/uones: map sentinel to 0/1 and supervise all classes.",
    )
    p.add_argument(
        "--use_pos_weight",
        action="store_true",
        help="Per-class pos_weight=neg/pos on train (helps rare labels).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument(
        "--runs_base",
        type=Path,
        default=Path("/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/checkpoints/runs"),
    )
    p.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Subfolder under runs_base. Empty -> auto from model + policy + timestamp.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="If set, write checkpoints here (overrides runs_base/run_name).",
    )
    return p.parse_args()


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    args.runs_base.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        name = args.run_name
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{args.model_name}_{args.uncertain_policy}_{ts}"
    out = args.runs_base / name
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = _resolve_output_dir(args)
    args.output_dir = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.txt", "w", encoding="utf-8") as f:
        for k, v in sorted(vars(args).items()):
            f.write(f"{k}={v!r}\n")
    print(f"Output directory: {out_dir}")
    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_auc,val_loss,val_auc\n")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    for p in [args.train_csv, args.val_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input CSV: {p}")

    train_df = _validate_manifest(pd.read_csv(args.train_csv), args.missing_value, "train_csv")
    val_df = _validate_manifest(pd.read_csv(args.val_csv), args.missing_value, "val_csv")
    print(f"Usable rows -> train: {len(train_df)}, val: {len(val_df)}")

    train_ds = XrayDataset(
        train_df,
        transform=get_transforms(args.image_size, train=True),
        missing_value=args.missing_value,
        uncertain_policy=args.uncertain_policy,
    )
    val_ds = XrayDataset(
        val_df,
        transform=get_transforms(args.image_size, train=False),
        missing_value=args.missing_value,
        uncertain_policy=args.uncertain_policy,
    )

    train_pos_weight: torch.Tensor | None = None
    if args.use_pos_weight:
        train_pos_weight = compute_pos_weight_tensor(
            train_df, args.uncertain_policy, args.missing_value
        )
        print(f"pos_weight (neg/pos): {train_pos_weight.numpy().round(3)}")

    pin = device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = build_model(model_name=args.model_name, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = -1.0
    epochs_no_improve = 0
    best_path = out_dir / "best_model.pt"
    final_path = out_dir / "final_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_auc = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            training=True,
            pos_weight=train_pos_weight,
        )
        val_loss, val_auc = run_epoch(
            model, val_loader, optimizer, device, training=False, pos_weight=None
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"Train loss={train_loss:.4f} AUC={train_auc:.4f} | "
            f"Val loss={val_loss:.4f} AUC={val_auc:.4f} [{elapsed:.0f}s]"
        )

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{train_auc:.6f},{val_loss:.6f},{val_auc:.6f}\n"
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_auc": val_auc,
                "args": vars(args),
                "label_cols": LABEL_COLS,
            },
            final_path,
        )

        valid_auc = not np.isnan(val_auc)
        improved = valid_auc and (val_auc > best_auc)
        if improved:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_auc": val_auc,
                    "args": vars(args),
                    "label_cols": LABEL_COLS,
                },
                best_path,
            )
            print(f"  New best saved: {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if not best_path.exists() and final_path.exists():
        torch.save(torch.load(final_path, map_location="cpu"), best_path)
        print(f"[WARN] best_model.pt not selected by AUC; copied from final_model.pt")

    print(f"Done. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
