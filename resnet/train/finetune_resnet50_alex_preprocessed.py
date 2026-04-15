"""
Fine-tune ResNet50 on Alex's EfficientNet image cache + preprocessed manifests.

Expects CSVs under efficient_net_data/manifests_preprocessed/ with
`preprocessed_path` (or `abs_path`) pointing at RGB 224×224 cached images.
Train/val splits match the shared team manifests for comparable baselines.

finetune_resnet50_alex_preprocessed.sh
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from finetune_resnet50 import LABEL_COLS, MISSING, build_model, run_epoch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _pick_path_column(df: pd.DataFrame) -> str:
    if "preprocessed_path" in df.columns:
        return "preprocessed_path"
    if "abs_path" in df.columns:
        return "abs_path"
    raise ValueError("Manifest must contain either 'preprocessed_path' or 'abs_path'.")


def _validate_manifest(df: pd.DataFrame, missing_value: float, name: str) -> tuple[pd.DataFrame, str]:
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

    return df, path_col


class AlexManifestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        path_col: str = "preprocessed_path",
        missing_value: float = MISSING,
    ):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.path_col = path_col
        self.missing_value = missing_value

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.path_col]
        if pd.isna(img_path):
            raise ValueError(f"Row {idx} has no usable image path.")

        img = Image.open(str(img_path)).convert("RGB")
        if self.transform:
            img = self.transform(img)

        raw_labels = row[LABEL_COLS].values.astype(np.float32)
        mask = torch.tensor((raw_labels != self.missing_value).astype(np.float32))
        labels = torch.tensor(
            np.where(raw_labels == self.missing_value, 0.0, raw_labels).astype(np.float32)
        )
        return img, labels, mask


def get_transforms(image_size: int, augment: bool):
    """Align with Alex's EfficientNet training script (light aug + ImageNet norm)."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    size = (image_size, image_size)
    if augment:
        return transforms.Compose(
            [
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(7),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def parse_args() -> argparse.Namespace:
    root = Path("/resnick/groups/CS156b/from_central/2026/haa/front_512_data/manifests_preprocessed")
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=Path, default=root / "train_manifest_preprocessed.csv")
    p.add_argument("--val_csv", type=Path, default=root / "val_manifest_preprocessed.csv")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument(
        "--warmup_epochs",
        type=int,
        default=2,
        help="Epochs to train only the new head (backbone frozen)",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--output_dir", type=Path, default=Path("checkpoints_512"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--missing_value", type=float, default=float(MISSING))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.output_dir / "metrics.csv"
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
    pin = device == "cuda"

    pos_weight = torch.tensor(
        [
            0.4727,
            1.7229,
            0.4004,
            0.0560,
            6.8783,
            0.1410,
            0.7660,
            0.1487,
            0.0209,
        ],
        dtype=torch.float32,
        device=device,
    )

    for path in (args.train_csv, args.val_csv):
        if not path.exists():
            raise FileNotFoundError(f"Missing input CSV: {path}")

    train_df, train_path_col = _validate_manifest(
        pd.read_csv(args.train_csv), args.missing_value, "train_csv"
    )
    val_df, val_path_col = _validate_manifest(
        pd.read_csv(args.val_csv), args.missing_value, "val_csv"
    )
    if train_path_col != val_path_col:
        print(f"[WARN] train uses path column {train_path_col!r}, val uses {val_path_col!r}")

    print(f"Usable rows -> train: {len(train_df):,}, val: {len(val_df):,}")

    train_ds = AlexManifestDataset(
        train_df,
        transform=get_transforms(args.image_size, augment=True),
        path_col=train_path_col,
        missing_value=args.missing_value,
    )
    val_ds = AlexManifestDataset(
        val_df,
        transform=get_transforms(args.image_size, augment=False),
        path_col=val_path_col,
        missing_value=args.missing_value,
    )

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

    model = build_model(freeze_backbone=True).to(device)
    optimizer = torch.optim.AdamW(
        model.fc.parameters(), lr=args.lr * 10, weight_decay=args.weight_decay
    )
    scheduler = None

    best_auc = 0.0
    epochs_no_improve = 0
    best_path = args.output_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if epoch == args.warmup_epochs + 1:
            print("\n Unfreezing")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(args.epochs - args.warmup_epochs, 1)
            )

        train_loss, train_auc = run_epoch(
            model, train_loader, optimizer, device, training=True, pos_weight=pos_weight
        )
        val_loss, val_auc = run_epoch(
            model, val_loader, optimizer, device, training=False, pos_weight=pos_weight
        )

        if scheduler is not None and epoch > args.warmup_epochs:
            scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"Train loss={train_loss:.4f} AUC={train_auc:.4f}  |  "
            f"Val loss={val_loss:.4f} AUC={val_auc:.4f}  "
            f"[{elapsed:.0f}s]"
        )

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.4f},{train_auc:.4f},{val_loss:.4f},{val_auc:.4f}\n")

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_auc": val_auc,
                    "args": vars(args),
                    "script": "finetune_resnet50_alex_preprocessed.py",
                },
                best_path,
            )
            print(f"  New best saved (AUC={best_auc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nTraining done. Best val AUC: {best_auc:.4f}:  {best_path}")


if __name__ == "__main__":
    main()
