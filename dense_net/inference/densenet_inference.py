"""Inference for DenseNet scaled-MSE chest X-ray model on full-view data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_TRAIN_DIR = Path(__file__).resolve().parents[1] / "train"
sys.path.append(str(_TRAIN_DIR))

from train_densenet import (  # noqa: E402
    DENSE_NET_RESULTS,
    FULL_512_ROOT,
    LABEL_COLS,
    build_model,
    get_transforms,
)


def _resolve_image_path(row: pd.Series) -> str | None:
    for column in ("preprocessed_path", "abs_path"):
        if column in row.index:
            candidate = row[column]
            if pd.isna(candidate):
                continue
            candidate = str(candidate)
            if Path(candidate).exists():
                return candidate
    return None


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["_image_path"]
        if pd.isna(path):
            raise FileNotFoundError(f"Row {idx} has no usable image path.")
        img = Image.open(str(path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row["Id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--csv",
        type=Path,
        default=FULL_512_ROOT / "manifests_preprocessed" / "test_manifest_preprocessed.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DENSE_NET_RESULTS / "inference" / "submission.csv",
    )
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--model_name", choices=["densenet121", "densenet169"], default=None)
    parser.add_argument("--dropout", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    if not args.csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {args.csv}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)

    model_name = args.model_name or checkpoint.get("model_name", "densenet169")
    image_size = args.image_size or checkpoint.get("image_size", 512)
    dropout = args.dropout if args.dropout is not None else checkpoint.get("dropout", 0.3)

    model = build_model(
        model_name=model_name,
        dropout=dropout,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    df = pd.read_csv(args.csv).copy()
    if "Id" not in df.columns:
        df["Id"] = np.arange(len(df))
    df["_image_path"] = df.apply(_resolve_image_path, axis=1)
    before = len(df)
    df = df[df["_image_path"].notna()].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] Dropped {dropped} rows with missing/nonexistent image files")

    loader = DataLoader(
        TestDataset(df, transform=get_transforms(image_size, train=False)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    all_ids: list[object] = []
    all_preds: list[np.ndarray] = []
    with torch.no_grad():
        for imgs, ids in loader:
            imgs = imgs.to(device, non_blocking=True)
            preds = model(imgs).cpu().numpy()
            all_preds.append(preds)
            if isinstance(ids, torch.Tensor):
                all_ids.extend(ids.cpu().numpy().tolist())
            else:
                all_ids.extend(list(ids))

    preds = np.concatenate(all_preds, axis=0)
    submission = pd.DataFrame(preds, columns=LABEL_COLS)
    submission.insert(0, "Id", all_ids)
    submission.to_csv(args.output, index=False)

    print(f"Device: {device}")
    print(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')} "
        f"(val_scaled_mse={checkpoint.get('val_scaled_mse', float('nan')):.4f}, "
        f"val_auc={checkpoint.get('val_auc', float('nan')):.4f})"
    )
    print(f"Prediction range: [{submission[LABEL_COLS].min().min():+.3f}, {submission[LABEL_COLS].max().max():+.3f}]")
    print(f"Saved {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
