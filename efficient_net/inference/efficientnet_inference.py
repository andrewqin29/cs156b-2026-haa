"""Leaderboard inference for EfficientNet scaled-MSE chest X-ray model.

Produces a Kaggle-style CSV with one row per input manifest row:
    Id,No Finding,Enlarged Cardiomediastinum,...

By default, predictions are confidence-weighted within each study and then
written back onto every original manifest row for that study. Use
``--aggregate none`` to submit raw image-level predictions instead.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_TRAIN_DIR = Path(__file__).resolve().parents[1] / "train"
sys.path.append(str(_TRAIN_DIR))

from train_efficientnet import (  # noqa: E402
    EFFICIENT_NET_RESULTS,
    FULL_512_ROOT,
    LABEL_COLS,
    build_model,
    get_transforms,
)


def get_study_id(row: pd.Series) -> str:
    if "Path" in row.index and pd.notna(row["Path"]):
        parts = Path(str(row["Path"])).parts
        if len(parts) >= 3:
            return f"{parts[-3]}_{parts[-2]}"
    if "patient_id" in row.index and pd.notna(row["patient_id"]):
        return str(row["patient_id"])
    if "Id" in row.index and pd.notna(row["Id"]):
        return str(row["Id"])
    return "unknown"


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
        return img, int(row["_orig_index"]), row["_study_id"]


def aggregate_predictions(keys: list[str], scores: np.ndarray) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for key, score in zip(keys, scores):
        grouped[key].append(score)

    aggregated: dict[str, np.ndarray] = {}
    for key, preds in grouped.items():
        stacked = np.asarray(preds)
        if stacked.shape[0] == 1:
            aggregated[key] = stacked[0]
            continue
        confidence = np.abs(stacked)
        weights = confidence / (confidence.sum(axis=0, keepdims=True) + 1e-8)
        aggregated[key] = (stacked * weights).sum(axis=0)
    return aggregated


def build_submission(
    original_df: pd.DataFrame,
    pred_indices: list[int],
    pred_study_ids: list[str],
    pred_scores: np.ndarray,
    aggregate: str,
) -> pd.DataFrame:
    if aggregate == "none":
        row_scores = {idx: score for idx, score in zip(pred_indices, pred_scores)}
        key_col = None
        grouped_scores = None
    elif aggregate == "study":
        key_col = "_study_id"
        grouped_scores = aggregate_predictions(pred_study_ids, pred_scores)
        row_scores = None
    elif aggregate == "id":
        key_col = "Id"
        pred_ids = original_df.iloc[pred_indices]["Id"].astype(str).tolist()
        grouped_scores = aggregate_predictions(pred_ids, pred_scores)
        row_scores = None
    else:
        raise ValueError(f"Unsupported aggregate mode: {aggregate}")

    rows = []
    missing = 0
    for idx, row in original_df.iterrows():
        if aggregate == "none":
            score = row_scores.get(idx) if row_scores is not None else None
        else:
            key = str(row[key_col]) if key_col is not None else ""
            score = grouped_scores.get(key) if grouped_scores is not None else None

        if score is None:
            score = np.zeros(len(LABEL_COLS), dtype=np.float32)
            missing += 1
        rows.append(score)

    if missing > 0:
        print(f"[WARN] {missing} rows had no prediction; filled with zeros.")

    submission = pd.DataFrame(rows, columns=LABEL_COLS)
    submission.insert(0, "Id", original_df["Id"].values)
    return submission


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
        default=EFFICIENT_NET_RESULTS / "inference" / "submission.csv",
    )
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--model_name", choices=["efficientnet_b0", "efficientnet_b3"], default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--aggregate",
        choices=["study", "id", "none"],
        default="study",
        help="How to combine multiple image predictions before writing the row-level submission.",
    )
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

    model_name = args.model_name or checkpoint.get("model_name", "efficientnet_b0")
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
        raise ValueError("Input manifest must contain an 'Id' column for leaderboard submissions.")
    original_df = df.copy()
    original_df["_study_id"] = original_df.apply(get_study_id, axis=1)
    original_df["_orig_index"] = np.arange(len(original_df))

    df = original_df.copy()
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
        persistent_workers=(args.num_workers > 0),
    )

    all_indices: list[int] = []
    all_study_ids: list[str] = []
    all_preds: list[np.ndarray] = []
    with torch.no_grad():
        for imgs, orig_indices, study_ids in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                preds = model(imgs).float().cpu().numpy()
            all_preds.append(preds)
            all_indices.extend(orig_indices.cpu().numpy().astype(int).tolist())
            all_study_ids.extend(list(study_ids))

    preds = np.concatenate(all_preds, axis=0)
    submission = build_submission(original_df, all_indices, all_study_ids, preds, args.aggregate)
    if len(submission) != len(original_df):
        raise RuntimeError(f"Row count mismatch: got {len(submission)}, expected {len(original_df)}")
    submission.to_csv(args.output, index=False)

    print(f"Device: {device}")
    print(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')} "
        f"(val_scaled_mse={checkpoint.get('val_scaled_mse', float('nan')):.4f}, "
        f"val_auc={checkpoint.get('val_auc', float('nan')):.4f})"
    )
    print(f"Input rows: {len(original_df)}; images processed: {len(all_indices)}; aggregate={args.aggregate}")
    print(f"Saved {len(submission)} predictions to {args.output}")
    print("Prediction range per class:")
    for column in LABEL_COLS:
        print(f"  {column:<35} [{submission[column].min():+.3f}, {submission[column].max():+.3f}]")


if __name__ == "__main__":
    main()
