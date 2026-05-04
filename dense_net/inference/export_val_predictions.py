"""Export DenseNet validation predictions for ensembling and meta-learning."""

from __future__ import annotations

import argparse
import json
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
    SENTINEL,
    build_model,
    compute_auc,
    compute_scaled_mse,
    get_transforms,
    resolve_existing_image_path,
)


def get_study_id(row: pd.Series) -> str:
    if "Path" in row.index and pd.notna(row["Path"]):
        parts = Path(str(row["Path"])).parts
        if len(parts) >= 3:
            return f"{parts[-3]}_{parts[-2]}"
    if "patient_id" in row.index and pd.notna(row["patient_id"]):
        return str(row["patient_id"])
    return str(row.get("row_index", "unknown"))


def get_view_type(row: pd.Series) -> str:
    for column in ("Path", "preprocessed_path", "abs_path"):
        if column not in row.index or pd.isna(row[column]):
            continue
        filename = Path(str(row[column])).name.lower()
        if "frontal" in filename:
            return "frontal"
        if "lateral" in filename:
            return "lateral"
    return "unknown"


def _default_output() -> Path:
    return DENSE_NET_RESULTS / "val_predictions" / "densenet169_full512_scaled_mse_mask0_baseline_val_predictions.csv"


def _metrics_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_metrics.json")


def _json_float(value: float) -> float | None:
    value = float(value)
    return None if np.isnan(value) else value


def _prefixed_label(label: str) -> str:
    return f"label_{label}"


def _prefixed_pred(label: str) -> str:
    return f"pred_{label}"


class ValPredictionDataset(Dataset):
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
        return img, int(row["row_index"]), torch.tensor(raw_labels, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    default_csv = FULL_512_ROOT / "manifests_preprocessed" / "val_manifest_preprocessed.csv"
    parser = argparse.ArgumentParser(
        description="Export DenseNet validation predictions with labels and metadata."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=default_csv)
    parser.add_argument("--output", type=Path, default=_default_output())
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--model_name", choices=["densenet121", "densenet169"], default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--limit_rows", type=int, default=0, help="Optional smoke-test row limit.")
    return parser.parse_args()


def prepare_manifest(csv_path: Path, limit_rows: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    missing_cols = [column for column in LABEL_COLS if column not in df.columns]
    if missing_cols:
        raise ValueError(f"Validation manifest missing label columns: {missing_cols}")

    if limit_rows > 0:
        df = df.head(limit_rows).copy()

    df.insert(0, "row_index", np.arange(len(df), dtype=np.int64))
    for column in LABEL_COLS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[LABEL_COLS] = df[LABEL_COLS].where(~df[LABEL_COLS].isna(), other=SENTINEL)
    df["_image_path"] = df.apply(resolve_existing_image_path, axis=1)

    missing_images = df[df["_image_path"].isna()]
    if len(missing_images) > 0:
        examples = missing_images[["row_index", "Path"]].head(5).to_dict(orient="records")
        raise FileNotFoundError(f"{len(missing_images)} validation rows have no usable image path. Examples: {examples}")

    df["study_id"] = df.apply(get_study_id, axis=1)
    df["view_type"] = df.apply(get_view_type, axis=1)
    return df.reset_index(drop=True)


def load_checkpoint(path: Path, device: str) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def run_predictions(
    model: torch.nn.Module,
    df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    loader = DataLoader(
        ValPredictionDataset(df, transform=get_transforms(image_size, train=False)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(num_workers > 0),
    )

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_indices: list[int] = []
    model.eval()
    with torch.no_grad():
        for imgs, row_indices, raw_labels in loader:
            imgs = imgs.to(device, non_blocking=(device == "cuda"))
            if device == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    preds = model(imgs).float().cpu().numpy()
            else:
                preds = model(imgs).float().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(raw_labels.cpu().numpy())
            all_indices.extend(row_indices.cpu().numpy().astype(int).tolist())

    if not all_preds:
        raise ValueError("No validation rows were processed.")
    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0), all_indices


def build_output_df(
    df: pd.DataFrame,
    preds: np.ndarray,
    row_indices: list[int],
    run_name: str,
    checkpoint: Path,
    checkpoint_epoch: object,
) -> pd.DataFrame:
    pred_df = pd.DataFrame(preds, columns=[_prefixed_pred(label) for label in LABEL_COLS])
    pred_df.insert(0, "row_index", row_indices)

    metadata_cols = ["row_index", "Path", "abs_path", "preprocessed_path", "patient_id", "study_id", "view_type"]
    metadata_cols = [column for column in metadata_cols if column in df.columns]
    out = df[metadata_cols].merge(pred_df, on="row_index", how="left", validate="one_to_one")
    if out[[ _prefixed_pred(label) for label in LABEL_COLS ]].isna().any().any():
        raise RuntimeError("Missing predictions after joining by row_index.")

    out["run_name"] = run_name
    out["checkpoint"] = str(checkpoint)
    out["checkpoint_epoch"] = checkpoint_epoch

    for label in LABEL_COLS:
        out[_prefixed_label(label)] = df[label].to_numpy(dtype=np.float32)

    ordered_cols = metadata_cols + ["run_name", "checkpoint", "checkpoint_epoch"]
    ordered_cols += [_prefixed_label(label) for label in LABEL_COLS]
    ordered_cols += [_prefixed_pred(label) for label in LABEL_COLS]
    return out[ordered_cols]


def write_metrics(
    output_path: Path,
    checkpoint_path: Path,
    checkpoint: dict,
    run_name: str,
    rows: int,
    images_predicted: int,
    preds: np.ndarray,
    raw_labels: np.ndarray,
) -> None:
    if "label_variance" not in checkpoint:
        raise KeyError("Checkpoint is missing label_variance; cannot compute scaled MSE.")
    variance = torch.tensor(checkpoint["label_variance"], dtype=torch.float32)
    scaled_mse, per_label_scaled_mse = compute_scaled_mse(preds, raw_labels, variance)
    auc, per_label_auc = compute_auc(preds, raw_labels)
    known_counts = {
        label: int(((raw_labels[:, idx] == 1.0) | (raw_labels[:, idx] == -1.0)).sum())
        for idx, label in enumerate(LABEL_COLS)
    }

    metrics = {
        "checkpoint": str(checkpoint_path),
        "epoch": checkpoint.get("epoch", None),
        "run_name": run_name,
        "rows": int(rows),
        "images_predicted": int(images_predicted),
        "scaled_mse": _json_float(scaled_mse),
        "auc": _json_float(auc),
        "per_label_scaled_mse": {key: _json_float(value) for key, value in per_label_scaled_mse.items()},
        "per_label_auc": {key: _json_float(value) for key, value in per_label_auc.items()},
        "known_counts": known_counts,
        "label_policy": "score only known labels where value is -1 or 1; ignore 0 and -999",
    }
    with open(_metrics_path(output_path), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    if not args.csv.exists():
        raise FileNotFoundError(f"Missing validation CSV: {args.csv}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint = load_checkpoint(args.checkpoint, device)
    model_name = args.model_name or checkpoint.get("model_name", "densenet169")
    image_size = args.image_size or checkpoint.get("image_size", 512)
    dropout = args.dropout if args.dropout is not None else checkpoint.get("dropout", 0.3)
    run_name = args.run_name or checkpoint.get("args", {}).get("run_name", args.checkpoint.parent.name)

    model = build_model(
        model_name=model_name,
        dropout=dropout,
        pretrained=False,
        freeze_backbone=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    df = prepare_manifest(args.csv, args.limit_rows)
    preds, raw_labels, row_indices = run_predictions(
        model=model,
        df=df,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    output_df = build_output_df(
        df=df,
        preds=preds,
        row_indices=row_indices,
        run_name=run_name,
        checkpoint=args.checkpoint,
        checkpoint_epoch=checkpoint.get("epoch", "unknown"),
    )
    if len(output_df) != len(df):
        raise RuntimeError(f"Row count mismatch: output={len(output_df)} input={len(df)}")
    output_df.to_csv(args.output, index=False)
    write_metrics(
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
        run_name=run_name,
        rows=len(df),
        images_predicted=len(row_indices),
        preds=preds,
        raw_labels=raw_labels,
    )

    pred_cols = [_prefixed_pred(label) for label in LABEL_COLS]
    print(f"Device: {device}")
    print(f"Loaded checkpoint epoch {checkpoint.get('epoch', 'unknown')} from {args.checkpoint}")
    print(f"Rows exported: {len(output_df)}")
    print(f"Prediction range: [{output_df[pred_cols].min().min():+.3f}, {output_df[pred_cols].max().max():+.3f}]")
    print(f"Saved predictions to {args.output}")
    print(f"Saved metrics to {_metrics_path(args.output)}")


if __name__ == "__main__":
    main()
