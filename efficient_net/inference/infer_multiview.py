"""
infer_multiview.py — Inference for the EfficientNet multi-view model.

Usage:
    python infer_multiview.py \
        --frontal_checkpoint /path/to/best_frontal.pt \
        --lateral_checkpoint /path/to/best_lateral.pt \
        --csv /path/to/test_manifest.csv \
        --output submission.csv
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
from torchvision.transforms import v2
from tqdm import tqdm


# ── Import shared constants and model from training script ────────────────────
_TRAIN_DIR = Path(__file__).resolve().parents[1] / "train"
sys.path.append(str(_TRAIN_DIR))

from train_multiview import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABEL_COLS,
    NUM_CLASSES,
    VIEW_FRONTAL,
    VIEW_LATERAL,
    VIEW_UNKNOWN,
    build_model,
    get_view_type,
    get_study_id,
)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, path_col: str):
        df = df.copy()
        df["_view_type"] = df["Path"].apply(get_view_type)
        df["_study_id"]  = df.apply(get_study_id, axis=1)
        if "Id" not in df.columns:
            df["Id"] = df["_study_id"]
        self.df        = df.reset_index(drop=True)
        self.transform = transform

        # Pre-extract for fast indexing
        self.paths      = self.df[path_col].tolist()
        self.study_ids  = self.df["_study_id"].tolist()
        self.view_types = self.df["_view_type"].tolist()
        self.row_ids    = self.df["Id"].tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = str(self.paths[idx])
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing image: {path}")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.study_ids[idx], self.view_types[idx], self.row_ids[idx]


def _get_path_col(df: pd.DataFrame) -> str:
    for col in ("preprocessed_path", "abs_path"):
        if col in df.columns:
            return col
    raise ValueError("Test manifest must contain 'preprocessed_path' or 'abs_path'.")


def get_transform(image_size: int) -> v2.Compose:
    """Must match get_transforms(augment=False) from training script."""
    return v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((image_size, image_size), antialias=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────────────────────────────────────

def load_checkpoint(path: Path, device: str) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        meta  = {k: v for k, v in ckpt.items() if k != "model_state"}
    else:
        state = ckpt
        meta  = {}

    dropout  = meta.get("dropout",  0.4)
    img_size = meta.get("img_size", 512)
    epoch    = meta.get("epoch",    "?")
    view      = meta.get("view",      "?") 
    study_mse = meta.get("study_mse", float("nan"))
    val_auc   = meta.get("val_auc",   float("nan"))

    model_name = meta.get("model_name", "efficientnet_b0")

    model = build_model(dropout, model_name).to(device)
    model.load_state_dict(state)
    model.eval()

    print(
        f"  Loaded {path.name}  view={view}  epoch={epoch}  "
        f"study_mse={study_mse:.4f}  val_auc={val_auc:.4f}  "
        f"model={model_name}  "
        f"dropout={dropout}  img_size={img_size}"
    )
    return model, {"img_size": img_size, "dropout": dropout, "view": view, "model_name": model_name}


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:       torch.nn.Module,
    df:          pd.DataFrame,
    view_filter: str,
    image_size:  int,
    batch_size:  int,
    num_workers: int,
    device:      str,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:

    path_col = _get_path_col(df)
    ds_full  = TestDataset(df, get_transform(image_size), path_col)

    # Filter to the requested view
    mask = [vt == view_filter for vt in ds_full.view_types]
    ds_full.paths      = [x for x, m in zip(ds_full.paths,      mask) if m]
    ds_full.study_ids  = [x for x, m in zip(ds_full.study_ids,  mask) if m]
    ds_full.view_types = [x for x, m in zip(ds_full.view_types, mask) if m]
    ds_full.row_ids    = [x for x, m in zip(ds_full.row_ids,    mask) if m]

    if len(ds_full) == 0:
        print(f"  No {view_filter} images found — skipping.")
        return np.empty((0, NUM_CLASSES)), [], [], []

    pin = (device == "cuda")
    loader = DataLoader(
        ds_full,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    all_scores, all_sids, all_vtypes, all_rids = [], [], [], []

    for imgs, study_ids, view_types, row_ids in tqdm(loader, desc=view_filter, leave=False):
        imgs = imgs.to(device, non_blocking=pin)
        with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
            preds = model(imgs).float()
        all_scores.append(preds.cpu().numpy())
        all_sids.extend(list(study_ids))
        all_vtypes.extend(list(view_types))
        all_rids.extend(list(row_ids))

    return np.concatenate(all_scores, axis=0), all_sids, all_vtypes, all_rids


# ──────────────────────────────────────────────────────────────────────────────
# Study-level aggregation — confidence-weighted (matches new training script)
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_study_predictions(
    study_ids: list[str],
    scores:    np.ndarray,
) -> dict[str, np.ndarray]:
    studies: dict[str, list] = defaultdict(list)
    for sid, s in zip(study_ids, scores):
        studies[sid].append(s)

    study_scores: dict[str, np.ndarray] = {}
    for sid, preds in studies.items():
        stacked = np.array(preds)
        if stacked.shape[0] == 1:
            study_scores[sid] = stacked[0]
        else:
            confidence  = np.abs(stacked)
            weight      = confidence / (confidence.sum(axis=0, keepdims=True) + 1e-8)
            study_scores[sid] = (stacked * weight).sum(axis=0)

    return study_scores


# ──────────────────────────────────────────────────────────────────────────────
# Build submission
# ──────────────────────────────────────────────────────────────────────────────

def build_submission(
    df:         pd.DataFrame,
    all_sids:   list[str],
    all_scores: np.ndarray,
) -> pd.DataFrame:
    study_scores = aggregate_study_predictions(all_sids, all_scores)

    df = df.copy()
    df["_study_id"] = df.apply(get_study_id, axis=1)

    rows    = []
    missing = 0
    for _, row in df.iterrows():
        sid   = row["_study_id"]
        score = study_scores.get(sid, None)
        if score is None:
            score = np.zeros(NUM_CLASSES)
            missing += 1
        rows.append(score)

    if missing > 0:
        print(f"  WARNING: {missing} rows had no predictions — filled with zeros.")

    submission = pd.DataFrame(rows, columns=LABEL_COLS)
    submission.insert(0, "Id", df["Id"].values)
    return submission


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-view EfficientNet inference -> study-level submission CSV"
    )
    p.add_argument("--frontal_checkpoint", type=Path, default=None)
    p.add_argument("--lateral_checkpoint", type=Path, default=None)
    p.add_argument("--csv",          required=True,
                   help="Test manifest CSV")
    p.add_argument("--output",       default="submission.csv")
    p.add_argument("--batch_size",   type=int, default=128)
    p.add_argument("--num_workers",  type=int, default=8)
    p.add_argument("--image_size",   type=int, default=None,
                   help="Override image size (default: read from checkpoint)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.frontal_checkpoint is None and args.lateral_checkpoint is None:
        raise ValueError("At least one of --frontal_checkpoint or --lateral_checkpoint is required.")

    device = (
        "cuda" if torch.cuda.is_available()          else
        "mps"  if torch.backends.mps.is_available()  else
        "cpu"
    )
    print(f"Device: {device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "Id" not in df.columns:
        raise ValueError("Test manifest must contain an 'Id' column.")
    print(f"Test manifest: {len(df)} rows  ({df['Id'].nunique()} unique Ids)")

    all_scores: list[np.ndarray] = []
    all_sids:   list[str]        = []

    if args.frontal_checkpoint is not None:
        print(f"\nLoading frontal checkpoint: {args.frontal_checkpoint}")
        model, meta = load_checkpoint(args.frontal_checkpoint, device)
        img_size = args.image_size or meta["img_size"] or 512

        print(f"Running frontal inference (img_size={img_size}) ...")
        scores, sids, _, _ = run_inference(
            model, df, VIEW_FRONTAL, img_size,
            args.batch_size, args.num_workers, device,
        )
        print(f"  {len(sids)} frontal images processed")
        all_scores.append(scores)
        all_sids.extend(sids)
        del model

    if args.lateral_checkpoint is not None:
        print(f"\nLoading lateral checkpoint: {args.lateral_checkpoint}")
        model, meta = load_checkpoint(args.lateral_checkpoint, device)
        img_size = args.image_size or meta["img_size"] or 512

        print(f"Running lateral inference (img_size={img_size}) ...")
        scores, sids, _, _ = run_inference(
            model, df, VIEW_LATERAL, img_size,
            args.batch_size, args.num_workers, device,
        )
        print(f"  {len(sids)} lateral images processed")
        all_scores.append(scores)
        all_sids.extend(sids)
        del model

    combined_scores = np.concatenate(all_scores, axis=0)

    print("\nAggregating to study level (confidence-weighted) ...")
    submission = build_submission(df, all_sids, combined_scores)

    print(f"  Submission rows: {len(submission)}  (expected: {len(df)})")
    assert len(submission) == len(df), "Row count mismatch — check study ID extraction."

    submission.to_csv(out_path, index=False)
    print(f"\nSaved {len(submission)} predictions to {out_path}")

    print("\nPrediction range per class:")
    for col in LABEL_COLS:
        mn, mx = submission[col].min(), submission[col].max()
        print(f"  {col:<35} [{mn:+.3f}, {mx:+.3f}]")


if __name__ == "__main__":
    main()
