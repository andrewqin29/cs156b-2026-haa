
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
from torchvision import transforms
from tqdm import tqdm


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


class TestDataset(Dataset):

    def __init__(self, df: pd.DataFrame, transform, path_col: str):
        df = df.copy()
        df["_view_type"] = df["Path"].apply(get_view_type)
        df["_study_id"]  = df.apply(get_study_id, axis=1)
        if "Id" not in df.columns:
            df["Id"] = df["_study_id"]
        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.path_col  = path_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row  = self.df.iloc[idx]
        path = str(row[self.path_col])
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing image: {path}")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row["_study_id"], row["_view_type"], str(row["Id"])


def _get_path_col(df: pd.DataFrame) -> str:
    for col in ("preprocessed_path", "abs_path"):
        if col in df.columns:
            return col
    raise ValueError("Test manifest must contain 'preprocessed_path' or 'abs_path'.")


def get_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_checkpoint(path: Path, device: str) -> tuple[torch.nn.Module, dict]:
    """Load a training checkpoint and return (model, metadata_dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        meta  = {k: v for k, v in ckpt.items() if k != "model_state"}
    else:
        state = ckpt
        meta  = {}

    dropout  = meta.get("dropout",  0.4)
    img_size = meta.get("img_size", None)
    epoch    = meta.get("epoch",    "?")
    val_auc  = meta.get("val_auc",  float("nan"))
    view     = meta.get("view",     "?")

    model = build_model(dropout).to(device)
    model.load_state_dict(state)
    model.eval()

    print(
        f"  Loaded {path.name}  view={view}  epoch={epoch}  "
        f"val_auc={val_auc:.4f}  dropout={dropout}"
    )
    return model, {"img_size": img_size, "dropout": dropout, "view": view}


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    df: pd.DataFrame,
    view_filter: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> tuple[np.ndarray, list[str], list[str], list[str]]:

    path_col = _get_path_col(df)
    ds = TestDataset(df, get_transform(image_size), path_col)
    ds.df = ds.df[ds.df["_view_type"] == view_filter].reset_index(drop=True)

    if len(ds) == 0:
        return np.empty((0, NUM_CLASSES)), [], [], []

    pin = device == "cuda"
    loader = DataLoader(
        ds,
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
            preds = model(imgs)
        all_scores.append(preds.cpu().numpy())
        all_sids.extend(list(study_ids))
        all_vtypes.extend(list(view_types))
        all_rids.extend(list(row_ids))

    return np.concatenate(all_scores, axis=0), all_sids, all_vtypes, all_rids


def build_per_image_submission(
    df: pd.DataFrame,
    all_sids: list[str],
    all_vtypes: list[str],
    all_scores: np.ndarray,
    frontal_weight: float,
) -> pd.DataFrame:
    study_preds: dict[str, dict] = defaultdict(
        lambda: {VIEW_FRONTAL: [], VIEW_LATERAL: []}
    )
    for sid, vt, s in zip(all_sids, all_vtypes, all_scores):
        bucket = vt if vt in (VIEW_FRONTAL, VIEW_LATERAL) else VIEW_FRONTAL
        study_preds[sid][bucket].append(s)

    study_scores: dict[str, np.ndarray] = {}
    for sid, views in study_preds.items():
        f = np.mean(views[VIEW_FRONTAL], axis=0) if views[VIEW_FRONTAL] else None
        l = np.mean(views[VIEW_LATERAL], axis=0) if views[VIEW_LATERAL] else None
        if f is not None and l is not None:
            study_scores[sid] = frontal_weight * f + (1 - frontal_weight) * l
        elif f is not None:
            study_scores[sid] = f
        else:
            study_scores[sid] = l

    df = df.copy()
    df["_study_id"] = df.apply(get_study_id, axis=1)

    rows = []
    for _, row in df.iterrows():
        sid = row["_study_id"]
        score = study_scores.get(sid, np.zeros(NUM_CLASSES))
        rows.append(score)

    submission = pd.DataFrame(rows, columns=LABEL_COLS)
    submission.insert(0, "Id", df["Id"].values)
    return submission


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-view inference: frontal + lateral → per-image submission."
    )
    p.add_argument("--frontal_checkpoint", type=Path, default=None)
    p.add_argument("--lateral_checkpoint", type=Path, default=None)
    p.add_argument("--csv",              required=True,
                   help="Test manifest CSV (same format as train/val CSVs)")
    p.add_argument("--output",           default="submission.csv")
    p.add_argument("--batch_size",       type=int,   default=128)
    p.add_argument("--num_workers",      type=int,   default=8)
    p.add_argument("--image_size",       type=int,   default=None,
                   help="Override image size (default: read from checkpoint metadata)")
    p.add_argument("--frontal_weight",   type=float, default=0.65)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.frontal_checkpoint is None and args.lateral_checkpoint is None:
        raise ValueError("At least one of --frontal_checkpoint or --lateral_checkpoint is required.")

    device = (
        "cuda"  if torch.cuda.is_available()         else
        "mps"   if torch.backends.mps.is_available() else
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
    all_vtypes: list[str]        = []

    if args.frontal_checkpoint is not None:
        print(f"\nLoading frontal checkpoint: {args.frontal_checkpoint}")
        frontal_model, frontal_meta = load_checkpoint(args.frontal_checkpoint, device)
        img_size = args.image_size or frontal_meta["img_size"] or 512

        print(f"Running frontal inference (img_size={img_size}) ...")
        scores, sids, vtypes, _ = run_inference(
            frontal_model, df, VIEW_FRONTAL, img_size,
            args.batch_size, args.num_workers, device,
        )
        print(f"  {len(sids)} frontal images")
        all_scores.append(scores)
        all_sids.extend(sids)
        all_vtypes.extend(vtypes)
        del frontal_model

    if args.lateral_checkpoint is not None:
        print(f"\nLoading lateral checkpoint: {args.lateral_checkpoint}")
        lateral_model, lateral_meta = load_checkpoint(args.lateral_checkpoint, device)
        img_size = args.image_size or lateral_meta["img_size"] or 512

        print(f"Running lateral inference (img_size={img_size}) ...")
        scores, sids, vtypes, _ = run_inference(
            lateral_model, df, VIEW_LATERAL, img_size,
            args.batch_size, args.num_workers, device,
        )
        print(f"  {len(sids)} lateral images")
        all_scores.append(scores)
        all_sids.extend(sids)
        all_vtypes.extend(vtypes)
        del lateral_model

    combined_scores = np.concatenate(all_scores, axis=0)

    print(f"\nBuilding per-image submission (frontal_weight={args.frontal_weight:.2f}) ...")
    submission = build_per_image_submission(
        df, all_sids, all_vtypes, combined_scores, args.frontal_weight
    )

    print(f"  Submission rows : {len(submission)}  (expected: {len(df)})")
    assert len(submission) == len(df), "Row count mismatch — check study ID extraction."

    submission.to_csv(out_path, index=False)
    print(f"\nSaved {len(submission)} per-image predictions to {out_path}")


if __name__ == "__main__":
    main()