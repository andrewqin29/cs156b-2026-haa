"""
Build manifests for training/inference.

Design goals:
- Keep raw course data read-only (do not copy train/test image trees)
- Write lightweight CSV manifests to team-writable area
- Produce train/val split with patient-level grouping to reduce leakage
- Normalize labels into a training-friendly format with missing sentinel

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train_csv",
        type=Path,
        default=Path("/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"),
    )
    p.add_argument(
        "--train_img_root",
        type=Path,
        default=Path("/resnick/groups/CS156b/from_central/data/train"),
    )
    p.add_argument(
        "--test_ids_csv",
        type=Path,
        default=Path("/resnick/groups/CS156b/from_central/data/student_labels/test_ids.csv"),
    )
    p.add_argument(
        "--test_img_root",
        type=Path,
        default=Path("/resnick/groups/CS156b/from_central/data/test"),
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("/resnick/groups/CS156b/from_central/2026/haa/preprocessed"),
    )
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--missing_value", type=float, default=-999.0)
    p.add_argument(
        "--frontal_only",
        action="store_true",
        default=True,
        help="Keep only frontal views (default: on)",
    )
    p.add_argument(
        "--no_frontal_only",
        dest="frontal_only",
        action="store_false",
        help="Disable frontal-only filtering",
    )
    return p.parse_args()


def _extract_patient_id(path_series: pd.Series) -> pd.Series:
    # CheXpert path format includes pidXXXXX; fall back to path if missing pattern.
    pid = path_series.astype(str).str.extract(r"(pid\d+)", expand=False)
    return pid.fillna(path_series.astype(str))


def _prepare_train_df(
    train_csv: Path,
    train_img_root: Path,
    frontal_only: bool,
    missing_value: float,
) -> pd.DataFrame:
    df = pd.read_csv(train_csv)

    required = {"Path", *LABEL_COLS}
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"train CSV missing required columns: {missing_cols}")

    if frontal_only:
        df = df[df["Path"].astype(str).str.contains("frontal", case=False, na=False)].copy()

    df["abs_path"] = df["Path"].astype(str).str.replace(
        r"^train/", str(train_img_root) + "/", regex=True
    )

    # Convert labels to float and replace uncertain/missing with sentinel.
    df[LABEL_COLS] = df[LABEL_COLS].astype(float)
    df[LABEL_COLS] = df[LABEL_COLS].replace(-1.0, missing_value).fillna(missing_value)

    df["patient_id"] = _extract_patient_id(df["Path"])
    return df


def _patient_split(df: pd.DataFrame, val_split: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    unique_patients = df["patient_id"].dropna().astype(str).unique()
    rng = np.random.default_rng(seed)
    unique_patients - np.array(unique_patients)
    rng.shuffle(unique_patients)

    n_val_patients = max(1, int(round(len(unique_patients) * val_split)))
    val_patients = set(unique_patients[:n_val_patients])

    is_val = df["patient_id"].astype(str).isin(val_patients)
    train_df = df[~is_val].reset_index(drop=True)
    val_df = df[is_val].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Patient-level split produced an empty train or val set.")

    return train_df, val_df


def _prepare_test_df(test_ids_csv: Path, test_img_root: Path) -> pd.DataFrame:
    test_df = pd.read_csv(test_ids_csv)

    if "Path" not in test_df.columns:
        raise ValueError("test IDs CSV must include a 'Path' column")

    if "Id" not in test_df.columns:
        # fallback in case IDs are named differently or absent
        test_df = test_df.copy()
        test_df["Id"] = np.arange(len(test_df))

    test_df["abs_path"] = test_df["Path"].astype(str).str.replace(
        r"^test/", str(test_img_root) + "/", regex=True
    )
    return test_df


def main() -> None:
    args = parse_args()

    for p in [args.train_csv, args.train_img_root, args.test_ids_csv, args.test_img_root]:
        if not p.exists():
            raise FileNotFoundError(f"Required path does not exist: {p}")

    out_manifest_dir = args.output_root / "manifests"
    out_manifest_dir.mkdir(parents=True, exist_ok=True)

    train_all = _prepare_train_df(
        train_csv=args.train_csv,
        train_img_root=args.train_img_root,
        frontal_only=args.frontal_only,
        missing_value=args.missing_value,
    )

    train_df, val_df = _patient_split(train_all, val_split=args.val_split, seed=args.seed)

    keep_cols = ["Path", "abs_path", "patient_id", *LABEL_COLS]
    train_path = out_manifest_dir / "train_manifest.csv"
    val_path = out_manifest_dir / "val_manifest.csv"

    train_df[keep_cols].to_csv(train_path, index=False)
    val_df[keep_cols].to_csv(val_path, index=False)

    test_df = _prepare_test_df(args.test_ids_csv, args.test_img_root)
    test_path = out_manifest_dir / "test_manifest.csv"
    test_df[["Id", "Path", "abs_path"]].to_csv(test_path, index=False)

    summary = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_unique_patients": int(train_df["patient_id"].nunique()),
        "val_unique_patients": int(val_df["patient_id"].nunique()),
        "frontal_only": bool(args.frontal_only),
        "missing_value": float(args.missing_value),
        "output_root": str(args.output_root),
        "manifests": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
    }
    summary_path = out_manifest_dir / "manifest_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
