"""
Build manifest CSVs for a DenseNet-based chest X-ray pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dense_net.common import LABEL_COLS, MISSING_VALUE, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--train_img_root", type=Path, required=True)
    parser.add_argument("--test_ids_csv", type=Path, required=True)
    parser.add_argument("--test_img_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing_value", type=float, default=MISSING_VALUE)
    parser.add_argument(
        "--frontal_only",
        action="store_true",
        default=True,
        help="Keep only frontal views in train/val manifests (default: on).",
    )
    parser.add_argument(
        "--no_frontal_only",
        dest="frontal_only",
        action="store_false",
        help="Disable frontal-only filtering.",
    )
    return parser.parse_args()


def _extract_patient_id(path_series: pd.Series) -> pd.Series:
    patient_id = path_series.astype(str).str.extract(r"(pid\d+)", expand=False)
    return patient_id.fillna(path_series.astype(str))


def _prepare_train_df(
    train_csv: Path,
    train_img_root: Path,
    frontal_only: bool,
    missing_value: float,
) -> pd.DataFrame:
    df = pd.read_csv(train_csv)

    required_cols = {"Path", *LABEL_COLS}
    missing_cols = sorted(required_cols.difference(df.columns))
    if missing_cols:
        raise ValueError(f"Train CSV missing required columns: {missing_cols}")

    if frontal_only:
        df = df[df["Path"].astype(str).str.contains("frontal", case=False, na=False)].copy()

    df["abs_path"] = df["Path"].astype(str).str.replace(
        r"^train/",
        str(train_img_root).rstrip("/") + "/",
        regex=True,
    )

    df[LABEL_COLS] = df[LABEL_COLS].astype(float)
    df[LABEL_COLS] = df[LABEL_COLS].replace(-1.0, missing_value).fillna(missing_value)
    df["patient_id"] = _extract_patient_id(df["Path"])
    return df


def _patient_split(
    df: pd.DataFrame,
    val_split: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    unique_patients = df["patient_id"].dropna().astype(str).unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_patients)

    n_val_patients = max(1, int(round(len(unique_patients) * val_split)))
    val_patients = set(unique_patients[:n_val_patients])

    is_val = df["patient_id"].astype(str).isin(val_patients)
    train_df = df[~is_val].reset_index(drop=True)
    val_df = df[is_val].reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise ValueError("Patient-level split produced an empty train or val set.")

    return train_df, val_df


def _prepare_test_df(test_ids_csv: Path, test_img_root: Path) -> pd.DataFrame:
    df = pd.read_csv(test_ids_csv)
    if "Path" not in df.columns:
        raise ValueError("Test IDs CSV must include a 'Path' column.")

    if "Id" not in df.columns:
        df = df.copy()
        df["Id"] = np.arange(len(df))

    df["abs_path"] = df["Path"].astype(str).str.replace(
        r"^test/",
        str(test_img_root).rstrip("/") + "/",
        regex=True,
    )
    return df


def main() -> None:
    args = parse_args()

    for path in [args.train_csv, args.train_img_root, args.test_ids_csv, args.test_img_root]:
        if not path.exists():
            raise FileNotFoundError(f"Required path does not exist: {path}")

    manifest_dir = args.output_root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_all = _prepare_train_df(
        train_csv=args.train_csv,
        train_img_root=args.train_img_root,
        frontal_only=args.frontal_only,
        missing_value=args.missing_value,
    )
    train_df, val_df = _patient_split(train_all, val_split=args.val_split, seed=args.seed)
    test_df = _prepare_test_df(args.test_ids_csv, args.test_img_root)

    keep_cols = ["Path", "abs_path", "patient_id", *LABEL_COLS]
    train_manifest_path = manifest_dir / "train_manifest.csv"
    val_manifest_path = manifest_dir / "val_manifest.csv"
    test_manifest_path = manifest_dir / "test_manifest.csv"

    train_df[keep_cols].to_csv(train_manifest_path, index=False)
    val_df[keep_cols].to_csv(val_manifest_path, index=False)
    test_df[["Id", "Path", "abs_path"]].to_csv(test_manifest_path, index=False)

    summary = {
        "frontal_only": bool(args.frontal_only),
        "missing_value": float(args.missing_value),
        "seed": int(args.seed),
        "val_split": float(args.val_split),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_unique_patients": int(train_df["patient_id"].nunique()),
        "val_unique_patients": int(val_df["patient_id"].nunique()),
        "manifests": {
            "train": str(train_manifest_path),
            "val": str(val_manifest_path),
            "test": str(test_manifest_path),
        },
    }
    save_json(summary, manifest_dir / "manifest_summary.json")

    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
