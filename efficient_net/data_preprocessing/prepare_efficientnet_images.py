"""
Create a lightweight preprocessed image cache for EfficientNet.

What this script does:
- Reads manifests produced by build_efficientnet_manifests.py
- Loads images from abs_path
- Converts to RGB (3 channels)
- Resizes to square input size (default 224)
- Saves cached images under team-writable output root (outside repo)
- Writes new manifests with preprocessed_path

What it does NOT do:
- No random augmentation (that should happen during training)
- No per-pixel normalization persisted to files
  (normalization should be applied in the training transforms)
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest_root",
        type=Path,
        default=Path("/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests"),
        help="Directory containing train/val/test manifests",
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data"),
        help="Team-writable root for cached images and updated manifests",
    )
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--image_format", choices=["jpg", "png"], default="jpg")
    p.add_argument("--jpg_quality", type=int, default=95)
    return p.parse_args()


def _cache_one_split(
    df: pd.DataFrame,
    split_name: str,
    out_img_dir: Path,
    image_size: int,
    image_format: str,
    jpg_quality: int,
) -> pd.DataFrame:
    from PIL import Image

    try:
        from tqdm import tqdm as _tqdm
    except Exception:
        _tqdm = None

    if "abs_path" not in df.columns:
        raise ValueError(f"{split_name}: expected column 'abs_path' in manifest")

    out_split_dir = out_img_dir / split_name
    out_split_dir.mkdir(parents=True, exist_ok=True)

    new_paths: list[str | None] = []

    row_iter = df.iterrows()
    if _tqdm is not None:
        row_iter = _tqdm(row_iter, total=len(df), desc=f"processing {split_name}")

    for _, row in row_iter:
        src = Path(str(row["abs_path"]))

        # Stable filename from original relative path if available.
        rel_token = str(row.get("Path", src.name)).replace("/", "_")
        stem = Path(rel_token).stem
        out_name = f"{stem}.{image_format}"
        dst = out_split_dir / out_name

        if dst.exists():
            new_paths.append(str(dst))
            continue

        try:
            with Image.open(src) as img:
                img = img.convert("RGB").resize((image_size, image_size), Image.LANCZOS)
                if image_format == "jpg":
                    img.save(dst, quality=jpg_quality)
                else:
                    img.save(dst)
            new_paths.append(str(dst))
        except Exception as e:
            print(f"[{split_name}] failed: {src} -> {e}")
            new_paths.append(None)

    out_df = df.copy()
    out_df["preprocessed_path"] = new_paths
    out_df = out_df.dropna(subset=["preprocessed_path"]).reset_index(drop=True)
    return out_df


def main() -> None:
    args = parse_args()

    train_manifest = args.manifest_root / "train_manifest.csv"
    val_manifest = args.manifest_root / "val_manifest.csv"
    test_manifest = args.manifest_root / "test_manifest.csv"

    for p in [train_manifest, val_manifest, test_manifest]:
        if not p.exists():
            raise FileNotFoundError(f"Missing manifest: {p}")

    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest)
    test_df = pd.read_csv(test_manifest)

    out_img_dir = args.output_root / "preprocessed_images"
    out_manifest_dir = args.output_root / "manifests_preprocessed"
    out_manifest_dir.mkdir(parents=True, exist_ok=True)

    train_out = _cache_one_split(
        train_df,
        split_name="train",
        out_img_dir=out_img_dir,
        image_size=args.image_size,
        image_format=args.image_format,
        jpg_quality=args.jpg_quality,
    )
    val_out = _cache_one_split(
        val_df,
        split_name="val",
        out_img_dir=out_img_dir,
        image_size=args.image_size,
        image_format=args.image_format,
        jpg_quality=args.jpg_quality,
    )
    test_out = _cache_one_split(
        test_df,
        split_name="test",
        out_img_dir=out_img_dir,
        image_size=args.image_size,
        image_format=args.image_format,
        jpg_quality=args.jpg_quality,
    )

    train_out_path = out_manifest_dir / "train_manifest_preprocessed.csv"
    val_out_path = out_manifest_dir / "val_manifest_preprocessed.csv"
    test_out_path = out_manifest_dir / "test_manifest_preprocessed.csv"

    train_out.to_csv(train_out_path, index=False)
    val_out.to_csv(val_out_path, index=False)
    test_out.to_csv(test_out_path, index=False)

    print("Done.")
    print(f"train: {len(train_out)} -> {train_out_path}")
    print(f"val:   {len(val_out)} -> {val_out_path}")
    print(f"test:  {len(test_out)} -> {test_out_path}")
    print(f"images cached under: {out_img_dir}")


if __name__ == "__main__":
    main()
