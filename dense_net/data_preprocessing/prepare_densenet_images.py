"""
Prepare a cached image set for DenseNet training/inference from manifest CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_format", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpg_quality", type=int, default=95)
    return parser.parse_args()


def _cache_one_split(
    df: pd.DataFrame,
    split_name: str,
    image_dir: Path,
    image_size: int,
    image_format: str,
    jpg_quality: int,
) -> pd.DataFrame:
    if "abs_path" not in df.columns:
        raise ValueError(f"{split_name}: expected an 'abs_path' column in manifest.")

    out_split_dir = image_dir / split_name
    out_split_dir.mkdir(parents=True, exist_ok=True)

    cached_paths: list[str | None] = []
    for _, row in df.iterrows():
        src = Path(str(row["abs_path"]))
        rel_token = str(row.get("Path", src.name)).replace("/", "_")
        stem = Path(rel_token).stem
        dst = out_split_dir / f"{stem}.{image_format}"

        if dst.exists():
            cached_paths.append(str(dst))
            continue

        try:
            with Image.open(src) as image:
                image = image.convert("L").convert("RGB")
                image = image.resize((image_size, image_size), Image.LANCZOS)
                if image_format == "jpg":
                    image.save(dst, quality=jpg_quality)
                else:
                    image.save(dst)
            cached_paths.append(str(dst))
        except Exception as exc:
            print(f"[{split_name}] failed: {src} -> {exc}")
            cached_paths.append(None)

    out_df = df.copy()
    out_df["preprocessed_path"] = cached_paths
    out_df = out_df.dropna(subset=["preprocessed_path"]).reset_index(drop=True)
    return out_df


def main() -> None:
    args = parse_args()

    train_manifest = args.manifest_root / "train_manifest.csv"
    val_manifest = args.manifest_root / "val_manifest.csv"
    test_manifest = args.manifest_root / "test_manifest.csv"

    for path in [train_manifest, val_manifest, test_manifest]:
        if not path.exists():
            raise FileNotFoundError(f"Missing manifest: {path}")

    train_df = pd.read_csv(train_manifest)
    val_df = pd.read_csv(val_manifest)
    test_df = pd.read_csv(test_manifest)

    image_dir = args.output_root / "preprocessed_images"
    manifest_dir = args.output_root / "manifests_preprocessed"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    train_out = _cache_one_split(
        train_df,
        split_name="train",
        image_dir=image_dir,
        image_size=args.image_size,
        image_format=args.image_format,
        jpg_quality=args.jpg_quality,
    )
    val_out = _cache_one_split(
        val_df,
        split_name="val",
        image_dir=image_dir,
        image_size=args.image_size,
        image_format=args.image_format,
        jpg_quality=args.jpg_quality,
    )
    test_out = _cache_one_split(
        test_df,
        split_name="test",
        image_dir=image_dir,
        image_size=args.image_size,
        image_format=args.image_format,
        jpg_quality=args.jpg_quality,
    )

    train_out_path = manifest_dir / "train_manifest_preprocessed.csv"
    val_out_path = manifest_dir / "val_manifest_preprocessed.csv"
    test_out_path = manifest_dir / "test_manifest_preprocessed.csv"

    train_out.to_csv(train_out_path, index=False)
    val_out.to_csv(val_out_path, index=False)
    test_out.to_csv(test_out_path, index=False)

    print("Done.")
    print(f"train: {len(train_out)} -> {train_out_path}")
    print(f"val:   {len(val_out)} -> {val_out_path}")
    print(f"test:  {len(test_out)} -> {test_out_path}")
    print(f"cached images: {image_dir}")


if __name__ == "__main__":
    main()
