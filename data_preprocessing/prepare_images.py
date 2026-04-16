"""
sped up using process pool
"""
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--manifest_root",
        type=Path,
        default=Path("/resnick/groups/CS156b/from_central/2026/haa/preprocessed/manifests"),
        help="Directory containing train/val/test manifests",
    )
    p.add_argument(
        "--output_root",
        type=Path,
        default=Path("/resnick/groups/CS156b/from_central/2026/haa/preprocessed"),
        help="Team-writable root for cached images and updated manifests",
    )
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--image_format", choices=["jpg", "png"], default="jpg")
    p.add_argument("--jpg_quality", type=int, default=95)
    # Added workers arg so you can override manually if needed
    p.add_argument("--num_workers", type=int, default=None)
    return p.parse_args()


def _process_single_image(
    row_tuple: tuple,
    out_split_dir: Path,
    image_size: int,
    image_format: str,
    jpg_quality: int,
) -> str | None:
    from PIL import Image
    
    _, row = row_tuple
    src = Path(str(row["abs_path"]))

    rel_token = str(row.get("Path", src.name)).replace("/", "_")
    stem = Path(rel_token).stem
    out_name = f"{stem}.{image_format}"
    dst = out_split_dir / out_name

    if dst.exists():
        return str(dst)

    try:
        with Image.open(src) as img:
            # Resize using LANCZOS (high quality)
            img = img.convert("RGB").resize((image_size, image_size), Image.LANCZOS)
            if image_format == "jpg":
                img.save(dst, quality=jpg_quality)
            else:
                img.save(dst)
        return str(dst)
    except Exception:
        # Silently return None on failure; main loop handles logging/dropping
        return None


def _cache_one_split(
    df: pd.DataFrame,
    split_name: str,
    out_img_dir: Path,
    image_size: int,
    image_format: str,
    jpg_quality: int,
    num_workers: int | None = None,
) -> pd.DataFrame:
    if "abs_path" not in df.columns:
        raise ValueError(f"{split_name}: expected column 'abs_path' in manifest")

    out_split_dir = out_img_dir / split_name
    out_split_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    print(f"[{split_name}] processing {len(df)} images with {num_workers} workers...")

    worker_fn = partial(
        _process_single_image,
        out_split_dir=out_split_dir,
        image_size=image_size,
        image_format=image_format,
        jpg_quality=jpg_quality,
    )

    tasks = list(df.iterrows())
    new_paths: list[str | None] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        try:
            from tqdm import tqdm
            new_paths = list(tqdm(executor.map(worker_fn, tasks), total=len(df), desc=split_name))
        except ImportError:
            new_paths = list(executor.map(worker_fn, tasks))

    out_df = df.copy()
    out_df["preprocessed_path"] = new_paths
    
    # Report failures if any
    failed_count = out_df["preprocessed_path"].isna().sum()
    if failed_count > 0:
        print(f"[{split_name}] Warning: {failed_count} images failed to process.")

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

    out_img_dir = args.output_root / "preprocessed_images"
    out_manifest_dir = args.output_root / "manifests_preprocessed"
    out_manifest_dir.mkdir(parents=True, exist_ok=True)

    processed_dfs = {}
    splits = [
        ("train", train_manifest),
        ("val", val_manifest),
        ("test", test_manifest),
    ]

    for name, path in splits:
        df = pd.read_csv(path)
        processed_dfs[name] = _cache_one_split(
            df,
            split_name=name,
            out_img_dir=out_img_dir,
            image_size=args.image_size,
            image_format=args.image_format,
            jpg_quality=args.jpg_quality,
            num_workers=args.num_workers
        )

    for name in ["train", "val", "test"]:
        out_path = out_manifest_dir / f"{name}_manifest_preprocessed.csv"
        processed_dfs[name].to_csv(out_path, index=False)
        print(f"{name}: {len(processed_dfs[name])} -> {out_path}")

    print(f"\nDone. Images cached under: {out_img_dir}")


if __name__ == "__main__":
    main()