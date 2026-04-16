"""
Train EfficientNet on front_512_data manifests.

Same training loop as `train_efficientnet.py`, with defaults pointing at
`/resnick/groups/CS156b/from_central/2026/haa/front_512_data` and `--image_size 512`.
Checkpoints and logs go under `results/efficient_net/runs/` (see `train_efficientnet.py`).

Run from this directory (or ensure `train_efficientnet` is importable). Extra CLI
arguments are appended after the defaults; argparse uses the last value for duplicates.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    root = Path("/resnick/groups/CS156b/from_central/2026/haa/front_512_data")
    prefix = [
        sys.argv[0],
        "--train_csv",
        str(root / "manifests_preprocessed/train_manifest_preprocessed.csv"),
        "--val_csv",
        str(root / "manifests_preprocessed/val_manifest_preprocessed.csv"),
        "--image_size",
        "512",
    ]
    sys.argv = prefix + sys.argv[1:]

    from train_efficientnet import main as train_main

    train_main()


if __name__ == "__main__":
    main()
