"""
Run DenseNet multilabel inference from a manifest CSV and a saved checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dense_net.common import LABEL_COLS, get_device  # noqa: E402
from dense_net.data import XrayManifestDataset, get_image_transforms, load_manifest  # noqa: E402
from dense_net.model import build_densenet_model  # noqa: E402

DEFAULT_PREPROCESSED_MANIFEST_ROOT = Path(
    "/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests_preprocessed"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_PREPROCESSED_MANIFEST_ROOT / "test_manifest_preprocessed.csv",
    )
    parser.add_argument("--output", type=Path, default=Path("densenet_submission.csv"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_name", choices=["densenet121", "densenet169"], default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = args.model_name or checkpoint.get("model_name", "densenet121")
    image_size = args.image_size or checkpoint.get("image_size", 224)
    dropout = args.dropout
    if dropout is None:
        dropout = checkpoint.get("dropout", 0.3)

    model = build_densenet_model(
        model_name=model_name,
        dropout=dropout,
        freeze_backbone=False,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    df = load_manifest(args.csv)
    dataset = XrayManifestDataset(
        df,
        transform=get_image_transforms(image_size=image_size, train=False),
        require_labels=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    all_ids = []
    all_probs = []
    with torch.no_grad():
        for images, image_ids in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

            if isinstance(image_ids, torch.Tensor):
                all_ids.extend(image_ids.cpu().numpy().tolist())
            else:
                all_ids.extend(list(image_ids))

    probabilities = np.concatenate(all_probs, axis=0)
    submission = pd.DataFrame(probabilities, columns=LABEL_COLS)
    submission.insert(0, "Id", all_ids)
    submission.to_csv(args.output, index=False)

    print(f"Device: {device}")
    print(
        f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')} "
        f"(val_auc={checkpoint.get('val_auc', float('nan')):.4f})"
    )
    print(f"Saved {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
