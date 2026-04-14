"""
Inference for EfficientNet multi-label CheXpert model.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

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


class TestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.get("preprocessed_path", row.get("abs_path"))
        if pd.isna(img_path):
            raise ValueError(f"Row {idx} has no usable image path.")

        img = Image.open(str(img_path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row["Id"]


def get_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_model(model_name: str = "efficientnet_b0"):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=None)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(LABEL_COLS))
    return model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("/resnick/groups/CS156b/from_central/2026/haa/askumar/efficient_net_data/manifests_preprocessed/test_manifest_preprocessed.csv"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/resnick/groups/CS156b/from_central/2026/haa/askumar/cs156b-2026-haa/efficient_net/inference/submission.csv"),
    )
    p.add_argument("--model_name", default="efficientnet_b0", choices=["efficientnet_b0", "efficientnet_b3"])
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--recompute_no_finding", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {args.checkpoint}")
    if not args.csv.exists():
        raise FileNotFoundError(f"Missing test CSV: {args.csv}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model(model_name=args.model_name).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    df = pd.read_csv(args.csv)
    if "Id" not in df.columns:
        df = df.copy()
        df["Id"] = np.arange(len(df))

    loader = DataLoader(
        TestDataset(df, transform=get_transform(args.image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    all_ids = []
    all_preds = []
    with torch.no_grad():
        for imgs, ids in loader:
            imgs = imgs.to(device, non_blocking=True)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            all_preds.append(probs)
            if isinstance(ids, torch.Tensor):
                all_ids.extend(ids.cpu().numpy().tolist())
            else:
                all_ids.extend(list(ids))

    pred = np.concatenate(all_preds, axis=0)

    if args.recompute_no_finding:
        no_finding_idx = LABEL_COLS.index("No Finding")
        other_idx = [i for i, c in enumerate(LABEL_COLS) if c != "No Finding"]
        pred[:, no_finding_idx] = 1.0 - pred[:, other_idx].max(axis=1)

    submission = pd.DataFrame(pred, columns=LABEL_COLS)
    submission.insert(0, "Id", all_ids)
    submission.to_csv(args.output, index=False)
    print(f"Saved {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()
