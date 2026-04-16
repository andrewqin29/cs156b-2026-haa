from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from dense_net.common import IMAGENET_MEAN, IMAGENET_STD, LABEL_COLS, MISSING_VALUE


def get_image_transforms(image_size: int = 224, train: bool = False):
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


class XrayManifestDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        missing_value: float = MISSING_VALUE,
        require_labels: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.missing_value = missing_value
        self.require_labels = require_labels

        required_cols = ["abs_path"]
        if require_labels:
            required_cols.extend(LABEL_COLS)

        missing_cols = [column for column in required_cols if column not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Dataset manifest missing required columns: {missing_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, row: pd.Series) -> Path:
        img_path = row.get("preprocessed_path", row.get("abs_path"))
        if pd.isna(img_path):
            raise ValueError("Row has no usable image path in 'preprocessed_path' or 'abs_path'.")
        return Path(str(img_path))

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(row)

        image = Image.open(img_path).convert("L").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        if not self.require_labels:
            image_id = row.get("Id", idx)
            return image, image_id

        raw_labels = row[LABEL_COLS].to_numpy(dtype=np.float32)
        mask = torch.tensor((raw_labels != self.missing_value).astype(np.float32))
        labels = torch.tensor(
            np.where(raw_labels == self.missing_value, 0.0, raw_labels).astype(np.float32)
        )
        return image, labels, mask


def load_manifest(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    present_label_cols = [column for column in LABEL_COLS if column in df.columns]
    if present_label_cols:
        df[present_label_cols] = df[present_label_cols].replace(-1.0, MISSING_VALUE)
        df[present_label_cols] = df[present_label_cols].fillna(MISSING_VALUE)
    return df
