"""
inference.py — run inference with trained ResNet50 and output submission CSV
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

import sys
sys.path.append('/resnick/groups/CS156b/from_central/2026/haa/cs156b-2026-haa/resnet/train')
from finetune_resnet50 import build_model, LABEL_COLS

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["preprocessed_path"]).convert("L").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row["Id"]


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   required=True,
                   help="Path to best_model.pt")
    p.add_argument("--csv",          required=True,
                   help="Path to preprocessed_test_labels.csv")
    p.add_argument("--output",       default="submission.csv")
    p.add_argument("--batch_size",   type=int, default=32)
    p.add_argument("--num_workers",  type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val AUC={checkpoint['val_auc']:.4f})")

    # Dataset
    df = pd.read_csv(args.csv)
    ds = TestDataset(df, transform=get_transform())
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers,
                        pin_memory=True)

    all_ids, all_preds = [], []

    with torch.no_grad():
        for imgs, ids in tqdm(loader):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_ids.extend(ids.numpy() if isinstance(ids, torch.Tensor) else ids)

    all_preds = np.concatenate(all_preds, axis=0)

    submission = pd.DataFrame(all_preds, columns=LABEL_COLS)
    submission.insert(0, "Id", all_ids)
    submission.to_csv(args.output, index=False)
    print(f"Saved {len(submission)} predictions to {args.output}")


if __name__ == "__main__":
    main()