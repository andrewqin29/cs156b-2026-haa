import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.append("/resnick/groups/CS156b/from_central/2026/haa/hgaston/cs156b-2026-haa/resnet/train")
from finetune_resnet50 import build_model, LABEL_COLS

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _test_path_column(df: pd.DataFrame) -> str:
    if "preprocessed_path" in df.columns:
        return "preprocessed_path"
    if "abs_path" in df.columns:
        return "abs_path"
    raise ValueError("Test manifest must contain 'preprocessed_path' or 'abs_path'.")


class TestDataset(Dataset):
    def __init__(self, df, transform=None, path_col: str = "preprocessed_path"):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.path_col = path_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = str(row[self.path_col])
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing image file: {path}")
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row["Id"]


def get_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--csv", required=True, help="e.g. test_manifest_preprocessed.csv")
    p.add_argument("--output", default="submission.csv")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--image_size", type=int, default=224)
    return p.parse_args()


def main():
    args = parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    pin = device == "cuda"

    out_path = Path(args.output)
    if out_path.parent and not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)
    model = build_model().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(
        f"Loaded checkpoint from epoch {checkpoint['epoch']} "
        f"(val AUC={checkpoint['val_auc']:.4f})"
    )

    df = pd.read_csv(args.csv)
    path_col = _test_path_column(df)
    ds = TestDataset(df, transform=get_transform(args.image_size), path_col=path_col)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    all_ids, all_preds = [], []

    with torch.no_grad():
        for imgs, ids in tqdm(loader):
            imgs = imgs.to(device, non_blocking=pin)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_ids.extend(ids if not isinstance(ids, torch.Tensor) else ids.numpy().tolist())

    all_preds = np.concatenate(all_preds, axis=0)
    # change no finding logic to incorporate other predictions
    # other_cols = [i for i, col in enumerate(LABEL_COLS) if col != "No Finding"]
    # no_finding_idx = LABEL_COLS.index("No Finding")
    # all_preds[:, no_finding_idx] = 1 - all_preds[:, other_cols].max(axis=1)
    # delete block if does not improve no finding

    submission = pd.DataFrame(all_preds, columns=LABEL_COLS)
    submission.insert(0, "Id", all_ids)
    submission.to_csv(out_path, index=False)
    print(f"Saved {len(submission)} predictions to {out_path}")


if __name__ == "__main__":
    main()