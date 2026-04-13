"""
fine-tune ResNet50 for use with data_preprocessing1.py
"""

import argparse
import os
import time
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm

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

MISSING = -999
NUM_CLASSES = len(LABEL_COLS)

class XrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
 
        img = Image.open(row["preprocessed_path"]).convert("L")
        img = img.convert("RGB") 
 
        if self.transform:
            img = self.transform(img)
 
        raw_labels = row[LABEL_COLS].values.astype(np.float32)
        mask = torch.tensor((raw_labels != MISSING).astype(np.float32))
        labels = torch.tensor(np.where(raw_labels == MISSING, 0.0, raw_labels))
 
        return img, labels, mask


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_transforms(augment: bool):
    base = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if augment:
        aug = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return transforms.Compose(aug + base)
    return transforms.Compose(base)

def build_model(backbone: str = "resnet50", freeze_backbone: bool = False) -> nn.Module:
    model = models.resnet50(weights="IMAGENET1K_V1")

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
 
    return model

def masked_bce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    element_loss = loss_fn(logits, labels)
    masked = element_loss * mask 
    denom = mask.sum().clamp(min=1)
    return masked.sum() / denom

def run_epoch(model, loader, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss = 0.0
    all_logits, all_labels, all_masks = [], [], []
 
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels, masks in tqdm(loader, leave=False):
            imgs   = imgs.to(device)
            labels = labels.to(device)
            masks  = masks.to(device)
 
            logits = model(imgs)
            loss   = masked_bce_loss(logits, labels, masks)
 
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
 
            total_loss += loss.item() * imgs.size(0)
            all_logits.append(logits.cpu().detach())
            all_labels.append(labels.cpu())
            all_masks.append(masks.cpu())
 
    n = len(loader.dataset)
    avg_loss = total_loss / n
 
    all_logits = torch.cat(all_logits).sigmoid().numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_masks  = torch.cat(all_masks).numpy().astype(bool)
 
    aucs = []
    for c in range(NUM_CLASSES):
        mask_c = all_masks[:, c]
        if mask_c.sum() < 2 or len(np.unique(all_labels[mask_c, c])) < 2:
            continue
        auc = roc_auc_score(all_labels[mask_c, c], all_logits[mask_c, c])
        aucs.append(auc)
        print(f"  {LABEL_COLS[c]}: {auc:.4f}")
 
    mean_auc = np.mean(aucs) if aucs else float("nan")
    return avg_loss, mean_auc

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",            default="preprocessed_labels.csv")
    p.add_argument("--epochs",         type=int,   default=15)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--weight_decay",   type=float, default=1e-5)
    p.add_argument("--val_split",      type=float, default=0.15)
    p.add_argument("--warmup_epochs",  type=int,   default=2,
                   help="Epochs to train only the new head (backbone frozen)")
    p.add_argument("--num_workers",    type=int,   default=4)
    p.add_argument("--output_dir",     default="checkpoints")
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()
 
 
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
 
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
 
    # ── Data ──
    df = pd.read_csv(args.csv)
    n_val = int(len(df) * args.val_split)
    n_train = len(df) - n_val
    train_df, val_df = random_split(
        range(len(df)), [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    train_df = df.iloc[list(train_df)].reset_index(drop=True)
    val_df   = df.iloc[list(val_df)].reset_index(drop=True)
 
    train_ds = XrayDataset(train_df, transform=get_transforms(augment=True))
    val_ds   = XrayDataset(val_df,   transform=get_transforms(augment=False))
 
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
 
    print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")
 
    model = build_model(freeze_backbone=True).to(device)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=args.lr * 10,
                                  weight_decay=args.weight_decay)
 
    best_auc = 0.0
    best_path = os.path.join(args.output_dir, "best_model.pt")
 
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
 
        if epoch == args.warmup_epochs + 1:
            print("\n── Unfreezing backbone ──")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                          weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.warmup_epochs
            )
 
        train_loss, train_auc = run_epoch(model, train_loader, optimizer, device, training=True)
        val_loss,   val_auc   = run_epoch(model, val_loader,   optimizer, device, training=False)
 
        if epoch > args.warmup_epochs and 'scheduler' in dir():
            scheduler.step()
 
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"Train loss={train_loss:.4f} AUC={train_auc:.4f}  |  "
            f"Val loss={val_loss:.4f} AUC={val_auc:.4f}  "
            f"[{elapsed:.0f}s]"
        )
 
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_auc": val_auc, "args": vars(args)}, best_path)
            print(f"  ✓ New best saved (AUC={best_auc:.4f})")
 
    print(f"\nTraining done. Best val AUC: {best_auc:.4f}  →  {best_path}")
 
 
if __name__ == "__main__":
    main()
