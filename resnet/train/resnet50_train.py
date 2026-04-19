

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
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
NUM_CLASSES = len(LABEL_COLS)
MISSING     = -999.0

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_model(dropout=0.4):
    """ResNet50 with custom head. Backbone starts frozen."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, NUM_CLASSES),
    )
    _freeze_backbone(model)
    return model


def _freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc")


def _unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True


def make_optimizer(model, lr, phase):
    if phase == 1:
        return torch.optim.AdamW(
            model.fc.parameters(),
            lr=lr * 10,
            weight_decay=1e-4,
        )
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    return torch.optim.AdamW(
        [
            {"params": backbone_params,       "lr": lr,      "weight_decay": 1e-4},
            {"params": model.fc.parameters(), "lr": lr * 10, "weight_decay": 1e-4},
        ]
    )



def masked_bce_loss(logits, labels, mask, pos_weight=None):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    loss = loss_fn(logits, labels)
    loss = loss * mask
    return loss.sum() / mask.sum().clamp(min=1)

def _binary_auc(y_true, y_score):
    y_true  = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)

    sum_ranks_pos = ranks[y_true == 1].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _mean_auc_from_lists(all_logits, all_labels, all_masks):
    probs  = torch.cat(all_logits).sigmoid().numpy()
    labels = torch.cat(all_labels).numpy()
    masks  = torch.cat(all_masks).numpy().astype(bool)

    aucs = []
    for c in range(NUM_CLASSES):
        m = masks[:, c]
        if m.sum() < 2:
            continue
        y = labels[m, c]
        if len(np.unique(y)) < 2:
            continue
        aucs.append(_binary_auc(y, probs[m, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def compute_per_class_auc(all_logits, all_labels, all_masks):
    probs  = torch.cat(all_logits).sigmoid().numpy()
    labels = torch.cat(all_labels).numpy()
    masks  = torch.cat(all_masks).numpy().astype(bool)

    out = {}
    for i, name in enumerate(LABEL_COLS):
        m = masks[:, i]
        if m.sum() < 2:
            out[name] = np.nan
            continue
        y = labels[m, i]
        if len(np.unique(y)) < 2:
            out[name] = np.nan
            continue
        out[name] = _binary_auc(y, probs[m, i])
    return out


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    training,
    pos_weight=None,
    scaler=None,
    return_outputs=False,
):
    model.train() if training else model.eval()

    total_loss = 0.0
    all_logits, all_labels, all_masks = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels, masks in tqdm(loader, leave=False, mininterval=10):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            pw = pos_weight.to(device) if (pos_weight is not None and training) else None

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(imgs)
                loss   = masked_bce_loss(logits, labels, masks, pw)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_masks.append(masks.detach().cpu())

    avg_loss = total_loss / len(loader.dataset)
    mean_auc = _mean_auc_from_lists(all_logits, all_labels, all_masks)

    if return_outputs:
        return avg_loss, mean_auc, all_logits, all_labels, all_masks
    return avg_loss, mean_auc


class AlexManifestDataset(Dataset):
    def __init__(self, df, transform, path_col, missing_value):
        self.df            = df.reset_index(drop=True)
        self.transform     = transform
        self.path_col      = path_col
        self.missing_value = missing_value

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        img    = Image.open(row[self.path_col]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        raw    = row[LABEL_COLS].values.astype(np.float32)
        mask   = (raw != self.missing_value).astype(np.float32)
        labels = np.where(raw == self.missing_value, 0.0, raw)

        return img, torch.tensor(labels), torch.tensor(mask)


def get_transforms(size, augment):
    ops = [transforms.Resize((size, size))]

    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ]

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",      type=Path,  required=True)
    parser.add_argument("--val_csv",        type=Path,  required=True)
    parser.add_argument("--output_dir",     type=Path,  default=Path("out"))
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=128)
    parser.add_argument("--lr",             type=float, default=2e-4)
    parser.add_argument("--img_size",       type=int,   default=512)
    parser.add_argument("--unfreeze_epoch", type=int,   default=2,
                        help="Epoch at which backbone unfreezes (1-indexed, default 6)")
    parser.add_argument("--num_workers",    type=int,   default=6,
                        help="DataLoader workers (~75%% of --cpus-per-task)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    print(
        f"Device: {device}  |  img_size: {args.img_size}  |  batch: {args.batch_size}  |  "
        f"workers: {args.num_workers}  |  lr: {args.lr:.1e}  |  "
        f"unfreeze @ epoch {args.unfreeze_epoch}"
    )

    epoch_csv = args.output_dir / "epoch_auc.csv"
    with open(epoch_csv, "w") as f:
        f.write("epoch,phase,train_auc,val_auc,epoch_time_s\n")

    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)

    path_col = "preprocessed_path" if "preprocessed_path" in train_df.columns else "abs_path"

    train_ds      = AlexManifestDataset(train_df, get_transforms(args.img_size, True),  path_col, MISSING)
    val_ds        = AlexManifestDataset(val_df,   get_transforms(args.img_size, False), path_col, MISSING)
    train_eval_ds = AlexManifestDataset(train_df, get_transforms(args.img_size, False), path_col, MISSING)

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    train_loader      = DataLoader(train_ds,      batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader        = DataLoader(val_ds,        batch_size=args.batch_size, **loader_kwargs)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=args.batch_size, **loader_kwargs)

    model     = build_model().to(device)
    optimizer = make_optimizer(model, args.lr, phase=1)
    phase     = 1

    best_auc  = 0.0
    best_path = args.output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):

        if epoch == args.unfreeze_epoch and phase == 1:
            print(
                f"\nEpoch {epoch}: unfreezing backbone — "
                f"backbone lr={args.lr:.1e}, fc lr={args.lr*10:.1e}"
            )
            _unfreeze_backbone(model)
            optimizer = make_optimizer(model, args.lr, phase=2)
            phase = 2

        t0 = time.time()
        train_loss, train_auc = run_epoch(model, train_loader, optimizer, device, True,  scaler=scaler)
        val_loss,   val_auc   = run_epoch(model, val_loader,   optimizer, device, False)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d} [phase {phase}]: "
            f"train_auc={train_auc:.4f}  val_auc={val_auc:.4f}  ({elapsed:.0f}s)"
        )

        with open(epoch_csv, "a") as f:
            f.write(f"{epoch},{phase},{train_auc:.6f},{val_auc:.6f},{elapsed:.1f}\n")

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ new best val AUC: {best_auc:.4f}")

    print(f"\nLoading best checkpoint (val_auc={best_auc:.4f})...")
    model.load_state_dict(torch.load(best_path, map_location=device))

    _, _, tl, tlab, tm = run_epoch(model, train_eval_loader, optimizer, device, False, return_outputs=True)
    _, _, vl, vlab, vm = run_epoch(model, val_loader,        optimizer, device, False, return_outputs=True)

    train_auc_pc = compute_per_class_auc(tl, tlab, tm)
    val_auc_pc   = compute_per_class_auc(vl, vlab, vm)

    rows = [
        {"class": k, "train_auc": train_auc_pc[k], "val_auc": val_auc_pc[k]}
        for k in LABEL_COLS
    ]
    pd.DataFrame(rows).to_csv(args.output_dir / "per_class_auc.csv", index=False)

    print("\nPer-class AUC (best checkpoint):")
    print(f"  {'Class':<35} {'Train':>8} {'Val':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8}")
    for r in rows:
        t = f"{r['train_auc']:.4f}" if not np.isnan(r["train_auc"]) else "     nan"
        v = f"{r['val_auc']:.4f}"   if not np.isnan(r["val_auc"])   else "     nan"
        print(f"  {r['class']:<35} {t:>8} {v:>8}")

    print(f"\nOutputs saved to {args.output_dir}/")
    print("  epoch_auc.csv     — per-epoch train/val AUC + phase + wall time")
    print("  per_class_auc.csv — per-class AUC on best checkpoint")
    print("  best.pt           — best model weights by val AUC")
    print("Done.")


if __name__ == "__main__":
    main()