"""
Train DenseNet on standardized front_512 manifests.

Expected inputs live outside the repo under:
  /resnick/groups/CS156b/from_central/2026/haa/front_512_data

Each run writes to:
  /resnick/groups/CS156b/from_central/2026/haa/results/dense_net/runs/<run_name>/
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

NUM_CLASSES = len(LABEL_COLS)
FRONT_512_ROOT = Path("/resnick/groups/CS156b/from_central/2026/haa/front_512_data")
RESULTS_DENSE_NET = Path("/resnick/groups/CS156b/from_central/2026/haa/results/dense_net")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pick_path_column(df: pd.DataFrame) -> str:
    if "preprocessed_path" in df.columns:
        return "preprocessed_path"
    if "abs_path" in df.columns:
        return "abs_path"
    raise ValueError("Manifest must contain either 'preprocessed_path' or 'abs_path'.")


def resolve_existing_image_path(row: pd.Series) -> str | None:
    for column in ("preprocessed_path", "abs_path"):
        if column in row.index:
            candidate = row[column]
            if pd.isna(candidate):
                continue
            candidate = str(candidate)
            if Path(candidate).exists():
                return candidate
    return None


def _validate_manifest(
    df: pd.DataFrame,
    missing_value: float,
    name: str,
) -> pd.DataFrame:
    missing_cols = [column for column in LABEL_COLS if column not in df.columns]
    if missing_cols:
        raise ValueError(f"{name}: missing label columns: {missing_cols}")

    _pick_path_column(df)

    for column in LABEL_COLS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[LABEL_COLS] = df[LABEL_COLS].where(~df[LABEL_COLS].isna(), other=missing_value)

    df = df.copy()
    df["_image_path"] = df.apply(resolve_existing_image_path, axis=1)

    before = len(df)
    df = df[df["_image_path"].notna()].copy().reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"[WARN] {name}: dropped {dropped} rows with missing/nonexistent image files")

    if len(df) == 0:
        raise ValueError(f"{name}: no usable rows after validation")

    return df


def compute_pos_weight_tensor(
    train_df: pd.DataFrame,
    uncertain_policy: str,
    missing_value: float,
) -> torch.Tensor:
    arr = train_df[LABEL_COLS].to_numpy(dtype=np.float64)
    missing = arr == missing_value
    uncertain = arr == -1.0

    if uncertain_policy == "mask":
        labels = np.where(missing | uncertain, 0.0, arr)
        mask = ~(missing | uncertain)
    elif uncertain_policy == "uzeros":
        labels = np.where(missing, 0.0, arr)
        labels = np.where(uncertain, 0.0, labels)
        mask = ~missing
    elif uncertain_policy == "uones":
        labels = np.where(missing, 0.0, arr)
        labels = np.where(uncertain, 1.0, labels)
        mask = ~missing
    else:
        raise ValueError(f"Unknown uncertain_policy: {uncertain_policy}")

    weights: list[float] = []
    for idx in range(NUM_CLASSES):
        mask_c = mask[:, idx]
        if mask_c.sum() < 1:
            weights.append(1.0)
            continue
        positives = float(((labels[:, idx] == 1.0) & mask_c).sum())
        negatives = float(((labels[:, idx] == 0.0) & mask_c).sum())
        weights.append(negatives / max(positives, 1.0))
    return torch.tensor(weights, dtype=torch.float32)


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


class XrayDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform=None,
        missing_value: float = -999.0,
        uncertain_policy: str = "mask",
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.missing_value = missing_value
        self.uncertain_policy = uncertain_policy

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(str(row["_image_path"])).convert("RGB")
        if self.transform:
            img = self.transform(img)

        raw_labels = row[LABEL_COLS].to_numpy(dtype=np.float32)
        missing = raw_labels == self.missing_value
        uncertain = raw_labels == -1.0

        if self.uncertain_policy == "mask":
            labels = np.where(missing | uncertain, 0.0, raw_labels).astype(np.float32)
            mask = (~(missing | uncertain)).astype(np.float32)
        elif self.uncertain_policy == "uzeros":
            labels = np.where(missing, 0.0, raw_labels).astype(np.float32)
            labels = np.where(uncertain, 0.0, labels).astype(np.float32)
            mask = (~missing).astype(np.float32)
        elif self.uncertain_policy == "uones":
            labels = np.where(missing, 0.0, raw_labels).astype(np.float32)
            labels = np.where(uncertain, 1.0, labels).astype(np.float32)
            mask = (~missing).astype(np.float32)
        else:
            raise ValueError(f"Unknown uncertain_policy: {self.uncertain_policy}")

        return img, torch.tensor(labels), torch.tensor(mask)


def get_transforms(image_size: int, train: bool):
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(7),
                transforms.ColorJitter(brightness=0.05, contrast=0.05),
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


def configure_torch_home(torch_home: Path | None) -> None:
    if torch_home is None:
        return
    torch_home.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(torch_home))


def set_feature_extractor_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.features.parameters():
        param.requires_grad = trainable


def get_head_parameters(model: nn.Module):
    return model.classifier.parameters()


def build_model(
    model_name: str = "densenet121",
    dropout: float = 0.3,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    torch_home: Path | None = None,
) -> nn.Module:
    configure_torch_home(torch_home)
    try:
        if model_name == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.densenet121(weights=weights)
        elif model_name == "densenet169":
            weights = models.DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
            model = models.densenet169(weights=weights)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
    except Exception as exc:
        warnings.warn(f"Could not load pretrained weights ({exc}); falling back to random init.")
        if model_name == "densenet121":
            model = models.densenet121(weights=None)
        elif model_name == "densenet169":
            model = models.densenet169(weights=None)
        else:
            raise

    if freeze_backbone:
        set_feature_extractor_trainable(model, trainable=False)

    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


def masked_bce_loss(logits, labels, mask, pos_weight=None):
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    element_loss = loss_fn(logits, labels)
    masked = element_loss * mask
    denom = mask.sum().clamp(min=1)
    return masked.sum() / denom


def run_epoch(model, loader, optimizer, device, training: bool, pos_weight=None):
    model.train() if training else model.eval()
    total_loss = 0.0
    all_logits, all_labels, all_masks = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, labels, masks in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(imgs)
            pw = pos_weight.to(device) if (pos_weight is not None and training) else None
            loss = masked_bce_loss(logits, labels, masks, pos_weight=pw)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_masks.append(masks.detach().cpu())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    probs = torch.cat(all_logits).sigmoid().numpy()
    labels = torch.cat(all_labels).numpy()
    masks = torch.cat(all_masks).numpy().astype(bool)

    per_label_auc: dict[str, float] = {}
    for idx, label in enumerate(LABEL_COLS):
        mask_c = masks[:, idx]
        if mask_c.sum() < 2:
            continue
        y_true = labels[mask_c, idx]
        if len(np.unique(y_true)) < 2:
            continue
        auc = _binary_auc(y_true, probs[mask_c, idx])
        if not np.isnan(auc):
            per_label_auc[label] = float(auc)

    mean_auc = float(np.mean(list(per_label_auc.values()))) if per_label_auc else float("nan")
    return avg_loss, mean_auc, per_label_auc


def parse_args() -> argparse.Namespace:
    root = FRONT_512_ROOT / "manifests_preprocessed"
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=Path, default=root / "train_manifest_preprocessed.csv")
    parser.add_argument("--val_csv", type=Path, default=root / "val_manifest_preprocessed.csv")
    parser.add_argument("--model_name", choices=["densenet121", "densenet169"], default="densenet121")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head_lr_multiplier", type=float, default=10.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--missing_value", type=float, default=-999.0)
    parser.add_argument("--uncertain_policy", choices=["mask", "uzeros", "uones"], default="mask")
    parser.add_argument("--use_pos_weight", action="store_true")
    parser.add_argument("--pos_weight_clip", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--torch_home", type=Path, default=Path("/resnick/groups/CS156b/from_central/2026/haa/asqin/.torch_test_cache"))
    parser.add_argument("--runs_base", type=Path, default=RESULTS_DENSE_NET / "runs")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    args.runs_base.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        return args.runs_base / args.run_name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.runs_base / f"{args.model_name}_{args.uncertain_policy}_{args.image_size}px_{ts}"


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    out_dir = resolve_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in (args.train_csv, args.val_csv):
        if not path.exists():
            raise FileNotFoundError(f"Missing input CSV: {path}")

    train_df = _validate_manifest(pd.read_csv(args.train_csv), args.missing_value, "train_csv")
    val_df = _validate_manifest(pd.read_csv(args.val_csv), args.missing_value, "val_csv")

    with open(out_dir / "run_config.txt", "w", encoding="utf-8") as handle:
        for key, value in sorted(vars(args).items()):
            handle.write(f"{key}={value!r}\n")

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        handle.write("epoch,train_loss,train_auc,val_loss,val_auc\n")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Output directory: {out_dir}")
    print(f"Device: {device}")
    print(f"Usable rows -> train: {len(train_df):,}, val: {len(val_df):,}")

    train_ds = XrayDataset(
        train_df,
        transform=get_transforms(args.image_size, train=True),
        missing_value=args.missing_value,
        uncertain_policy=args.uncertain_policy,
    )
    val_ds = XrayDataset(
        val_df,
        transform=get_transforms(args.image_size, train=False),
        missing_value=args.missing_value,
        uncertain_policy=args.uncertain_policy,
    )

    pos_weight = None
    if args.use_pos_weight:
        pos_weight = compute_pos_weight_tensor(train_df, args.uncertain_policy, args.missing_value)
        if args.pos_weight_clip > 0:
            pos_weight = torch.clamp(pos_weight, max=float(args.pos_weight_clip))
        print(f"pos_weight (neg/pos): {pos_weight.numpy().round(3)}")

    pin = device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = build_model(
        model_name=args.model_name,
        dropout=args.dropout,
        pretrained=True,
        freeze_backbone=args.warmup_epochs > 0,
        torch_home=args.torch_home,
    ).to(device)

    if args.warmup_epochs > 0:
        optimizer = torch.optim.AdamW(
            get_head_parameters(model),
            lr=args.lr * args.head_lr_multiplier,
            weight_decay=args.weight_decay,
        )
        scheduler = None
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = -1.0
    epochs_no_improve = 0
    best_path = out_dir / "best_model.pt"
    final_path = out_dir / "final_model.pt"
    best_metrics_path = out_dir / "best_metrics.json"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if args.warmup_epochs > 0 and epoch == args.warmup_epochs + 1:
            print("\nUnfreezing DenseNet backbone")
            set_feature_extractor_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(args.epochs - args.warmup_epochs, 1),
            )

        train_loss, train_auc, train_per_label_auc = run_epoch(
            model, train_loader, optimizer, device, training=True, pos_weight=pos_weight
        )
        val_loss, val_auc, val_per_label_auc = run_epoch(
            model, val_loader, optimizer, device, training=False, pos_weight=None
        )

        if scheduler is not None and (args.warmup_epochs == 0 or epoch > args.warmup_epochs):
            scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} [{elapsed:.0f}s]"
        )

        with open(metrics_path, "a", encoding="utf-8") as handle:
            handle.write(
                f"{epoch},{train_loss:.6f},{train_auc:.6f},{val_loss:.6f},{val_auc:.6f}\n"
            )

        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_auc": val_auc,
                "per_label_auc": val_per_label_auc,
                "model_name": args.model_name,
                "image_size": args.image_size,
                "dropout": args.dropout,
                "args": vars(args),
            },
            final_path,
        )

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_auc": val_auc,
                    "per_label_auc": val_per_label_auc,
                    "model_name": args.model_name,
                    "image_size": args.image_size,
                    "dropout": args.dropout,
                    "args": vars(args),
                },
                best_path,
            )
            save_json(
                {
                    "best_epoch": epoch,
                    "best_val_auc": val_auc,
                    "val_per_label_auc": val_per_label_auc,
                    "train_per_label_auc": train_per_label_auc,
                },
                best_metrics_path,
            )
            print(f"  Saved new best checkpoint: {best_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"Training complete. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
