"""
Train a DenseNet transfer-learning model for 9-label chest X-ray classification.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from dense_net.common import (  # noqa: E402
    LABEL_COLS,
    MISSING_VALUE,
    compute_multilabel_auc,
    compute_pos_weight_from_df,
    ensure_dir,
    get_device,
    masked_bce_loss,
    save_json,
    seed_everything,
)
from dense_net.data import XrayManifestDataset, get_image_transforms, load_manifest  # noqa: E402
from dense_net.model import (  # noqa: E402
    build_densenet_model,
    get_head_parameters,
    set_feature_extractor_trainable,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--val_csv", type=Path, required=True)
    parser.add_argument("--model_name", choices=["densenet121", "densenet169"], default="densenet121")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head_lr_multiplier", type=float, default=10.0)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--missing_value", type=float, default=MISSING_VALUE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max_pos_weight", type=float, default=20.0)
    parser.add_argument("--disable_pos_weight", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=Path("dense_net/artifacts/checkpoints"))
    return parser.parse_args()


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    training: bool,
    pos_weight: torch.Tensor | None = None,
) -> tuple[float, float, dict[str, float]]:
    model.train() if training else model.eval()
    total_loss = 0.0
    all_logits = []
    all_labels = []
    all_masks = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels, masks in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(images)
            loss = masked_bce_loss(logits, labels, masks, pos_weight=pos_weight)

            if training:
                if optimizer is None:
                    raise ValueError("Training epoch requires an optimizer.")
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_masks.append(masks.detach().cpu())

    avg_loss = total_loss / max(len(loader.dataset), 1)

    probabilities = torch.cat(all_logits).sigmoid().numpy()
    labels_np = torch.cat(all_labels).numpy()
    masks_np = torch.cat(all_masks).numpy()
    mean_auc, per_label_auc = compute_multilabel_auc(probabilities, labels_np, masks_np)
    return avg_loss, mean_auc, per_label_auc


def _build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
    train_df = load_manifest(args.train_csv)
    val_df = load_manifest(args.val_csv)

    train_ds = XrayManifestDataset(
        train_df,
        transform=get_image_transforms(image_size=args.image_size, train=True),
        missing_value=args.missing_value,
    )
    val_ds = XrayManifestDataset(
        val_df,
        transform=get_image_transforms(image_size=args.image_size, train=False),
        missing_value=args.missing_value,
    )

    pin_memory = args.device == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_df


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    args: argparse.Namespace,
    val_auc: float,
    per_label_auc: dict[str, float],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "val_auc": val_auc,
            "per_label_auc": per_label_auc,
            "model_name": args.model_name,
            "image_size": args.image_size,
            "dropout": args.dropout,
            "missing_value": args.missing_value,
            "label_cols": LABEL_COLS,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    args.device = get_device()
    seed_everything(args.seed)

    ensure_dir(args.output_dir)
    for path in [args.train_csv, args.val_csv]:
        if not path.exists():
            raise FileNotFoundError(f"Missing input manifest: {path}")

    train_loader, val_loader, train_df = _build_dataloaders(args)

    pos_weight = None
    if not args.disable_pos_weight:
        pos_weight = compute_pos_weight_from_df(
            train_df,
            missing_value=args.missing_value,
            max_pos_weight=args.max_pos_weight,
        ).to(args.device)

    model = build_densenet_model(
        model_name=args.model_name,
        dropout=args.dropout,
        freeze_backbone=True,
    ).to(args.device)

    optimizer = torch.optim.AdamW(
        get_head_parameters(model),
        lr=args.lr * args.head_lr_multiplier,
        weight_decay=args.weight_decay,
    )
    scheduler = None

    metrics_path = args.output_dir / "metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "train_auc", "val_loss", "val_auc"])

    config_summary = {
        "train_csv": str(args.train_csv),
        "val_csv": str(args.val_csv),
        "model_name": args.model_name,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "head_lr_multiplier": args.head_lr_multiplier,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "missing_value": args.missing_value,
        "disable_pos_weight": bool(args.disable_pos_weight),
        "max_pos_weight": args.max_pos_weight,
        "device": args.device,
    }
    if pos_weight is not None:
        config_summary["pos_weight"] = {
            label: float(weight)
            for label, weight in zip(LABEL_COLS, pos_weight.detach().cpu().tolist())
        }
    save_json(config_summary, args.output_dir / "run_config.json")

    best_val_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    best_path = args.output_dir / "best_model.pt"
    last_path = args.output_dir / "last_model.pt"
    best_metrics_path = args.output_dir / "best_metrics.json"

    print(f"Device: {args.device}")
    print(f"Train rows: {len(train_loader.dataset):,}")
    print(f"Val rows:   {len(val_loader.dataset):,}")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        if epoch == args.warmup_epochs + 1:
            set_feature_extractor_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(args.epochs - args.warmup_epochs, 1),
            )

        train_loss, train_auc, train_per_label = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=args.device,
            training=True,
            pos_weight=pos_weight,
        )
        val_loss, val_auc, val_per_label = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=args.device,
            training=False,
            pos_weight=pos_weight,
        )

        if scheduler is not None and epoch > args.warmup_epochs:
            scheduler.step()

        with open(metrics_path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([epoch, train_loss, train_auc, val_loss, val_auc])

        _save_checkpoint(last_path, model, epoch, args, val_auc, val_per_label)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_auc={train_auc:.4f} "
            f"val_loss={val_loss:.4f} val_auc={val_auc:.4f} "
            f"[{elapsed:.0f}s]"
        )
        for label in LABEL_COLS:
            if label in val_per_label:
                print(f"  val {label}: {val_per_label[label]:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(best_path, model, epoch, args, val_auc, val_per_label)
            save_json(
                {
                    "best_epoch": best_epoch,
                    "best_val_auc": best_val_auc,
                    "val_per_label_auc": val_per_label,
                    "train_per_label_auc": train_per_label,
                },
                best_metrics_path,
            )
            print(f"  Saved new best checkpoint: {best_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print(f"Training complete. Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
