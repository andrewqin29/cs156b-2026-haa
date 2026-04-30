from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import v2
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

POSITIVE  =  1.0
NEGATIVE  = -1.0
UNCERTAIN =  0.0
SENTINEL = -999.0 #for new nan handling

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

VIEW_FRONTAL = "frontal"
VIEW_LATERAL = "lateral"
VIEW_UNKNOWN = "unknown"


def get_view_type(path_str: str) -> str:
    fname = Path(path_str).name.lower()
    if "frontal" in fname:
        return VIEW_FRONTAL
    if "lateral" in fname:
        return VIEW_LATERAL
    return VIEW_UNKNOWN


def get_study_id(row: pd.Series) -> str:

    parts = Path(str(row["Path"])).parts
    if len(parts) >= 3:
        return f"{parts[-3]}_{parts[-2]}"
    return str(row.get("patient_id", "unknown"))


def build_model(dropout: float = 0.4) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, NUM_CLASSES),
        nn.Tanh()
    )
    _freeze_backbone(model)
    return model


def _freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc")


def _unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def make_optimizer(model: nn.Module, lr: float, phase: int) -> torch.optim.Optimizer:
    if phase == 1:
        return torch.optim.AdamW(model.fc.parameters(), lr=lr * 10, weight_decay=1e-4)
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    return torch.optim.AdamW(
        [
            {"params": backbone_params,       "lr": lr,      "weight_decay": 1e-4},
            {"params": model.fc.parameters(), "lr": lr * 10, "weight_decay": 1e-4},
        ]
    )


def masked_mse_loss(
    preds:              torch.Tensor,
    raw_labels:         torch.Tensor,
    variance:           torch.Tensor | None = None,
    pw:                 torch.Tensor | None = None,
    nw:                 torch.Tensor | None = None,
    consistency_lambda: float = 0.0,
) -> torch.Tensor:
    consistency_lambda = consistency_lambda or 0.0

    mask        = (raw_labels != SENTINEL).float()
    weights     = torch.ones_like(raw_labels)

    if pw is not None:
        pw_expanded = pw.view(1, -1).expand_as(raw_labels)
        weights     = torch.where(raw_labels == POSITIVE, pw_expanded, weights)
    if nw is not None:
        nw_expanded = nw.view(1, -1).expand_as(raw_labels)
        weights     = torch.where(raw_labels == NEGATIVE, nw_expanded, weights)

    weights     = weights * mask
    safe_labels = raw_labels.clone()
    safe_labels[raw_labels == SENTINEL] = 0.0

    loss_matrix = nn.MSELoss(reduction="none")(preds, safe_labels)

    # Variance scaling — divide each class loss by its label variance
    if variance is not None:
        var         = variance.to(preds.device).clamp(min=1e-6)
        loss_matrix = loss_matrix / var.view(1, -1)

    base_loss = (loss_matrix * weights).sum() / mask.sum().clamp(min=1)

    no_finding  = preds[:, 0]
    pathologies = preds[:, 1:]
    penalty     = torch.relu(no_finding.unsqueeze(1)) * torch.relu(pathologies)

    return base_loss + (consistency_lambda * penalty.mean())


def compute_scaled_mse(
    scores:    np.ndarray,
    raw_labels: np.ndarray,
    variance:  torch.Tensor,
) -> tuple[float, dict[str, float]]:
    """
    Computes variance-scaled MSE matching the competition metric.
    Returns (mean_scaled_mse, per_class_dict).
    """
    var_np = variance.numpy()
    per_class = {}
    scaled_mses = []

    for i, name in enumerate(LABEL_COLS):
        mask = (raw_labels[:, i] != SENTINEL)
        if mask.sum() < 2:
            per_class[name] = float("nan")
            continue
        mse = ((scores[mask, i] - raw_labels[mask, i]) ** 2).mean()
        scaled = mse / max(var_np[i], 1e-6)
        per_class[name] = float(scaled)
        scaled_mses.append(scaled)

    mean_scaled = float(np.mean(scaled_mses)) if scaled_mses else float("nan")
    return mean_scaled, per_class

def compute_class_weights(
    train_df: pd.DataFrame,
    label_cols: list[str],
    max_weight: float = 2.0,
    min_neg_frac: float = 0.05,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos_weights = []
    neg_weights = []
    if verbose:
        print("Per-class weights:")
    for col in label_cols:
        vals  = train_df[col]
        n_pos = (vals == POSITIVE).sum()
        n_neg = (vals == NEGATIVE).sum()
        total = max(n_pos + n_neg, 1)

        if n_pos < n_neg:
            pw = min(total / (2 * max(n_pos, 1)), max_weight)
            nw = 1.0
        elif n_neg < n_pos:
            nw = min(total / (2 * max(n_neg, 1)), max_weight) \
                 if n_neg / total >= min_neg_frac else 1.0
            pw = 1.0
        else:
            pw, nw = 1.0, 1.0

        if verbose:
            print(
                f"  {col:<35} n_pos={n_pos:>6}  n_neg={n_neg:>6}  "
                f"pos_w={pw:>6.2f}  neg_w={nw:.2f}"
            )
        pos_weights.append(pw)
        neg_weights.append(nw)
    return (
        torch.tensor(pos_weights, dtype=torch.float32),
        torch.tensor(neg_weights, dtype=torch.float32),
    )

def compute_label_variance(train_df: pd.DataFrame, label_cols: list[str]) -> torch.Tensor:
    variances = []
    print("Per-class label variance:")
    for col in label_cols:
        vals  = train_df[col]
        known = vals[vals != SENTINEL]
        var   = float(known.var())
        print(f"  {col:<35} variance={var:.4f}")
        variances.append(var)
    return torch.tensor(variances, dtype=torch.float32)


def _known_mask(raw_labels: np.ndarray) -> np.ndarray:
    return (raw_labels == POSITIVE) | (raw_labels == NEGATIVE)


def _binary_from_raw(raw_labels: np.ndarray) -> np.ndarray:
    """Convert {-1, 0, +1} → {0, 0, 1} for AUC (uncertain → 0, masked out anyway)."""
    return (raw_labels == POSITIVE).astype(np.float32)


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true  = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Proper tie-aware ranking
    order = np.argsort(y_score)
    sorted_scores = y_score[order]

    ranks = np.zeros_like(sorted_scores, dtype=np.float64)
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[i:j+1] = avg_rank
        i = j + 1

    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks

    return (full_ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _mean_auc(scores: np.ndarray, raw_labels: np.ndarray) -> float:
    known = _known_mask(raw_labels)
    binary = _binary_from_raw(raw_labels)
    aucs = []
    for c in range(NUM_CLASSES):
        m = known[:, c]
        if m.sum() < 2:
            continue
        y = binary[m, c]
        if len(np.unique(y)) < 2:
            continue
        aucs.append(_binary_auc(y, scores[m, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def compute_per_class_auc(scores: np.ndarray, raw_labels: np.ndarray) -> dict[str, float]:
    known  = _known_mask(raw_labels)
    binary = _binary_from_raw(raw_labels)
    out = {}
    for i, name in enumerate(LABEL_COLS):
        m = known[:, i]
        if m.sum() < 2 or len(np.unique(binary[m, i])) < 2:
            out[name] = float("nan")
        else:
            out[name] = _binary_auc(binary[m, i], scores[m, i])
    return out


def aggregate_study_predictions(study_ids, scores, raw_labels):
    studies = defaultdict(lambda: {"scores": [], "labels": []})
    for sid, s, rl in zip(study_ids, scores, raw_labels):
        studies[sid]["scores"].append(s)
        studies[sid]["labels"].append(rl)

    all_scores, all_raw = [], []
    for sid in sorted(studies.keys()):
        sd = studies[sid]
        stacked = np.array(sd["scores"])
        if len(stacked) == 1:
            study_score = stacked[0]
        else:
            confidence = np.abs(stacked) 
            weights = confidence / (confidence.sum(axis=0, keepdims=True) + 1e-8)
            study_score = (stacked * weights).sum(axis=0)

        all_scores.append(study_score)
        labels = np.array(sd["labels"])
        study_label = np.zeros(labels.shape[1])
        for c in range(labels.shape[1]):
            col = labels[:, c]
            if (col == 1.0).any():
                study_label[c] = 1.0
            elif (col == -1.0).any():
                study_label[c] = -1.0
            elif (col == 0.0).any():
                study_label[c] = 0.0
            else:
                study_label[c] = SENTINEL
        all_raw.append(study_label)

    return np.array(all_scores), np.array(all_raw)



class MultiViewDataset(Dataset):
    def __init__(self, df, transform, path_col, view_filter=None):
        df = df.copy()
        df["_view_type"] = df["Path"].apply(get_view_type)
        df["_study_id"]  = df.apply(get_study_id, axis=1)
        if view_filter is not None:
            df = df[df["_view_type"] == view_filter]
        self.df        = df.reset_index(drop=True)
        self.transform = transform

        # Pre-extract for fast indexing
        self.paths      = self.df[path_col].tolist()
        self.labels     = self.df[LABEL_COLS].values.astype(np.float32)
        self.study_ids  = self.df["_study_id"].tolist()
        self.view_types = self.df["_view_type"].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return (
            img,
            torch.tensor(self.labels[idx]),
            self.study_ids[idx],
            self.view_types[idx],
        )


def get_transforms(size: int, augment: bool):
    ops = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((size, size), antialias=True)
    ]
    if augment:
        ops += [
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(10),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    ops += [
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return v2.Compose(ops)


def run_epoch(
    model:              nn.Module,
    loader:             DataLoader,
    optimizer,
    device:             str,
    training:           bool,
    variance:           torch.Tensor | None = None,
    pos_weight:         torch.Tensor | None = None,
    neg_weight:         torch.Tensor | None = None,
    consistency_lambda: float | None = None,
    scaler=None,
) -> tuple[float, float, np.ndarray, np.ndarray, list[str], list[str]]:
  
    model.train() if training else model.eval()

    total_loss_gpu = torch.tensor(0.0, device=device)
    all_preds, all_raw_labels = [], []
    all_study_ids, all_view_types = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        var_device = variance.to(device) if variance is not None else None
        pw = pos_weight.to(device) if (pos_weight is not None and training) else None
        nw = neg_weight.to(device) if (neg_weight is not None and training) else None

        for imgs, raw_labels, study_ids, view_types in tqdm(loader, leave=False, mininterval=10):
            imgs       = imgs.to(device, non_blocking=True)
            raw_labels = raw_labels.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                preds = model(imgs).float()
                loss = masked_mse_loss(preds, raw_labels, var_device, pw, nw, consistency_lambda)

            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss_gpu += loss.detach() * imgs.size(0)
            all_preds.append(preds.detach())
            all_raw_labels.append(raw_labels.detach())
            all_study_ids.extend(list(study_ids))
            all_view_types.extend(list(view_types))

    scores     = torch.cat(all_preds).cpu().numpy()
    raw_labels = torch.cat(all_raw_labels).cpu().numpy()

    avg_loss = total_loss_gpu.item() / len(loader.dataset)
    image_auc = _mean_auc(scores, raw_labels) 

    return avg_loss, image_auc, scores, raw_labels, all_study_ids, all_view_types



def train_view_model(
    view_name:  str,
    train_df:   pd.DataFrame,
    val_df:     pd.DataFrame,
    args,
    device:     str,
    scaler,
    output_dir: Path,
    epoch_csv_path: Path,
    variance:   torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
    neg_weight: torch.Tensor | None = None,
) -> nn.Module | None:

    print(f"\n{'='*60}")
    print(f"  Training {view_name.upper()} model")
    print(f"{'='*60}")

    path_col = "preprocessed_path" if "preprocessed_path" in train_df.columns else "abs_path"

    train_ds = MultiViewDataset(train_df, get_transforms(args.img_size, True),  path_col, view_filter=view_name)
    val_ds   = MultiViewDataset(val_df,   get_transforms(args.img_size, False), path_col, view_filter=view_name)

    if len(train_ds) == 0:
        print(f"  WARNING: No {view_name} images found in training set. Skipping.")
        return None

    print(f"  {view_name}: {len(train_ds)} train images, {len(val_ds)} val images")

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model     = build_model(args.dropout).to(device)
    optimizer = make_optimizer(model, args.lr, phase=1)
    phase     = 1

    best_mse = float("inf")
    best_path = output_dir / f"best_{view_name}.pt"

    patience_count = 0
    patience_limit = args.patience

    for epoch in range(1, args.epochs + 1):

        if epoch == args.unfreeze_epoch and phase == 1:
            print(
                f"\n  Epoch {epoch}: unfreezing backbone — "
                f"backbone lr={args.lr:.1e}, fc lr={args.lr*10:.1e}"
            )
            _unfreeze_backbone(model)
            optimizer = make_optimizer(model, args.lr, phase=2)
            phase = 2

        t0 = time.time()
        train_loss, train_auc, *_ = run_epoch(
            model, train_loader, optimizer, device,
            training=True,
            variance=variance,
            pos_weight=pos_weight,
            neg_weight=neg_weight,
            consistency_lambda=args.consistency_lambda,
            scaler=scaler,
        )

        val_loss, val_auc, val_scores, val_raw, val_sids, _ = run_epoch(
            model, val_loader, optimizer, device,
            training=False,
            variance=variance,
            consistency_lambda=args.consistency_lambda,
        )
        elapsed = time.time() - t0

        study_scores, study_raw = aggregate_study_predictions(val_sids, val_scores, val_raw)
        study_auc = _mean_auc(study_scores, study_raw)

        # Unweighted MSEs for reference
        known_mask_img   = (val_raw   != SENTINEL)
        known_mask_study = (study_raw != SENTINEL)
        val_mse   = ((val_scores[known_mask_img]     - val_raw[known_mask_img])     ** 2).mean()
        study_mse = ((study_scores[known_mask_study] - study_raw[known_mask_study]) ** 2).mean()

        # Variance-scaled study MSE — competition metric
        scaled_study_mse, scaled_per_class = compute_scaled_mse(study_scores, study_raw, variance)

        print(
            f"  [{view_name}] Epoch {epoch:3d} [phase {phase}]: "
            f"train_auc={train_auc:.4f}  train_mse={train_loss:.4f}  "
            f"val_study_auc={study_auc:.4f}  study_mse={study_mse:.4f}  "
            f"scaled_mse={scaled_study_mse:.4f}  ({elapsed:.0f}s)"
        )

        # CSV — add scaled_study_mse column
        with open(epoch_csv_path, "a") as f:
            f.write(
                f"{view_name},{epoch},{phase},"
                f"{train_auc:.6f},{train_loss:.6f},{val_auc:.6f},{study_auc:.6f},"
                f"{val_mse:.6f},{study_mse:.6f},{scaled_study_mse:.6f},{elapsed:.1f}\n"
            )

        # Checkpoint on scaled MSE
        if scaled_study_mse < best_mse:
            best_mse = scaled_study_mse
            patience_count = 0
            torch.save({
                "model_state":    model.state_dict(),
                "epoch":          int(epoch),
                "scaled_mse":     float(scaled_study_mse),
                "study_mse":      float(study_mse),
                "val_auc":        float(val_auc),
                "view":           str(view_name),
                "img_size":       int(args.img_size),
                "dropout":        float(args.dropout),
            }, best_path)
            print(f"    ✓ new best scaled MSE: {best_mse:.4f}")
            # Per-class scaled MSE
            print(f"    {'Class':<35} {'Scaled MSE':>10}")
            for name in LABEL_COLS:
                v = scaled_per_class[name]
                print(f"    {name:<35} {f'{v:.4f}' if not np.isnan(v) else '     nan':>10}")
        else:
            if epoch >= args.unfreeze_epoch:
                patience_count += 1

        if patience_count >= patience_limit:
            print(f"\n  [{view_name}] Early stopping triggered at epoch {epoch}.")
            break


    if best_path.exists():
        print(f"  [{view_name}] Loading best checkpoint (scaled_mse={best_mse:.4f})...")        
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    else:
        print(f"  WARNING: No best checkpoint found for {view_name}!")
        
    return model


def evaluate_combined(
    frontal_model, lateral_model, val_df, args, device, output_dir, split_name="val"
):
    path_col = "preprocessed_path" if "preprocessed_path" in val_df.columns else "abs_path"
    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    view_scores = {} 
    for model, view_name in [(frontal_model, VIEW_FRONTAL), (lateral_model, VIEW_LATERAL)]:
        if model is None: continue
        ds = MultiViewDataset(val_df, get_transforms(args.img_size, False), path_col, view_filter=view_name)
        if len(ds) == 0: continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        
        _, _, scores, raw, sids, _ = run_epoch(
            model, loader, None, device, False, variance=None, consistency_lambda=args.consistency_lambda
        )
        view_scores[view_name] = (scores, raw, sids)

    if not view_scores:
        print("  WARNING: No predictions to aggregate.")
        return

    all_scores = np.concatenate([v[0] for v in view_scores.values()], axis=0)
    all_raw    = np.concatenate([v[1] for v in view_scores.values()], axis=0)
    all_sids   = sum([v[2] for v in view_scores.values()], [])

    img_auc = _mean_auc(all_scores, all_raw)

    study_scores, study_raw = aggregate_study_predictions(all_sids, all_scores, all_raw)
    
    study_auc    = _mean_auc(study_scores, study_raw)
    study_auc_pc = compute_per_class_auc(study_scores, study_raw)

    print(f"\n  {split_name.upper()} — image-level mean AUC : {img_auc:.4f}")
    print(f"  {split_name.upper()} — study-level mean AUC : {study_auc:.4f}")
    print(f"  {'Class':<35} {'Study AUC':>10}")
    print(f"  {'-'*35} {'-'*10}")
    for name in LABEL_COLS:
        v = study_auc_pc[name]
        print(f"  {name:<35} {f'{v:.4f}' if not np.isnan(v) else '     nan':>10}")

    rows = [{"class": k, f"{split_name}_study_auc": v} for k, v in study_auc_pc.items()]
    pd.DataFrame(rows).to_csv(output_dir / f"per_class_study_auc_{split_name}.csv", index=False)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv",       type=Path,  required=True)
    parser.add_argument("--val_csv",         type=Path,  required=True)
    parser.add_argument("--output_dir",      type=Path,  default=Path("out"))
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--batch_size",      type=int,   default=128)
    parser.add_argument("--lr",              type=float, default=2e-4)
    parser.add_argument("--dropout",         type=float, default=0.4)
    parser.add_argument("--img_size",        type=int,   default=512)
    parser.add_argument("--unfreeze_epoch",  type=int,   default=2)
    parser.add_argument("--num_workers",     type=int,   default=6)
    parser.add_argument("--skip_frontal",    action="store_true")
    parser.add_argument("--skip_lateral",    action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--consistency_lambda", type=float, default=0.1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    print(
        f"Device: {device}  |  img_size: {args.img_size}  |  batch: {args.batch_size}  |  "
        f"workers: {args.num_workers}  |  lr: {args.lr:.1e}  |  "
        f"unfreeze @ epoch {args.unfreeze_epoch}"
    )

    epoch_csv = args.output_dir / "epoch_auc_lateral.csv"
    with open(epoch_csv, "w") as f:
        f.write("view,epoch,phase,train_img_auc,train_mse,val_img_auc,val_study_auc,"
            "val_img_mse,val_study_mse,val_scaled_mse,epoch_time_s\n")
    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)

    variance = compute_label_variance(train_df, LABEL_COLS)

    train_df["_view_type"] = train_df["Path"].apply(get_view_type)
    val_df["_view_type"]   = val_df["Path"].apply(get_view_type)

    print(
        f"\nTrain: {len(train_df)} images  "
        f"({(train_df['_view_type']=='frontal').sum()} frontal, "
        f"{(train_df['_view_type']=='lateral').sum()} lateral)"
    )
    print(
        f"Val  : {len(val_df)} images  "
        f"({(val_df['_view_type']=='frontal').sum()} frontal, "
        f"{(val_df['_view_type']=='lateral').sum()} lateral)"
    )

    frontal_model = None
    if not args.skip_frontal:
        frontal_model = train_view_model(
            VIEW_FRONTAL, train_df, val_df, args, device, scaler,
            args.output_dir, epoch_csv,
            variance=variance,
            pos_weight=None,        # no class weights
            neg_weight=None,
        )
    else:
        p = args.output_dir / f"best_{VIEW_FRONTAL}.pt"
        if p.exists():
            print(f"\nLoading existing frontal model from {p}")
            ckpt = torch.load(p, map_location=device, weights_only=False)
            frontal_model = build_model(ckpt.get("dropout", args.dropout)).to(device)
            frontal_model.load_state_dict(ckpt["model_state"])
        else:
            print(f"\nWARNING: --skip_frontal set but {p} not found.")

    lateral_model = None
    if not args.skip_lateral:
        lateral_model = train_view_model(
            VIEW_LATERAL, train_df, val_df, args, device, scaler,
            args.output_dir, epoch_csv,
            variance=variance,
            pos_weight=None,        # no class weights
            neg_weight=None,
        )
    else:
        p = args.output_dir / f"best_{VIEW_LATERAL}.pt"
        if p.exists():
            print(f"\nLoading existing lateral model from {p}")
            ckpt = torch.load(p, map_location=device, weights_only=False)
            lateral_model = build_model(ckpt.get("dropout", args.dropout)).to(device)
            lateral_model.load_state_dict(ckpt["model_state"])
        else:
            print(f"\nWARNING: --skip_lateral set but {p} not found.")

    print(f"\n{'='*60}")
    print("  Combined study-level evaluation (best checkpoints)")
    print(f"{'='*60}")

    if frontal_model is not None:
        frontal_model.eval()
    if lateral_model is not None:
        lateral_model.eval()

    evaluate_combined(frontal_model, lateral_model, val_df, args, device, args.output_dir)

    print(f"\nOutputs saved to {args.output_dir}/")
    print("  epoch_auc_lateral.csv                   — per-epoch AUC by view + phase")
    print("  best_frontal.pt                 — best frontal model weights")
    print("  best_lateral.pt                 — best lateral model weights")
    print("  per_class_study_auc_val.csv     — per-class study-level AUC on val set")
    print("Done.")


if __name__ == "__main__":
    main()