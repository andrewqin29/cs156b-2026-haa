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
from torchvision import models, transforms
from tqdm import tqdm

from scipy.optimize import minimize


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
    preds: torch.Tensor,
    raw_labels: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    0 labels (unmentioned) are fully masked out — not trained on at all.
    """
    mask = (raw_labels != 0).float() 

    weights = torch.ones_like(raw_labels)
    if pos_weight is not None:
        pw = pos_weight.view(1, -1).expand_as(raw_labels)
        weights = torch.where(raw_labels == POSITIVE, pw, weights)

    weights = weights * mask

    loss = nn.MSELoss(reduction="none")(preds, raw_labels) * weights
    return loss.sum() / mask.sum().clamp(min=1)

def compute_pos_weights(train_df: pd.DataFrame, label_cols: list[str]) -> torch.Tensor:
    weights = []
    print("Per-class pos_weight:")
    for col in label_cols:
        vals = train_df[col]
        n_pos = (vals ==  1.0).sum()
        n_neg = (vals == -1.0).sum()
        raw_w = n_neg / max(n_pos, 1)
        capped = max(raw_w, 1.0)
        print(f"  {col:<35} n_pos={n_pos:>6}  n_neg={n_neg:>6}  raw_w={raw_w:>6.2f}  capped={capped:.2f}")
        weights.append(capped)
    return torch.tensor(weights, dtype=torch.float32)


def blend_loss(w, frontal_preds, lateral_preds, labels, mask):
    blended = w * frontal_preds + (1-w) * lateral_preds
    err = (blended - labels) ** 2
    return (err * mask).sum() / mask.sum()


def _known_mask(raw_labels: np.ndarray) -> np.ndarray:
    return raw_labels != UNCERTAIN


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
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


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


def aggregate_study_predictions(
    study_ids: list[str],
    view_types: list[str],
    scores: np.ndarray,   
    raw_labels: np.ndarray,  
    frontal_weight: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    studies: dict[str, dict] = defaultdict(
        lambda: {VIEW_FRONTAL: [], VIEW_LATERAL: [], "raw_labels": []}
    )

    for sid, vt, s, rl in zip(study_ids, view_types, scores, raw_labels):
        bucket = vt if vt in (VIEW_FRONTAL, VIEW_LATERAL) else VIEW_FRONTAL
        studies[sid][bucket].append(s)
        studies[sid]["raw_labels"].append(rl)

    sorted_sids = sorted(studies.keys())
    all_scores, all_raw = [], []

    for sid in sorted_sids:
        sd = studies[sid]
        f = np.mean(sd[VIEW_FRONTAL], axis=0) if sd[VIEW_FRONTAL] else None
        l = np.mean(sd[VIEW_LATERAL], axis=0) if sd[VIEW_LATERAL] else None

        if f is not None and l is not None:
            score = frontal_weight * f + (1 - frontal_weight) * l
        else:
            score = f if f is not None else l

        study_raw = np.max(sd["raw_labels"], axis=0)

        all_scores.append(score)
        all_raw.append(study_raw)

    return np.array(all_scores), np.array(all_raw)



class MultiViewDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform,
        path_col: str,
        view_filter: str | None = None,
    ):
        df = df.copy()
        df["_view_type"] = df["Path"].apply(get_view_type)
        df["_study_id"]  = df.apply(get_study_id, axis=1)

        if view_filter is not None:
            df = df[df["_view_type"] == view_filter]

        self.df        = df.reset_index(drop=True)
        self.transform = transform
        self.path_col  = path_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row[self.path_col]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        raw_labels = torch.tensor(
            row[LABEL_COLS].values.astype(np.float32)
        )

        return img, raw_labels, row["_study_id"], row["_view_type"]


def get_transforms(size: int, augment: bool):
    ops = [transforms.Resize((size, size))]
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)



def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: str,
    training: bool,
    pos_weight: torch.Tensor | None = None,
    scaler=None,
) -> tuple[float, float, np.ndarray, np.ndarray, list[str], list[str]]:
  
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds, all_raw_labels = [], []
    all_study_ids: list[str]  = []
    all_view_types: list[str] = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, raw_labels, study_ids, view_types in tqdm(loader, leave=False, mininterval=10):
            imgs       = imgs.to(device, non_blocking=True)
            raw_labels = raw_labels.to(device, non_blocking=True)

            pw = pos_weight.to(device) if (pos_weight is not None and training) else None

            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                preds = model(imgs)
                loss  = masked_mse_loss(preds, raw_labels, pw)

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

            total_loss += loss.item() * imgs.size(0)
            all_preds.append(preds.detach().cpu())
            all_raw_labels.append(raw_labels.detach().cpu())
            all_study_ids.extend(list(study_ids))
            all_view_types.extend(list(view_types))

    scores     = torch.cat(all_preds).numpy()
    raw_labels = torch.cat(all_raw_labels).numpy()

    avg_loss  = total_loss / len(loader.dataset)
    image_auc = _mean_auc(scores, raw_labels) 

    return avg_loss, image_auc, scores, raw_labels, all_study_ids, all_view_types



def train_view_model(
    view_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    args,
    device: str,
    scaler,
    output_dir: Path,
    epoch_csv_path: Path,
    pos_weight
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
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model     = build_model(args.dropout).to(device)
    optimizer = make_optimizer(model, args.lr, phase=1)
    phase     = 1

    best_auc  = 0.0
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
            training=True, pos_weight=pos_weight, scaler=scaler,
        )
        val_loss, val_auc, val_scores, val_raw, val_sids, val_vtypes = run_epoch(
            model, val_loader, optimizer, device,
            training=False,
        )
        elapsed = time.time() - t0

        study_scores, study_raw = aggregate_study_predictions(
            val_sids, val_vtypes, val_scores, val_raw, args.frontal_weight
        )
        study_auc = _mean_auc(study_scores, study_raw)

        print(
            f"  [{view_name}] Epoch {epoch:3d} [phase {phase}]: "
            f"train_auc={train_auc:.4f}  val_img_auc={val_auc:.4f}  "
            f"val_study_auc={study_auc:.4f}  ({elapsed:.0f}s)"
        )

        with open(epoch_csv_path, "a") as f:
            f.write(
                f"{view_name},{epoch},{phase},"
                f"{train_auc:.6f},{val_auc:.6f},{study_auc:.6f},{elapsed:.1f}\n"
            )

        if val_auc > best_auc:
            best_auc = val_auc
            patience_count = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_auc": val_auc,
                    "view": view_name,
                    "img_size": args.img_size,
                    "dropout": args.dropout,
                },
                best_path,
            )
            print(f"    ✓ new best val image-AUC: {best_auc:.4f}")
        else:
            if epoch >= args.unfreeze_epoch:
                patience_count += 1

        if patience_count >= patience_limit:
            print(f"\n  [{view_name}] Early stopping triggered at epoch {epoch}.")
            break


    if best_path.exists():
        print(f"  [{view_name}] Loading best checkpoint (val_img_auc={best_auc:.4f})...")
        ckpt = torch.load(best_path, map_location=device)
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
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    view_scores = {} 
    for model, view_name in [(frontal_model, VIEW_FRONTAL), (lateral_model, VIEW_LATERAL)]:
        if model is None:
            continue
        ds = MultiViewDataset(val_df, get_transforms(args.img_size, False), path_col, view_filter=view_name)
        if len(ds) == 0:
            continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
        _, _, scores, raw, sids, vtypes = run_epoch(
            model, loader, optimizer=None, device=device, training=False
        )
        view_scores[view_name] = (scores, raw, sids, vtypes)

    if not view_scores:
        print("  WARNING: No predictions to aggregate.")
        return

    frontal_study_scores = lateral_study_scores = study_raw = None

    if VIEW_FRONTAL in view_scores and VIEW_LATERAL in view_scores:
        f_scores, f_raw, f_sids, f_vtypes = view_scores[VIEW_FRONTAL]
        l_scores, l_raw, l_sids, l_vtypes = view_scores[VIEW_LATERAL]

        frontal_study_scores, frontal_study_raw = aggregate_study_predictions(
            f_sids, f_vtypes, f_scores, f_raw, frontal_weight=1.0
        )
        lateral_study_scores, lateral_study_raw = aggregate_study_predictions(
            l_sids, l_vtypes, l_scores, l_raw, frontal_weight=0.0
        )

        f_sids_set = set(f_sids)
        l_sids_set = set(l_sids)
        both_views = f_sids_set & l_sids_set

        if len(both_views) > 10:
            sorted_both = sorted(both_views)
            all_f_sids = sorted(set(f_sids))
            all_l_sids = sorted(set(l_sids))
            f_idx = {s: i for i, s in enumerate(all_f_sids)}
            l_idx = {s: i for i, s in enumerate(all_l_sids)}

            f_arr = np.array([frontal_study_scores[f_idx[s]] for s in sorted_both])
            l_arr = np.array([lateral_study_scores[l_idx[s]] for s in sorted_both])
            lbl   = np.array([frontal_study_raw[f_idx[s]]    for s in sorted_both])
            mask  = (lbl != 0)

            result = minimize(
                lambda w: ((w[0] * f_arr + (1-w[0]) * l_arr - lbl) ** 2)[mask].mean(),
                x0=np.array([args.frontal_weight]),
                bounds=[(0, 1)],
                method="L-BFGS-B",
            )
            optimal_weight = float(result.x[0])
            print(f"  Optimal frontal_weight: {optimal_weight:.4f}  (was {args.frontal_weight:.2f})")
        else:
            optimal_weight = args.frontal_weight
            print(f"  Not enough dual-view studies to optimize weight, using {optimal_weight:.2f}")
    else:
        optimal_weight = args.frontal_weight

    all_scores = np.concatenate([v[0] for v in view_scores.values()], axis=0)
    all_raw    = np.concatenate([v[1] for v in view_scores.values()], axis=0)
    all_sids   = sum([v[2] for v in view_scores.values()], [])
    all_vtypes = sum([v[3] for v in view_scores.values()], [])

    img_auc = _mean_auc(all_scores, all_raw)

    study_scores, study_raw = aggregate_study_predictions(
        all_sids, all_vtypes, all_scores, all_raw, optimal_weight
    )
    study_auc    = _mean_auc(study_scores, study_raw)
    study_auc_pc = compute_per_class_auc(study_scores, study_raw)

    print(f"\n  {split_name.upper()} — image-level mean AUC : {img_auc:.4f}")
    print(f"  {split_name.upper()} — study-level mean AUC : {study_auc:.4f}")
    print(f"  (optimal_frontal_weight={optimal_weight:.4f})\n")
    print(f"  {'Class':<35} {'Study AUC':>10}")
    print(f"  {'-'*35} {'-'*10}")
    for name in LABEL_COLS:
        v = study_auc_pc[name]
        print(f"  {name:<35} {f'{v:.4f}' if not np.isnan(v) else '     nan':>10}")

    pd.DataFrame([{"optimal_frontal_weight": optimal_weight}]).to_csv(
        output_dir / "optimal_blend_weight.csv", index=False
    )
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
    parser.add_argument("--frontal_weight",  type=float, default=0.6,
                        help="Blend weight for frontal in study-level aggregation")
    parser.add_argument("--skip_frontal",    action="store_true")
    parser.add_argument("--skip_lateral",    action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    print(
        f"Device: {device}  |  img_size: {args.img_size}  |  batch: {args.batch_size}  |  "
        f"workers: {args.num_workers}  |  lr: {args.lr:.1e}  |  "
        f"unfreeze @ epoch {args.unfreeze_epoch}  |  "
        f"frontal_weight: {args.frontal_weight:.2f}"
    )

    epoch_csv = args.output_dir / "epoch_auc.csv"
    with open(epoch_csv, "w") as f:
        f.write("view,epoch,phase,train_img_auc,val_img_auc,val_study_auc,epoch_time_s\n")

    train_df = pd.read_csv(args.train_csv)
    val_df   = pd.read_csv(args.val_csv)

    pos_weight = compute_pos_weights(train_df, LABEL_COLS)

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
            VIEW_FRONTAL, train_df, val_df, args, device, scaler, args.output_dir, epoch_csv, pos_weight
        )
    else:
        p = args.output_dir / f"best_{VIEW_FRONTAL}.pt"
        if p.exists():
            print(f"\nLoading existing frontal model from {p}")
            ckpt = torch.load(p, map_location=device)
            frontal_model = build_model(ckpt.get("dropout", args.dropout)).to(device)
            frontal_model.load_state_dict(ckpt["model_state"])
        else:
            print(f"\nWARNING: --skip_frontal set but {p} not found.")

    lateral_model = None
    if not args.skip_lateral:
        lateral_model = train_view_model(
            VIEW_LATERAL, train_df, val_df, args, device, scaler, args.output_dir, epoch_csv, pos_weight
        )
    else:
        p = args.output_dir / f"best_{VIEW_LATERAL}.pt"
        if p.exists():
            print(f"\nLoading existing lateral model from {p}")
            ckpt = torch.load(p, map_location=device)
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
    print("  epoch_auc.csv                   — per-epoch AUC by view + phase")
    print("  best_frontal.pt                 — best frontal model weights")
    print("  best_lateral.pt                 — best lateral model weights")
    print("  per_class_study_auc_val.csv     — per-class study-level AUC on val set")
    print("Done.")


if __name__ == "__main__":
    main()