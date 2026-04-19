from __future__ import annotations
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from PIL import Image

C_POS, C_NEG, C_UNC, C_NAN = "#4C8CB5", "#E07B54", "#E8C13A", "#CCCCCC"
LABEL_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Pneumonia", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"
]
SHORT_LABELS = [
    "No Find", "Enl Card", "Cardio", "Opacity", "Pneumo", "Effus", "Other", "Frac", "Support"
]

DEFAULT_TRAIN_CSV = Path(
    "/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"
)
DEFAULT_IMG_ROOT = Path("/resnick/groups/CS156b/from_central/data/train")

def apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.grid": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "DejaVu Sans", "font.size": 10
    })

def save(fig, path, dpi):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path.name}")

def load_grey(path: str, size: int = 256) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            return np.asarray(im.convert("L").resize((size, size), Image.LANCZOS), dtype=np.float32)
    except: return None

def fig_label_stats(df, out_dir, dpi):
    counts = {c: {"pos": (df[c]==1).sum(), "neg": (df[c]==-1).sum(), "unc": (df[c]==0).sum()} for c in LABEL_COLS}
    pos = [counts[c]["pos"] for c in LABEL_COLS]
    neg = [counts[c]["neg"] for c in LABEL_COLS]
    unc = [counts[c]["unc"] for c in LABEL_COLS]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(LABEL_COLS))
    ax.bar(x, pos, color=C_POS, label="Positive (1)")
    ax.bar(x, neg, bottom=pos, color=C_NEG, label="Negative (-1)")
    ax.bar(x, unc, bottom=np.array(pos)+np.array(neg), color=C_UNC, label="Uncertain (0)")
    ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS, rotation=30)
    ax.set_title("Label Distributions"); ax.legend()
    save(fig, out_dir / "01_label_distributions.png", dpi)

    fig, ax = plt.subplots(figsize=(12, 4))
    prev = [(df[c]==1).sum() / df[c].isin([-1,1]).sum() * 100 for c in LABEL_COLS]
    ax.bar(x, prev, color=C_POS)
    ax.set_xticks(x); ax.set_xticklabels(SHORT_LABELS, rotation=30)
    ax.set_ylabel("Positive Rate (%)")
    ax.set_title("Condition Prevalence (Definite Cases Only)")
    save(fig, out_dir / "02_label_prevalence.png", dpi)

def fig_demographics(df, out_dir, dpi):
    if "Age" not in df.columns or "Sex" not in df.columns: return
    fig, ax = plt.subplots(figsize=(8, 5))
    for s, c in zip(["Male", "Female"], [C_POS, C_NEG]):
        sub = df[df["Sex"].str.startswith(s[0], na=False)]
        ax.hist(sub["Age"], bins=30, alpha=0.6, label=s, color=c)
    ax.set_title("Patient Demographics"); ax.legend(); ax.set_xlabel("Age")
    save(fig, out_dir / "04_demographics.png", dpi)

def fig_technical_analysis(df, out_dir, dpi, n, rng):
    idx = rng.choice(df.index, size=min(n, len(df)), replace=False)
    paths = df.loc[idx, "abs_path"]
    
    stats = {"bright": [], "contrast": [], "sharp": [], "ratio": [], "sym": []}
    
    for p in paths:
        if not os.path.exists(p): continue
        try:
            with Image.open(p) as im:
                stats["ratio"].append(im.width / im.height)
                arr = np.asarray(im.convert("L"), dtype=np.float32)
                stats["bright"].append(np.mean(arr))
                stats["contrast"].append(np.std(arr))
                
                lap = (np.roll(arr,1,0)+np.roll(arr,-1,0)+np.roll(arr,1,1)+np.roll(arr,-1,1)-4*arr)
                stats["sharp"].append(np.var(lap))
                
                w_mid = arr.shape[1] // 2
                left, right = arr[:, :w_mid], np.fliplr(arr[:, w_mid:w_mid*2])
                stats["sym"].append(np.corrcoef(left.ravel(), right.ravel())[0,1])
        except: continue

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    metrics = [("bright", "Brightness", C_POS), ("contrast", "Contrast", C_NEG), 
               ("sharp", "Sharpness", C_UNC), ("ratio", "Aspect Ratio", "#9B59B6")]
    for ax, (k, t, c) in zip(axes, metrics):
        ax.hist(stats[k], bins=35, color=c, alpha=0.8)
        ax.set_title(t); ax.axvline(np.median(stats[k]), color="black", ls="--")
    save(fig, out_dir / "06_technical_metrics.png", dpi)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(stats["sym"], bins=40, color="#1ABC9C")
    ax.set_title("Bilateral Symmetry Score (L/R Correlation)")
    ax.set_xlabel("Correlation Coefficient"); ax.set_xlim(0, 1)
    save(fig, out_dir / "07_symmetry_analysis.png", dpi)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--img_root", type=Path, default=DEFAULT_IMG_ROOT)
    p.add_argument("--out_dir", type=Path, default=Path("eda_output"))
    p.add_argument("--sample", type=int, default=2000)
    args = p.parse_args()
    
    apply_style()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.train_csv)
    df["abs_path"] = df["Path"].apply(lambda x: str(args.img_root / x.replace("train/","")))
    rng = np.random.default_rng(42)

    print("Generating Technical EDA...")
    fig_label_stats(df, args.out_dir, 150)
    fig_demographics(df, args.out_dir, 150)
    fig_technical_analysis(df, args.out_dir, 150, args.sample, rng)
    print("Done.")

if __name__ == "__main__":
    main()