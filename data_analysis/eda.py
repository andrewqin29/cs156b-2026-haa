"""
eda4.py — focused EDA for CheXpert-style course data.

Outputs (under --out_dir), one PNG per figure:
  01_label_distributions.png      — stacked bar per condition (pos / neg / unc / nan)
  02_label_prevalence.png         — positive rate among definite 0/1 rows, per condition
  03_resolution_scatter.png       — width vs height scatter + marginal histograms
  04_filesize_hist.png            — file size distribution (KB)
  05_grid_<label>.png             — grid of sample images per condition (positive examples)
  06_unc_vs_pos_<label>.png       — uncertain (-1) vs positive (1) side-by-side per condition

All images are read lazily; pass --skip_image_io to skip PIL-dependent figures.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from PIL import Image

# ── colour palette ──────────────────────────────────────────────────────────
C_POS = "#4C8CB5"   # steelblue
C_NEG = "#E07B54"   # coral
C_UNC = "#E8C13A"   # amber
C_NAN = "#CCCCCC"   # light grey

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

SHORT_LABELS = [
    "No Finding",
    "Enl. Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

DEFAULT_TRAIN_CSV = Path(
    "/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"
)
DEFAULT_IMG_ROOT = Path("/resnick/groups/CS156b/from_central/data/train")


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Focused EDA figures (eda4).")
    p.add_argument("--train_csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--img_root", type=Path, default=DEFAULT_IMG_ROOT)
    p.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "eda4_output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frontal_only", action="store_true", default=True,
                   help="Use only frontal images (default: True).")
    p.add_argument("--resolution_sample", type=int, default=3000,
                   help="Number of images to sample for resolution/size stats.")
    p.add_argument("--grid_n", type=int, default=12,
                   help="Images per label grid (shown as NxN grid, rounded down to square).")
    p.add_argument("--unc_pos_n", type=int, default=6,
                   help="Images per class (uncertain / positive) in side-by-side figures.")
    p.add_argument("--skip_image_io", action="store_true",
                   help="Skip all PIL reads (produces only label-stat figures).")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


# ── helpers ──────────────────────────────────────────────────────────────────
def abs_path(path_col: pd.Series, img_root: Path) -> pd.Series:
    return path_col.astype(str).str.replace(
        r"^train/", str(img_root) + "/", regex=True
    )


def label_counts(series: pd.Series) -> dict[str, int]:
    s = pd.to_numeric(series, errors="coerce")
    return {
        "pos": int((s == 1.0).sum()),
        "neg": int((s == 0.0).sum()),
        "unc": int((s == -1.0).sum()),
        "nan": int(s.isna().sum()),
    }


def sample_paths(df: pd.DataFrame, label: str, code: float, n: int, rng: np.random.Generator) -> list[str]:
    """Return up to n absolute paths where label == code and file exists."""
    mask = pd.to_numeric(df[label], errors="coerce") == code
    sub = df.loc[mask, "abs_path"].dropna().tolist()
    rng.shuffle(sub)
    found = []
    for p in sub:
        if len(found) >= n:
            break
        if os.path.exists(p):
            found.append(p)
    return found


def load_grey(path: str, size: int = 256) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize((size, size), Image.LANCZOS)
            return np.asarray(im)
    except Exception:
        return None


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F7F7F7",
        "axes.edgecolor": "#CCCCCC",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.frameon": False,
    })


def save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved → {path.name}")


# ── figure 1: stacked label distribution bars ────────────────────────────────
def fig_label_distributions(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    counts = {c: label_counts(df[c]) for c in LABEL_COLS}
    pos = [counts[c]["pos"] for c in LABEL_COLS]
    neg = [counts[c]["neg"] for c in LABEL_COLS]
    unc = [counts[c]["unc"] for c in LABEL_COLS]
    nan = [counts[c]["nan"] for c in LABEL_COLS]

    x = np.arange(len(LABEL_COLS))
    fig, ax = plt.subplots(figsize=(13, 5))

    b1 = ax.bar(x, pos, color=C_POS, label="Positive (1)", zorder=3)
    b2 = ax.bar(x, neg, bottom=pos, color=C_NEG, label="Negative (0)", zorder=3)
    bot3 = [p + n for p, n in zip(pos, neg)]
    b3 = ax.bar(x, unc, bottom=bot3, color=C_UNC, label="Uncertain (−1)", zorder=3)
    bot4 = [b + u for b, u in zip(bot3, unc)]
    b4 = ax.bar(x, nan, bottom=bot4, color=C_NAN, label="Unlabelled (NaN)", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, rotation=35, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("Label distribution per condition (frontal images)")
    ax.legend(loc="upper right", ncol=4)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # annotate positive counts on top of bars
    for xi, p in zip(x, pos):
        if p > 0:
            ax.text(xi, p / 2, f"{p:,}", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold", zorder=4)

    fig.tight_layout()
    save(fig, out_dir / "01_label_distributions.png", dpi)


# ── figure 2: positive prevalence among definite 0/1 rows ───────────────────
def fig_label_prevalence(df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    m = df[LABEL_COLS].apply(pd.to_numeric, errors="coerce")
    prevalence = []
    n_definite = []
    for c in LABEL_COLS:
        mask = m[c].isin([0.0, 1.0])
        n_definite.append(mask.sum())
        prevalence.append(float((m.loc[mask, c] == 1.0).mean()) if mask.sum() else 0.0)

    x = np.arange(len(LABEL_COLS))
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(x, [p * 100 for p in prevalence], color=C_POS, zorder=3, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT_LABELS, rotation=35, ha="right")
    ax.set_ylabel("Positive rate (%)")
    ax.set_title("Positive prevalence per condition\n(among images with a definite 0 or 1 label)")
    ax.set_ylim(0, 100)

    for xi, pct, nd in zip(x, prevalence, n_definite):
        ax.text(xi, pct * 100 + 1.5, f"{pct*100:.1f}%\nn={nd:,}", ha="center",
                va="bottom", fontsize=7.5, color="#333333")

    fig.tight_layout()
    save(fig, out_dir / "02_label_prevalence.png", dpi)


# ── figure 3: resolution scatter + marginals ─────────────────────────────────
def fig_resolution(df: pd.DataFrame, out_dir: Path, dpi: int, n: int, rng: np.random.Generator) -> None:
    idx = rng.choice(df.index.to_numpy(), size=min(n, len(df)), replace=False)
    sample = df.loc[idx, "abs_path"].tolist()

    widths, heights = [], []
    for p in sample:
        if not os.path.exists(p):
            continue
        try:
            with Image.open(p) as im:
                widths.append(im.width)
                heights.append(im.height)
        except Exception:
            pass

    if not widths:
        print("  [resolution] no readable images found — skipping.")
        return

    fig = plt.figure(figsize=(9, 8))
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.05, wspace=0.05
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top  = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_side = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(widths, heights, alpha=0.15, s=6, color=C_POS, linewidths=0)
    ax_main.set_xlabel("Width (px)")
    ax_main.set_ylabel("Height (px)")

    ax_top.hist(widths, bins=60, color=C_POS, edgecolor="none")
    ax_top.set_ylabel("Count")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)

    ax_side.hist(heights, bins=60, color=C_POS, edgecolor="none", orientation="horizontal")
    ax_side.set_xlabel("Count")
    plt.setp(ax_side.get_yticklabels(), visible=False)
    ax_side.spines["top"].set_visible(False)
    ax_side.spines["right"].set_visible(False)

    med_w, med_h = np.median(widths), np.median(heights)
    ax_main.axvline(med_w, color="red", linestyle="--", linewidth=1, label=f"Median W={med_w:.0f}")
    ax_main.axhline(med_h, color="orange", linestyle="--", linewidth=1, label=f"Median H={med_h:.0f}")
    ax_main.legend(fontsize=8)

    fig.suptitle(f"Image resolution  (n={len(widths):,} sampled)", fontsize=13, y=0.98)
    save(fig, out_dir / "03_resolution_scatter.png", dpi)


# ── figure 4: file size histogram ────────────────────────────────────────────
def fig_filesize(df: pd.DataFrame, out_dir: Path, dpi: int, n: int, rng: np.random.Generator) -> None:
    idx = rng.choice(df.index.to_numpy(), size=min(n, len(df)), replace=False)
    sizes_kb = []
    for p in df.loc[idx, "abs_path"]:
        if os.path.exists(p):
            try:
                sizes_kb.append(os.path.getsize(p) / 1024)
            except OSError:
                pass

    if not sizes_kb:
        print("  [filesize] no readable paths — skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(sizes_kb, bins=80, color=C_NEG, edgecolor="none", zorder=3)
    ax.axvline(np.median(sizes_kb), color="red", linestyle="--", linewidth=1.2,
               label=f"Median {np.median(sizes_kb):.0f} KB")
    ax.set_xlabel("File size (KB)")
    ax.set_ylabel("Number of images")
    ax.set_title(f"File size distribution  (n={len(sizes_kb):,} sampled)")
    ax.legend()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))
    fig.tight_layout()
    save(fig, out_dir / "04_filesize_hist.png", dpi)


# ── figure 5: image grids per label ──────────────────────────────────────────
def fig_image_grid(df: pd.DataFrame, label: str, out_dir: Path, dpi: int, n: int, rng: np.random.Generator) -> None:
    paths = sample_paths(df, label, 1.0, n, rng)
    if not paths:
        print(f"  [grid] no positive images for '{label}' — skipping.")
        return

    cols = min(4, len(paths))
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.8))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for ax, p in zip(axes, paths):
        img = load_grey(p)
        if img is not None:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")

    slug = label.lower().replace(" ", "_").replace("/", "_")
    fig.suptitle(f"Positive examples — {label}  (n={len(paths)})", fontsize=12, y=1.01)
    fig.tight_layout(pad=0.5)
    save(fig, out_dir / f"05_grid_{slug}.png", dpi)


# ── figure 6: uncertain vs positive side-by-sides ────────────────────────────
def fig_unc_vs_pos(df: pd.DataFrame, label: str, out_dir: Path, dpi: int, n: int, rng: np.random.Generator) -> None:
    pos_paths = sample_paths(df, label, 1.0, n, rng)
    unc_paths = sample_paths(df, label, -1.0, n, rng)

    if not pos_paths and not unc_paths:
        print(f"  [unc_vs_pos] no images for '{label}' — skipping.")
        return

    n_pos = len(pos_paths)
    n_unc = len(unc_paths)
    n_rows = max(n_pos, n_unc)

    fig, axes = plt.subplots(n_rows, 2, figsize=(5.5, n_rows * 2.7))
    axes = np.array(axes).reshape(n_rows, 2)

    for row in range(n_rows):
        for col, (paths, code_label) in enumerate([(pos_paths, "Positive (1)"), (unc_paths, "Uncertain (−1)")]):
            ax = axes[row, col]
            ax.axis("off")
            if row < len(paths):
                img = load_grey(paths[row])
                if img is not None:
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            if row == 0:
                color = C_POS if col == 0 else C_UNC
                ax.set_title(code_label, fontsize=11, color=color, fontweight="bold", pad=6)

    slug = label.lower().replace(" ", "_").replace("/", "_")
    fig.suptitle(f"{label} — Positive vs. Uncertain", fontsize=13, y=1.01)
    fig.tight_layout(pad=0.5)
    save(fig, out_dir / f"06_unc_vs_pos_{slug}.png", dpi)


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    apply_style()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.train_csv} …")
    df = pd.read_csv(args.train_csv)
    if args.frontal_only:
        df = df[df["Path"].astype(str).str.contains("frontal", case=False, na=False)].copy()
    df["abs_path"] = abs_path(df["Path"], args.img_root)

    print(f"  {len(df):,} rows  |  {df['Path'].nunique():,} unique paths")
    print(f"Output dir: {args.out_dir}\n")

    # ── label stat figures (no IO needed) ────────────────────────────────────
    print("01  label distributions …")
    fig_label_distributions(df, args.out_dir, args.dpi)

    print("02  positive prevalence …")
    fig_label_prevalence(df, args.out_dir, args.dpi)

    if args.skip_image_io:
        print("\n--skip_image_io set — skipping all PIL figures.")
        return

    # ── resolution & file size ────────────────────────────────────────────────
    print(f"03  resolution scatter (sampling {args.resolution_sample:,}) …")
    fig_resolution(df, args.out_dir, args.dpi, args.resolution_sample, rng)

    print(f"04  file size histogram (sampling {args.resolution_sample:,}) …")
    fig_filesize(df, args.out_dir, args.dpi, args.resolution_sample, rng)

    # ── per-label image grids ─────────────────────────────────────────────────
    print(f"05  image grids ({args.grid_n} images per label) …")
    for label in LABEL_COLS:
        print(f"    {label}")
        fig_image_grid(df, label, args.out_dir, args.dpi, args.grid_n, rng)

    # ── uncertain vs positive side-by-sides ───────────────────────────────────
    print(f"06  uncertain vs positive ({args.unc_pos_n} per class per label) …")
    for label in LABEL_COLS:
        counts = label_counts(df[label])
        if counts["unc"] == 0:
            print(f"    {label} — no uncertain examples, skipping side-by-side.")
            continue
        print(f"    {label}")
        fig_unc_vs_pos(df, label, args.out_dir, args.dpi, args.unc_pos_n, rng)

    print("\nDone.")


if __name__ == "__main__":
    main()