"""
EDA3 — broad exploratory figures for CheXpert-style course data + optional team manifests.

Outputs (under --out_dir):
  - eda3_report.pdf     multi-page figure pack
  - eda3_summary.txt    key counts (plain text)

Defaults match eda1/eda2 (frontal-only train labels + raw image tree).
Optional inputs add patient-split manifests and preprocessed test manifest if present.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

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

DEFAULT_TRAIN_LABELS = Path(
    "/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"
)
DEFAULT_TRAIN_IMG_ROOT = Path("/resnick/groups/CS156b/from_central/data/train")
DEFAULT_TEST_IDS = Path("/resnick/groups/CS156b/from_central/data/student_labels/test_ids.csv")
DEFAULT_MANIFEST_ROOT = Path(
    "/resnick/groups/CS156b/from_central/2026/haa/efficient_net_data/manifests_preprocessed"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate EDA figures (eda3).")
    p.add_argument("--train_csv", type=Path, default=DEFAULT_TRAIN_LABELS)
    p.add_argument("--train_img_root", type=Path, default=DEFAULT_TRAIN_IMG_ROOT)
    p.add_argument("--test_csv", type=Path, default=DEFAULT_TEST_IDS)
    p.add_argument(
        "--manifest_dir",
        type=Path,
        default=DEFAULT_MANIFEST_ROOT,
        help="If these CSVs exist, include extra manifest summary plots.",
    )
    p.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "eda3_output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--include_all_views",
        action="store_true",
        help="Use all rows in train CSV (default: frontal-only, matching eda1/eda2).",
    )
    p.add_argument("--resize_sample", type=int, default=2500, help="Rows to sample for image stats.")
    p.add_argument("--skip_image_io", action="store_true", help="Skip PIL reads (faster; fewer plots).")
    return p.parse_args()


def _extract_patient_id(path_series: pd.Series) -> pd.Series:
    pid = path_series.astype(str).str.extract(r"(pid\d+)", expand=False)
    return pid.fillna(path_series.astype(str))


def _study_id(df: pd.DataFrame) -> pd.Series:
    parts = df["Path"].astype(str).str.split("/", expand=True)
    pid = parts[1] if parts.shape[1] > 1 else pd.Series([""] * len(df))
    study = parts[2] if parts.shape[1] > 2 else pd.Series([""] * len(df))
    return pid.astype(str) + "_" + study.astype(str)


def _abs_train_path(df: pd.DataFrame, train_img_root: Path) -> pd.Series:
    return df["Path"].astype(str).str.replace(r"^train/", str(train_img_root) + "/", regex=True)


def _view_tag(path: str) -> str:
    s = path.lower()
    if "frontal" in s:
        return "frontal"
    if "lateral" in s:
        return "lateral"
    return "other/unknown"


def _label_state_counts(series: pd.Series) -> dict[str, int]:
    vc = series.value_counts(dropna=False)
    out = {"1.0": int(vc.get(1.0, 0) + vc.get(1, 0)), "0.0": int(vc.get(0.0, 0) + vc.get(0, 0))}
    out["-1.0"] = int(vc.get(-1.0, 0) + vc.get(-1, 0))
    out["NaN"] = int(series.isna().sum())
    return out


def _binary_for_corr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """1 -> 1.0, 0 -> 0.0, else NaN (exclude uncertain/missing from correlation)."""
    m = df[cols].apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame(index=m.index)
    for c in cols:
        out[c] = np.where(m[c].isin([0.0, 1.0]), m[c], np.nan)
    return out


def _cooccurrence_given_a_pos(df_bin: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """P(B=1 | A=1) on rows where A is observed and A==1."""
    n = len(cols)
    mat = np.full((n, n), np.nan)
    X = df_bin[cols].to_numpy(dtype=float)
    for i, a in enumerate(cols):
        mask_a = X[:, i] == 1.0
        if mask_a.sum() == 0:
            continue
        for j, b in enumerate(cols):
            sub = X[mask_a, j]
            obs = np.isfinite(sub)
            if obs.sum() == 0:
                continue
            mat[i, j] = float((sub[obs] == 1.0).sum() / obs.sum())
    return mat


def _manifest_paths(mdir: Path) -> dict[str, Path]:
    names = [
        "train_manifest_preprocessed.csv",
        "val_manifest_preprocessed.csv",
        "test_manifest_preprocessed.csv",
        "train_manifest.csv",
        "val_manifest.csv",
    ]
    return {n: mdir / n for n in names if (mdir / n).exists()}


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    frontal_only = not args.include_all_views

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.out_dir / "eda3_report.pdf"
    txt_path = args.out_dir / "eda3_summary.txt"

    df = pd.read_csv(args.train_csv)
    if frontal_only:
        df = df[df["Path"].astype(str).str.contains("frontal", case=False, na=False)].copy()

    df["patient_id"] = _extract_patient_id(df["Path"])
    df["study_id"] = _study_id(df)
    df["abs_path"] = _abs_train_path(df, args.train_img_root)
    df["view_tag"] = df["Path"].map(_view_tag)

    study_df = df.drop_duplicates("study_id")

    lines: list[str] = []
    lines.append("EDA3 SUMMARY")
    lines.append(f"train_csv: {args.train_csv}")
    lines.append(f"frontal_only: {frontal_only}")
    lines.append(f"rows (images): {len(df):,}")
    lines.append(f"unique patients: {df['patient_id'].nunique():,}")
    lines.append(f"unique studies: {df['study_id'].nunique():,}")
    lines.append(f"images / patient: min {df.groupby('patient_id').size().min()} "
                 f"max {df.groupby('patient_id').size().max()} "
                 f"median {df.groupby('patient_id').size().median():.0f}")
    lines.append("view_tag counts (image rows):")
    for k, v in df["view_tag"].value_counts().items():
        lines.append(f"  {k}: {v:,}")
    lines.append("")

    # Label distributions — image level (raw codes)
    lines.append("LABEL RAW COUNTS (image rows):")
    for c in LABEL_COLS:
        d = _label_state_counts(df[c])
        lines.append(f"  {c}: pos={d['1.0']:,} neg={d['0.0']:,} unc={d['-1.0']:,} nan={d['NaN']:,}")
    lines.append("")
    lines.append("LABEL RAW COUNTS (unique studies, first row per study):")
    for c in LABEL_COLS:
        d = _label_state_counts(study_df[c])
        lines.append(f"  {c}: pos={d['1.0']:,} neg={d['0.0']:,} unc={d['-1.0']:,} nan={d['NaN']:,}")

    # Positives per image (among definite 0/1 only)
    m = df[LABEL_COLS].apply(pd.to_numeric, errors="coerce")
    definite = m[m[LABEL_COLS].isin([0.0, 1.0]).all(axis=1)]
    pos_counts = (definite[LABEL_COLS] == 1.0).sum(axis=1)
    lines.append("")
    lines.append(f"rows with all labels in {{0,1}}: {len(definite):,} / {len(df):,}")
    if len(definite):
        lines.append(
            f"positive labels per image (among definite rows): "
            f"mean {pos_counts.mean():.2f}, median {pos_counts.median():.0f}"
        )

    # Test set size
    if args.test_csv.exists():
        tdf = pd.read_csv(args.test_csv)
        lines.append("")
        lines.append(f"test_csv: {args.test_csv} rows={len(tdf):,} cols={list(tdf.columns)[:8]}...")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ---------- Figures ----------
    with PdfPages(pdf_path) as pdf:
        # Page 1 — text summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=8.5, va="top", fontfamily="monospace")
        fig.suptitle("EDA3 — text summary", fontsize=12)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Page 2 — stacked raw label counts (image level)
        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(LABEL_COLS))
        w = 0.2
        img_dets = {c: _label_state_counts(df[c]) for c in LABEL_COLS}
        for i, (key, color, lab) in enumerate(
            [("1.0", "steelblue", "pos"), ("0.0", "coral", "neg"), ("-1.0", "gold", "unc"), ("NaN", "lightgray", "nan")]
        ):
            ax.bar(x + i * w, [img_dets[c][key] for c in LABEL_COLS], width=w, label=lab, color=color, edgecolor="white", linewidth=0.3)
        ax.set_xticks(x + 1.5 * w)
        ax.set_xticklabels(LABEL_COLS, rotation=35, ha="right", fontsize=8)
        ax.set_title("Raw label codes — image rows")
        ax.legend(ncol=4, fontsize=8)
        ax.set_ylabel("count")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Page 3 — same for study-dedup
        fig, ax = plt.subplots(figsize=(12, 5))
        st_dets = {c: _label_state_counts(study_df[c]) for c in LABEL_COLS}
        for i, (key, color, lab) in enumerate(
            [("1.0", "steelblue", "pos"), ("0.0", "coral", "neg"), ("-1.0", "gold", "unc"), ("NaN", "lightgray", "nan")]
        ):
            ax.bar(x + i * w, [st_dets[c][key] for c in LABEL_COLS], width=w, label=lab, color=color, edgecolor="white", linewidth=0.3)
        ax.set_xticks(x + 1.5 * w)
        ax.set_xticklabels(LABEL_COLS, rotation=35, ha="right", fontsize=8)
        ax.set_title("Raw label codes — unique studies")
        ax.legend(ncol=4, fontsize=8)
        ax.set_ylabel("count")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Page 4 — images per patient & studies per patient
        ipc = df.groupby("patient_id").size()
        spc = df.groupby("patient_id")["study_id"].nunique()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(ipc, bins=60, color="steelblue", edgecolor="white")
        axes[0].set_title("Images per patient")
        axes[0].set_xlabel("count")
        axes[1].hist(spc, bins=60, color="coral", edgecolor="white")
        axes[1].set_title("Studies per patient (distinct study_id)")
        axes[1].set_xlabel("count")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Page 5 — view tags
        fig, ax = plt.subplots(figsize=(6, 4))
        vc = df["view_tag"].value_counts()
        ax.bar(vc.index.astype(str), vc.values, color=["steelblue", "coral", "gray"][: len(vc)])
        ax.set_title("View tag (from Path string)")
        ax.set_ylabel("images")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Page 6 — positive rate among definite 0/1 rows
        if len(definite):
            pr = (definite[LABEL_COLS] == 1.0).mean()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(len(LABEL_COLS)), pr.values, color="mediumpurple", edgecolor="white")
            ax.set_xticks(range(len(LABEL_COLS)))
            ax.set_xticklabels(LABEL_COLS, rotation=35, ha="right", fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("P(label=1 | row has 0/1 for all labels)")
            ax.set_title("Positive prevalence (fully-observed 0/1 rows only)")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

        # Page 7 — histogram of number of positive labels
        if len(definite):
            fig, ax = plt.subplots(figsize=(7, 4))
            k = int(pos_counts.max()) + 1
            ax.hist(pos_counts, bins=np.arange(-0.5, k + 0.5, 1), color="teal", edgecolor="white")
            ax.set_title("Number of positive labels per image (definite rows)")
            ax.set_xlabel("# positives")
            ax.set_ylabel("images")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

        # Page 8 — correlation among binary labels (0/1 only)
        bdf = _binary_for_corr(df, LABEL_COLS)
        if bdf.dropna(how="all").shape[0] > 50:
            C = bdf.corr(method="pearson", min_periods=200)
            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.imshow(C.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
            ax.set_xticks(range(len(LABEL_COLS)))
            ax.set_yticks(range(len(LABEL_COLS)))
            ax.set_xticklabels(LABEL_COLS, rotation=35, ha="right", fontsize=7)
            ax.set_yticklabels(LABEL_COLS, fontsize=7)
            ax.set_title("Pearson correlation on binary labels (non-NaN 0/1 only; pairwise counts vary)")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

        # Page 9 — P(B=1 | A=1)
        co = _cooccurrence_given_a_pos(bdf, LABEL_COLS)
        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(co, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(LABEL_COLS)))
        ax.set_yticks(range(len(LABEL_COLS)))
        ax.set_xticklabels(LABEL_COLS, rotation=35, ha="right", fontsize=7)
        ax.set_yticklabels(LABEL_COLS, fontsize=7)
        ax.set_title("Co-occurrence: P(B=1 | A=1) — rows where A observed & positive")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close()

        # Page 10 — optional demographics
        demo_cols = [c for c in ["Sex", "Age", "Frontal/Lateral", "AP/PA"] if c in df.columns]
        if demo_cols:
            fig, axes = plt.subplots(1, len(demo_cols), figsize=(4 * len(demo_cols), 4))
            if len(demo_cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, demo_cols):
                if col == "Age":
                    s = pd.to_numeric(df[col], errors="coerce")
                    ax.hist(s.dropna(), bins=40, color="steelblue", edgecolor="white")
                    ax.set_title("Age")
                else:
                    vc = df[col].astype(str).value_counts().head(20)
                    ax.barh(vc.index[::-1], vc.values[::-1], color="coral")
                    ax.set_title(col)
            fig.suptitle("Demographics / view metadata (if present in train CSV)", fontsize=11)
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

        # Pages 11+ — image size / mode sample
        if not args.skip_image_io and len(df) > 0:
            n = min(args.resize_sample, len(df))
            idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
            sample = df.loc[idx]
            widths, heights, modes, failed = [], [], [], []
            for fp in sample["abs_path"]:
                if not isinstance(fp, str) or not os.path.exists(fp):
                    failed.append(fp)
                    continue
                try:
                    with Image.open(fp) as im:
                        widths.append(im.width)
                        heights.append(im.height)
                        modes.append(im.mode)
                except Exception:
                    failed.append(fp)

            if widths:
                fig = plt.figure(figsize=(10, 8))
                gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], hspace=0.05, wspace=0.05)
                ax_main = fig.add_subplot(gs[1, 0])
                ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
                ax_side = fig.add_subplot(gs[1, 1], sharey=ax_main)
                ax_main.scatter(widths, heights, alpha=0.12, s=4, color="steelblue")
                ax_main.set_xlabel("width (px)")
                ax_main.set_ylabel("height (px)")
                ax_top.hist(widths, bins=50, color="steelblue")
                ax_top.axis("off")
                ax_side.hist(heights, bins=50, orientation="horizontal", color="steelblue")
                ax_side.axis("off")
                fig.suptitle(f"Resolution sample (n={len(widths)} readable / {n} tried)", fontsize=12)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()

                ar = np.array(widths) / np.maximum(np.array(heights), 1)
                fig, axes = plt.subplots(1, 2, figsize=(11, 4))
                axes[0].hist(ar, bins=60, color="mediumpurple", edgecolor="white")
                axes[0].axvline(1.0, color="red", linestyle="--", linewidth=1)
                axes[0].set_title("Aspect ratio w/h")
                sizes_kb = []
                for fp in sample["abs_path"]:
                    if isinstance(fp, str) and os.path.exists(fp):
                        try:
                            sizes_kb.append(os.path.getsize(fp) / 1024)
                        except OSError:
                            pass
                if sizes_kb:
                    axes[1].hist(sizes_kb, bins=60, color="coral", edgecolor="white")
                    axes[1].set_title("File size (KB, sampled paths)")
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()

                mc = Counter(modes)
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.pie(mc.values(), labels=[f"{k} ({v})" for k, v in mc.items()], autopct="%1.1f%%")
                ax.set_title("PIL image modes (sample)")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close()

        # Manifest appendix
        mpaths = _manifest_paths(args.manifest_dir)
        if mpaths:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(mpaths))))
            ax.axis("off")
            rows = []
            for name, path in sorted(mpaths.items()):
                try:
                    mdf = pd.read_csv(path)
                    row = f"{name}: rows={len(mdf):,} cols={len(mdf.columns)}"
                    if "preprocessed_path" in mdf.columns:
                        exists = mdf["preprocessed_path"].astype(str).map(lambda p: Path(p).exists())
                        row += f" | preprocessed_path exists: {exists.mean()*100:.1f}%"
                    if "patient_id" in mdf.columns:
                        row += f" | patients={mdf['patient_id'].nunique():,}"
                    rows.append(row)
                except Exception as e:
                    rows.append(f"{name}: ERROR reading {e}")
            ax.text(0.02, 0.98, "Manifest files\n\n" + "\n".join(rows), va="top", fontsize=9, family="monospace", transform=ax.transAxes)
            fig.suptitle(f"Manifest dir: {args.manifest_dir}", fontsize=11)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close()

    print(f"Wrote {pdf_path}")
    print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
