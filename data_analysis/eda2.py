import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter

# Data format: /train/pid00605/study1/view1_frontal.jpg

training_data_path = "/resnick/groups/CS156b/from_central/data/train"
training_labels_path = "/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"
path_col = "Path"
output_pdf = "data_summary.pdf"

label_cols = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
                "Pneumonia", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]


# load data
df = pd.read_csv(training_labels_path)
# filter for frontal images only for now
df = df[df[path_col].str.contains("frontal", case=False)].copy()

path_parts = df[path_col].str.split("/", expand=True)
df["pid"], df["study"] = path_parts[1], path_parts[2]
df["path"] = df[path_col].str.replace("^train/", training_data_path + "/", regex=True)
df["study_id"] = df["pid"] + "_" + df["study"]

study_df = df.drop_duplicates("study_id")

# create label summary
label_dets = {}
for col in label_cols:
    vc = study_df[col].value_counts(dropna=False)
    label_dets[col] = {
        "1.0": vc.get(1.0, 0),
        "0.0":  vc.get(0.0, 0),
        "-1.0": vc.get(-1.0, 0),
        "NaN":  study_df[col].isna().sum(),
    }


# load sample of images
sample_pids = np.random.choice(df["pid"].unique(), size=500, replace=False)
sample_df = df[df["pid"].isin(sample_pids)]

widths, heights, modes, failed, good_paths = [], [], [], [], []
for image in sample_df["path"]:
    try:
        with Image.open(image) as im:
            widths.append(im.width)
            heights.append(im.height)
            modes.append(im.mode)
            good_paths.append(image)
    except Exception as e:
        failed.append(image)

avg_w = np.mean(widths) if widths else 0
avg_h = np.mean(heights) if heights else 0

aspect_ratios = [w / h for w, h in zip(widths, heights)]
mode_counts = Counter(modes)

lines = [
    "DATA SUMMARY:",
    f"Patients : {df['pid'].nunique()}",
    f"Unique Studies : {df['study_id'].nunique()}",
    f"Images : {len(df)}",
    "",
    f"Readable images: {len(widths)}",
    f"Unreadable images: {len(failed)}",
    f"Width  (px) — min {min(widths)}, max {max(widths)}, mean {np.mean(widths):.0f}, std {np.std(widths):.0f}",
    f"Height (px) — min {min(heights)}, max {max(heights)}, mean {np.mean(heights):.0f}, std {np.std(heights):.0f}",
    f"Aspect ratio — min {min(aspect_ratios):.2f}, max {max(aspect_ratios):.2f}, mean {np.mean(aspect_ratios):.2f}",
    "",
    "Color modes:",
] + [f"  {m}: {c}" for m, c in mode_counts.items()] + [
    "",
    "LABEL BREAKDOWN (per patient):",
    f"  {'Label':<35} {'pos':>6} {'neg':>6} {'unc':>6} {'NaN':>6}",
] + [f"  {col:<35} {label_dets[col]['1.0']:>6} {label_dets[col]['0.0']:>6} {label_dets[col]['-1.0']:>6} {label_dets[col]['NaN']:>6}"
     for col in label_cols]
 

print("\n".join(lines))


#generate pdf with figures
with PdfPages(output_pdf) as pdf:

    #page 1
    fig, ax = plt.subplots(figsize=(10,8)); ax.axis("off")
    ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=9.5, va="top", fontfamily="monospace")
    pdf.savefig(fig, bbox_inches = "tight"); plt.close()

    #page 2
    fig, ax = plt.subplots(figsize=(12, 5))
    x, w = np.arange(len(label_cols)), 0.2
    for i, (key, color, l) in enumerate([("1.0", "steelblue", "Positive"), ("0.0", "coral", "Negative"),
        ("-1.0", "gold", "Uncertain"), ("NaN", "lightgrey", "NaN")]):
        ax.bar(x + i * w, [label_dets[c][key] for c in label_cols], width=w, label=l, color=color)
    ax.set_xticks(x + 1.5 * w); ax.set_xticklabels(label_cols, rotation=30, ha="right", fontsize=9)
    ax.set_title("Label Distribution per Pathology (per patient)"); ax.legend()
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close()

    #page3
    fig = plt.figure(figsize=(10, 8))
    gs  = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 3], hspace=0.05, wspace=0.05)
    ax_main = fig.add_subplot(gs[1, 0])
    ax_top  = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_side = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_main.scatter(widths, heights, alpha=0.1, s=3, color="steelblue")
    ax_main.set_xlabel("Width (px)"); ax_main.set_ylabel("Height (px)")
    ax_top.hist(widths, bins=60, color="steelblue"); ax_top.axis("off")
    ax_side.hist(heights, bins=60, orientation="horizontal", color="steelblue"); ax_side.axis("off")
    fig.suptitle("Image Dimensions", fontsize=13)
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
 
    #page4
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(aspect_ratios, bins=60, color="mediumpurple")
    axes[0].axvline(1.0, color="red", linestyle="--", label="Square (1:1)")
    axes[0].set_xlabel("Aspect ratio (w/h)"); axes[0].set_title("Aspect Ratio Distribution"); axes[0].legend()
    sizes_kb = [os.path.getsize(fp) / 1024 for fp in sample_df["path"] if os.path.exists(fp)]
    axes[1].hist(sizes_kb, bins=60, color="coral")
    axes[1].set_xlabel("File size (KB)"); axes[1].set_title("File Size Distribution")
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close()
 
    #page5
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(mode_counts.values(), labels=mode_counts.keys(), autopct="%1.1f%%",
           colors=["steelblue", "coral", "gold", "lightgrey"])
    ax.set_title("Color Modes")
    pdf.savefig(fig, bbox_inches="tight"); plt.close()
 
    #page6
    fig, axes = plt.subplots(len(mode_counts), 4, figsize=(12, 3 * len(mode_counts)))
    axes = np.array(axes)
    if axes.ndim == 1: axes = axes[np.newaxis, :]
    mode_list = list(mode_counts.keys())
    for i, mode in enumerate(mode_list):
        idxs = [j for j, m in enumerate(modes) if m == mode][:4]
        for j in range(4):
            axes[i, j].axis("off")
            if j < len(idxs):
                to_show = good_paths[idxs[j]]
                axes[i, j].imshow(Image.open(to_show), cmap="gray" if mode == "L" else None)
            if j == 0:
                axes[i, j].set_title(f"Mode: {mode}", fontsize=9, loc="left")
    fig.suptitle("Sample Images by Color Mode", fontsize=13)
    plt.tight_layout(); pdf.savefig(fig, bbox_inches="tight"); plt.close()