import pandas as pd
import numpy as np
import os
from PIL import Image

# Data format: /train/pid00605/study1/view1_frontal.jpg

training_data_path = "/resnick/groups/CS156b/from_central/data/train"
training_labels_path = "/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"
path_col = "Path"

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

widths, heights, failed = [], [], []
for image in sample_df["path"]:
    try:
        with Image.open(image) as im:
            widths.append(im.width)
            heights.append(im.height)
    except Exception as e:
        failed.append(image)

avg_w = np.mean(widths) if widths else 0
avg_h = np.mean(heights) if heights else 0

lines = [
    "DATA SUMMARY (FRONTAL ONLY):",
    f"Patients (with frontal images) : {df['pid'].nunique()}",
    f"Unique Clinical Studies        : {df['study_id'].nunique()}",
    f"Total Frontal Image Files      : {len(df)}",
    "",
    f"Sampled Readable Images        : {len(widths)}",
    f"Sampled Failed Images          : {len(failed)}",
    f"Avg Resolution (Sample)        : {avg_w:.0f}x{avg_h:.0f}px",
    "",
    f"{'Pathology':<30} {'Pos':>8} {'Neg':>8} {'Unc':>8} {'NaN':>8}",
]

for col in label_cols:
    d = label_dets[col]
    lines.append(f"{col:<30} {d['1.0']:>8} {d['0.0']:>8} {d['-1.0']:>8} {d['NaN']:>8}")

print("\n".join(lines))