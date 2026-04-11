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
path_parts = df[path_col].str.split("/", expand=True)
df["pid"], df["study"] = path_parts[1], path_parts[2]
df["path"] = df[path_col].str.replace("^train/", training_data_path + "/", regex=True)


# verify per patient label consistency
sort_idx = np.argsort(df["pid"].values)
p_sorted = df["pid"].values[sort_idx]
l_sorted = df[label_cols].fillna(-999).values[sort_idx]
is_same_pid = (p_sorted[:-1] == p_sorted[1:])
is_diff_label = np.any(l_sorted[:-1] != l_sorted[1:], axis=1)
inconsistent = np.unique(p_sorted[:-1][is_same_pid & is_diff_label])

patient_df = df.drop_duplicates("pid")

# create label summary
label_dets = {}
for col in label_cols:
    vc = patient_df[col].value_counts(dropna=False)
    label_dets[col] = {
        "1.0": vc.get(1.0, 0),
        "0.0":  vc.get(0.0, 0),
        "-1.0": vc.get(-1.0, 0),
        "NaN":  patient_df[col].isna().sum(),
    }

spp = df.groupby("pid")["study"].nunique()
ipp = df.groupby("pid").size()

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

lines = [
    "DATA SUMMARY:",
    f"Patients : {df['pid'].nunique()}",
    f"Studies : {df['study'].nunique()}",
    f"Images : {len(df)}", 
    f"Patients with inconsistent labels: {len(inconsistent)}",
    f"Studies/patient — min {spp.min()}, max {spp.max()}, mean {spp.mean():.1f}",
    f"Images/patient  — min {ipp.min()}, max {ipp.max()}, mean {ipp.mean():.1f}",
    "",
    f"Readable images: {len(widths)}",
    f"Unreadable images: {len(failed)}"
]

print("\n".join(lines))