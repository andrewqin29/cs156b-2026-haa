import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

# Data format: /train/pid00605/study1/view1_frontal.jpg

training_data_path = "/resnick/groups/CS156b/from_central/data/train"
training_labels_path = "/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"
path_col = "Path"
output_pdf = "data_summary.pdf"

label_cols = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
                "Pneumonia", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]


# load data
df = pd.read_csv(training_labels_path)
df["pid"] = df[path_col].str.split("/").str[1]
df["study"] = df[path_col].str.split("/").str[2]
df["path"] = df[path_col].str.replace("^train/", training_data_path + "/", regex=True)


# verify per patient label consistency
inconsistent = [pid for pid, g in df.groupby("pid") if g[label_cols].nunique().gt(1).any()]
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

# load images
images = df["path"]
widths, heights, failed = [], [], []
for image in images:
    try:
        im = Image.open(image)
        widths.append(im.width); heights.append(im.height)
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