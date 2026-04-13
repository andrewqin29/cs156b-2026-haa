"""
Initial data preprocessing script
 - resize images to single standard (224 x 224 to be used with pretrained ResNet)
 - fill Uncertain values (-1.0s) with 1.0s
 - keep NaNs for now (relabel as -999)

 source: https://medium.com/@maahip1304/the-complete-guide-to-image-preprocessing-techniques-in-python-dca30804550c

 - Note: no normalization done here, NORMALIZE BEFORE TRAINING 
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

training_data_path = "/resnick/groups/CS156b/from_central/data/train"
training_labels_path = "/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv"
output_folder = "/resnick/groups/CS156b/from_central/2026/haa/preprocessed_front_images_1"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# load data
df = pd.read_csv(training_labels_path)
df = df[df["Path"].str.contains("frontal", case=False)].copy()

label_cols = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
              "Pneumonia", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

#relabel -1 (uncertain) with 1 (treat as positive case)
df[label_cols] = df[label_cols].replace(-1.0, 1.0)

#relabel NaNs with -999
df[label_cols] = df[label_cols].fillna(-999)

image_size = (224, 224)
new_paths = []
print("started processing")

for idx, row in tqdm(df.iterrows(), total=len(df)):
    original_path = row["Path"].replace("train/", training_data_path + "/")
    file_name = f"{row['Path'].replace('/', '_')}"
    save_path = os.path.join(output_folder, file_name)

    if os.path.exists(save_path):
        new_paths.append(save_path)
        continue
    try:
        with Image.open(original_path) as img:
            img = img.convert("L").resize(image_size, Image.LANCZOS)
            img.save(save_path)
            new_paths.append(save_path)
    except Exception as e:
        new_paths.append(None)

df["preprocessed_path"] = new_paths
df_final = df.dropna(subset=["preprocessed_path"])
df_final.to_csv("preprocessed_labels.csv", index=False)
print("Preprocessing Complete.")