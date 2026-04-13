import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

test_ids_path = "/resnick/groups/CS156b/data/student_labels/test_ids.csv"
test_data_path = "/groups/CS156b/data/test"
output_folder = "/resnick/groups/CS156b/from_central/2026/haa/preprocessed_test_images"

os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv(test_ids_path)
image_size = (224, 224)
new_paths = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    original_path = row["Path"].replace("test/", test_data_path + "/")
    file_name = row["Path"].replace("/", "_")
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
        print(f"Error on {original_path}: {e}")
        new_paths.append(None)

df["preprocessed_path"] = new_paths
df.to_csv("preprocessed_test_labels.csv", index=False)
print("Done.")