import pandas as pd
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=Path, default="/resnick/groups/CS156b/from_central/data/student_labels/train2023.csv")
    p.add_argument("--out_file", type=Path, default="dataset_stats.txt")
    args = p.parse_args()

    df = pd.read_csv(args.train_csv)
    
    # Extract IDs: Path is typically 'train/patientXXXXX/studyX/viewX.jpg'
    path_parts = df['Path'].str.split('/')
    df['patient_id'] = path_parts.str[1]
    df['study_id'] = path_parts.str[1] + "_" + path_parts.str[2]

    # Aggregations
    studies_per_patient = df.groupby('patient_id')['study_id'].nunique()
    imgs_per_patient = df.groupby('patient_id').size()
    imgs_per_study = df.groupby('study_id').size()

    stats = [
        f"Dataset Summary: {args.train_csv.name}",
        "-" * 30,
        f"Total Images:   {len(df):,}",
        f"Total Patients: {df['patient_id'].nunique():,}",
        f"Total Studies:  {df['study_id'].nunique():,}",
        "",
        "Studies per Patient:",
        studies_per_patient.describe().to_string(),
        "",
        "Images per Patient:",
        imgs_per_patient.describe().to_string(),
        "",
        "Images per Study:",
        imgs_per_study.describe().to_string()
    ]

    output = "\n".join(stats)
    print(output)
    
    with open(args.out_file, "w") as f:
        f.write(output)
    print(f"\nStats saved to → {args.out_file}")

if __name__ == "__main__":
    main()