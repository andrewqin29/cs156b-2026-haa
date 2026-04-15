import pandas as pd

def update_csv_pandas(input_filepath, output_filepath):
    df = pd.read_csv(input_filepath)
    
    if 'No Finding' in df.columns:
        df['No Finding'] = 0
        
        df.to_csv(output_filepath, index=False)
        print(f"Success! Updated CSV saved to {output_filepath}")
    else:
        print("Error: The column 'No Finding' was not found in the CSV.")

update_csv_pandas("./submissions/resnet_alex_preprocessed.csv", "./submissions/resnet_alex_preprocessed_nofinding1.csv")