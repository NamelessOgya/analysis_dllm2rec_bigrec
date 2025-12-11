import pandas as pd
import json
import os
import argparse
from tqdm import tqdm

def verify_alignment(data_dir):
    print(f"Verifying data alignment in {data_dir}...")
    
    # Load DataFrame
    df_path = os.path.join(data_dir, 'train_data.df')
    if not os.path.exists(df_path):
        print(f"Error: {df_path} does not exist.")
        return False
    
    print(f"Loading {df_path}...")
    df = pd.read_pickle(df_path)
    
    # Check if 'uid' exists in DataFrame
    if 'uid' not in df.columns:
        print("Error: 'uid' column missing in train_data.df. Did you run the modified process.py?")
        return False
        
    # Load JSON
    json_path = os.path.join(data_dir, 'train.json')
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist.")
        return False
        
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        json_data = json.load(f)
        
    # Check lengths
    if len(df) != len(json_data):
        print(f"Error: Length mismatch! DF: {len(df)}, JSON: {len(json_data)}")
        return False
        
    print(f"Checking {len(df)} records...")
    
    mismatch_count = 0
    for idx, (df_row, json_item) in tqdm(enumerate(zip(df.itertuples(), json_data)), total=len(df)):
        df_uid = df_row.uid
        
        # Check if 'meta' exists in JSON item
        if 'meta' not in json_item or 'uid' not in json_item['meta']:
            print(f"Error at index {idx}: JSON item missing 'meta' or 'uid'.")
            return False
            
        json_uid = json_item['meta']['uid']
        
        if df_uid != json_uid:
            print(f"Mismatch at index {idx}: DF uid {df_uid} != JSON uid {json_uid}")
            mismatch_count += 1
            if mismatch_count > 10:
                print("Too many mismatches, aborting.")
                return False
                
    if mismatch_count == 0:
        print("SUCCESS: train_data.df and train.json are perfectly aligned by UID!")
        return True
    else:
        print(f"FAILED: Found {mismatch_count} mismatches.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./BIGRec/data/game_bigrec", help="Directory containing data files")
    args = parser.parse_args()
    
    verify_alignment(args.data_dir)
