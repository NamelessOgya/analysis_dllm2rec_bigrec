import pandas as pd
import numpy as np
import os
import ast
import csv
import argparse

# Config
parse = argparse.ArgumentParser()
parse.add_argument("--dataset", type=str, default="movie", help="dataset name")
args = parse.parse_args()

DATASET = args.dataset
BIGREC_DATA_DIR = f'BIGRec/data/{DATASET}'
DLLM2REC_DATA_DIR = f'DLLM2Rec/data/{DATASET}' # Reverted to match main.py expectation (line 442)

# Ensure output directory exists
os.makedirs(DLLM2REC_DATA_DIR, exist_ok=True)

def load_bigrec_csv(path):
    print(f"Loading {path}...")
    return pd.read_csv(path)

# Load all data to build global item mapping
train_df = load_bigrec_csv(os.path.join(BIGREC_DATA_DIR, 'train.csv'))
valid_df = load_bigrec_csv(os.path.join(BIGREC_DATA_DIR, 'valid.csv'))
test_df = load_bigrec_csv(os.path.join(BIGREC_DATA_DIR, 'test.csv'))

# Collect all unique movie IDs
all_movie_ids = set()

def extract_ids(df):
    for _, row in df.iterrows():
        # Target movie
        all_movie_ids.add(str(row['item_id'])) # Changed directly to item_id for game_bigrec? 
        # CSV header said 'item_id', 'item_asin'. 
        # Original code used 'movie_id'. Let's check CSV header again. 
        # Header: user_id,item_asins,item_asin,history_item_id,item_id,...
        # So 'item_id' seems correct standard.
        # But 'movie' dataset might have 'movie_id'.
        # Let's try flexible column name.
        
        target_col = 'movie_id' if 'movie_id' in df.columns else 'item_id'
        hist_col = 'history_movie_id' if 'history_movie_id' in df.columns else 'history_item_id'
        
        all_movie_ids.add(str(row[target_col]))
        
        # History movies
        history_ids = eval(str(row[hist_col])) 
        for mid in history_ids:
            all_movie_ids.add(str(mid))
            
    return target_col, hist_col

# Load id2name.txt just to get max item count and verify universe
id2name_path = os.path.join(BIGREC_DATA_DIR, 'id2name.txt')
print(f"Loading item count from {id2name_path}...")
item_count = 0
with open(id2name_path, 'r') as f:
    for line in f:
        item_count += 1
print(f"Total items in id2name.txt: {item_count}")

# SASRec needs 1-based indexing (0 is padding).
# BIGRec uses 0-based indexing (0 to item_count-1).
# Mapping: SASRec_ID = BIGRec_ID + 1

item_num = item_count

def convert_row(row, target_col, hist_col):
    # History
    # history_ids_raw is list of integers in string format e.g. "[10804, 15747]"
    # or list of strings if mixed? 
    # train.csv sample: "[10804, 15747, ...]" -> looks like ints.
    history_ids_raw = eval(str(row[hist_col]))
    
    # We assume history_ids_raw are already BIGRec IDs (integers).
    # Map to 1-based
    seq = [int(mid) + 1 for mid in history_ids_raw]
    
    # Target
    # target_col is 'item_id' (integer)
    target = int(row[target_col]) + 1
    
    return seq, len(seq), target

def process_df(df, is_train=False):
    # Determine columns. 'item_id' is standard in game_bigrec.
    target_col = 'item_id'
    if target_col not in df.columns:
        # Fallback if inconsistent
        target_col = 'movie_id' if 'movie_id' in df.columns else 'item_id'
    
    hist_col = 'history_item_id'
    if hist_col not in df.columns:
         hist_col = 'history_movie_id' if 'history_movie_id' in df.columns else 'history_item_id'
    
    data_list = []
    for _, row in df.iterrows():
        try:
            seq, len_seq, target = convert_row(row, target_col, hist_col)
            
            item = {
                'seq': seq,
                'len_seq': len_seq,
                'next': target
            }
            
            # Capture UID
            if 'uid' in row:
                 item['uid'] = int(row['uid'])
            elif 'user_id' in row:
                 try:
                     item['uid'] = int(row['user_id'])
                 except:
                     item['uid'] = -1
            
            data_list.append(item)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    new_df = pd.DataFrame(data_list)
    return new_df

print("Processing Train...")
train_converted = process_df(train_df, is_train=True)
# Save Train as pickle
train_converted.to_pickle(os.path.join(DLLM2REC_DATA_DIR, 'train_data.df'))

print("Processing Valid...")
valid_converted = process_df(valid_df)
# Save Valid as CSV
valid_converted.to_csv(os.path.join(DLLM2REC_DATA_DIR, 'val_data.csv'), index=False)

print("Processing Test...")
test_converted = process_df(test_df)
# Save Test as CSV
test_converted.to_csv(os.path.join(DLLM2REC_DATA_DIR, 'test_data.csv'), index=False)

# Create data_statis.df
seq_size = 200 
statis_data = {
    'seq_size': [seq_size], 
    'item_num': [item_num]
}
statis_df = pd.DataFrame(statis_data)
statis_df.to_pickle(os.path.join(DLLM2REC_DATA_DIR, 'data_statis.df'))

# Save item_map check (Identity mapping shifted by 1)
with open(os.path.join(DLLM2REC_DATA_DIR, 'item_map.txt'), 'w') as f:
    f.write(f"Identity Mapping (BIGRec ID N -> SASRec ID N+1)\n")
    f.write(f"Total Items: {item_num}\n")

print("Conversion complete.")
print(f"Files saved to {DLLM2REC_DATA_DIR}")
