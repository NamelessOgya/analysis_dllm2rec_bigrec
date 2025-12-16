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
DLLM2REC_DATA_DIR = f'DLLM2Rec/tocf/{DATASET}' # Updated to match where main.py expects it (based on run_dllm2rec_train.sh defaults usually, or verify?) 
# run_dllm2rec_train.sh uses: DLLM2Rec/tocf/{args.data}/ if not provided. 
# But wait, earlier script used DLLM2Rec/data/movie. 
# main.py defaults to `tocf/{args.data}` if embedding_path not provided?
# No, main.py data loading:
# args.data_path default is 'tocf/'. 
# Then `data_path = args.data_path + args.data`.
# So if dataset is 'game_bigrec', path is `tocf/game_bigrec`.
# Let's use `DLLM2Rec/tocf/{DATASET}`.

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

print("Extracting IDs from Train...")
t_col, h_col = extract_ids(train_df)
print("Extracting IDs from Valid...")
extract_ids(valid_df)
print("Extracting IDs from Test...")
extract_ids(test_df)

# Create mapping: Original ID (str) -> Mapped ID (int, 1-based)
sorted_ids = sorted(list(all_movie_ids))
id_map = {original_id: i + 1 for i, original_id in enumerate(sorted_ids)}
item_num = len(id_map)

print(f"Total items: {item_num}")

def convert_row(row, target_col, hist_col):
    # History
    history_ids_raw = eval(str(row[hist_col]))
    seq = [id_map[str(mid)] for mid in history_ids_raw]
    
    # Target
    target = id_map[str(row[target_col])]
    
    return seq, len(seq), target

def process_df(df, is_train=False):
    target_col = 'movie_id' if 'movie_id' in df.columns else 'item_id'
    hist_col = 'history_movie_id' if 'history_movie_id' in df.columns else 'history_item_id'
    
    data_list = []
    for _, row in df.iterrows():
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
             # Fallback if uid not present? But header had 'uid'.
             try:
                 item['uid'] = int(row['user_id'])
             except:
                 item['uid'] = -1
        
        data_list.append(item)
    
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
seq_size = 200 # SASRec default usually 200? BIGRec prompt len 10. 
# Check run_sasrec_baseline.sh args? max_len=200.
# So statis should report what?
# Actually run_sasrec_baseline says: ./cmd/run_sasrec_baseline.sh game_bigrec 1 200 ...
# The 200 is max_len.
# statis_df seq_size should probably match max_len or be sufficient.
# Let's assume 200 if not specified.
statis_data = {
    'seq_size': [200], 
    'item_num': [item_num]
}
statis_df = pd.DataFrame(statis_data)
statis_df.to_pickle(os.path.join(DLLM2REC_DATA_DIR, 'data_statis.df'))

# Save id_map check
with open(os.path.join(DLLM2REC_DATA_DIR, 'item_map.txt'), 'w') as f:
    for k, v in id_map.items():
        f.write(f"{k}\t{v}\n")

print("Conversion complete.")
print(f"Files saved to {DLLM2REC_DATA_DIR}")
