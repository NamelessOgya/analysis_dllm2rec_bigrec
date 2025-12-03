import pandas as pd
import numpy as np
import os
import ast
import csv

# Paths
BIGREC_DATA_DIR = 'BIGRec/data/movie'
DLLM2REC_DATA_DIR = 'DLLM2Rec/data/movie'

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
# history_movie_id is a string representation of a list of strings/ints
# movie_id is an int
all_movie_ids = set()

def extract_ids(df):
    for _, row in df.iterrows():
        # Target movie
        all_movie_ids.add(str(row['movie_id']))
        
        # History movies
        history_ids = eval(row['history_movie_id']) # eval because it's "['1', '2']"
        for mid in history_ids:
            all_movie_ids.add(str(mid))

print("Extracting IDs from Train...")
extract_ids(train_df)
print("Extracting IDs from Valid...")
extract_ids(valid_df)
print("Extracting IDs from Test...")
extract_ids(test_df)

# Create mapping: Original ID (str) -> Mapped ID (int, 1-based)
# DLLM2Rec seems to use 1-based indexing for items, 0 might be padding or unused.
# Let's check utility.py or similar if possible, but usually 1-based is safe for SASRec.
sorted_ids = sorted(list(all_movie_ids))
id_map = {original_id: i + 1 for i, original_id in enumerate(sorted_ids)}
item_num = len(id_map)

print(f"Total items: {item_num}")

def convert_row(row):
    # History
    history_ids_raw = eval(row['history_movie_id'])
    seq = [id_map[str(mid)] for mid in history_ids_raw]
    
    # Target
    target = id_map[str(row['movie_id'])]
    
    return seq, len(seq), target

def process_df(df, is_train=False):
    data_list = []
    for _, row in df.iterrows():
        seq, len_seq, target = convert_row(row)
        data_list.append({
            'seq': seq,
            'len_seq': len_seq,
            'next': target
        })
    
    new_df = pd.DataFrame(data_list)
    return new_df

print("Processing Train...")
train_converted = process_df(train_df, is_train=True)
# Save Train as pickle
train_converted.to_pickle(os.path.join(DLLM2REC_DATA_DIR, 'train_data.df'))

print("Processing Valid...")
valid_converted = process_df(valid_df)
# Save Valid as CSV
# Note: DLLM2Rec's val_data.csv has 'seq' as a string representation of list, 'len_seq', 'next'
valid_converted.to_csv(os.path.join(DLLM2REC_DATA_DIR, 'val_data.csv'), index=False)

print("Processing Test...")
test_converted = process_df(test_df)
# Save Test as CSV
test_converted.to_csv(os.path.join(DLLM2REC_DATA_DIR, 'test_data.csv'), index=False)

# Create data_statis.df
# It contains 'seq_size' and 'item_num'
# seq_size seems to be the max sequence length or the window size.
# In BIGRec process.py, seq_len = 10.
seq_size = 10 
statis_data = {
    'seq_size': [seq_size],
    'item_num': [item_num]
}
statis_df = pd.DataFrame(statis_data)
statis_df.to_pickle(os.path.join(DLLM2REC_DATA_DIR, 'data_statis.df'))

print("Conversion complete.")
print(f"Files saved to {DLLM2REC_DATA_DIR}")
