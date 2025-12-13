import pandas as pd
import numpy as np
import os


# Paths
TRAIN_DF_PATH = "BIGRec/data/game_bigrec/train_data.df"
RANK_TXT_PATH = "BIGRec/data/game_bigrec/train_rank.txt" # Assuming evaluate.py output name
# If rank file path is different, I need to know. 
# User didn't specify exact output path, but evaluate.py saves to `{input_file_basename}_rank.txt`
# If input was `train.json`, output is `train_rank.txt`.

def verify():
    if not os.path.exists(TRAIN_DF_PATH):
        print(f"Error: {TRAIN_DF_PATH} not found.")
        return
    
    print(f"Loading {TRAIN_DF_PATH}...")
    train_df = pd.read_pickle(TRAIN_DF_PATH)
    
    # Check max ID in train_df
    # seq is list of IDs. next is target ID.
    max_seq_id = 0
    max_next_id = 0
    
    # Sampling for speed
    print("Sampling 10000 rows...")
    sample = train_df.sample(min(10000, len(train_df)))
    
    for _, row in sample.iterrows():
        seq = row['seq']
        if isinstance(seq, str): seq = eval(seq)
        if len(seq) > 0:
            max_seq_id = max(max_seq_id, max(seq))
        max_next_id = max(max_next_id, row['next'])
        
    print(f"Train DF Sample - Max Seq ID: {max_seq_id}, Max Next ID: {max_next_id}")
    
    item_num_files = "BIGRec/data/game_bigrec/data_statis.df"
    if os.path.exists(item_num_files):
        statis = pd.read_pickle(item_num_files)
        item_num = statis['item_num'][0]
        print(f"Statis DF - Item Num: {item_num}")
        print(f"Expected ID Range: 0 to {item_num - 1}")
        print(f"Padding Token: {item_num}")
    
    # Check Rank File
    if not os.path.exists(RANK_TXT_PATH):
        print(f"Warning: {RANK_TXT_PATH} not found. Checking for 'myrank_train.txt' or similar.")
        # Try finding it
        found = False
        for f in os.listdir("BIGRec/data/game_bigrec"):
            if "rank" in f and "train" in f:
                RANK_TXT_PATH_FOUND = os.path.join("BIGRec/data/game_bigrec", f)
                print(f"Found {RANK_TXT_PATH_FOUND}")
                RANK_TXT_PATH_GLOBAL = RANK_TXT_PATH_FOUND
                found = True
                break
        if not found:
             print("No rank file found. Skipping rank verification.")
             return
    else:
        RANK_TXT_PATH_GLOBAL = RANK_TXT_PATH

    print(f"Loading {RANK_TXT_PATH_GLOBAL} (first 1000 lines)...")
    try:
        # Load partial
        with open(RANK_TXT_PATH_GLOBAL, 'r') as f:
            lines = [f.readline() for _ in range(1000)]
            
        vals = []
        for line in lines:
            if not line: continue
            row = list(map(float, line.strip().split())) # float to handle simple parsing
            vals.extend(row)
            
        min_rank_id = min(vals)
        max_rank_id = max(vals)
        
        print(f"Rank File Sample - Min ID: {min_rank_id}, Max ID: {max_rank_id}")
        
        if min_rank_id >= 1 and (max_rank_id > item_num - 1 if 'item_num' in locals() else True):
            print("CONCLUSION: Rank file appears to be 1-based (Min >= 1).")
            if min_rank_id == 1:
                print("Likely 1-based indexing confirmed.")
        elif min_rank_id == 0:
            print("CONCLUSION: Rank file appears to be 0-based.")
            
        if 'item_num' in locals():
            if max_rank_id > item_num:
                 print(f"WARNING: Max Rank ID {max_rank_id} exceeds Item Num {item_num}. Potential Mismatch + 1.")
            elif max_rank_id == item_num:
                 print(f"WARNING: Max Rank ID {max_rank_id} equals Item Num {item_num}. If items are 0..N-1, this is out of bounds (Padding?).")
        
    except Exception as e:
        print(f"Error verifies rank file: {e}")

def inspect_ci_scores():
    # Looking for .pt files in results
    print("\n--- Inspecting CI Scores (.pt files) ---")
    base_dir = "DLLM2Rec/results"
    
    # Walk and find .pt files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".pt") and ("train" in file or "val" in file or "test" in file):
                 full_path = os.path.join(root, file)
                 try:
                     # Use torch if available in environment (but we removed it earlier).
                     # We need torch to load pt.
                     # If I cannot load torch, I cannot inspect dimensions easily.
                     # I will try to re-import torch inside this scope if it works, else skip.
                     import torch
                     data = torch.load(full_path, map_location='cpu')
                     if isinstance(data, torch.Tensor):
                         print(f"File: {full_path}")
                         print(f"  Shape: {data.shape}")
                         if len(data.shape) == 2:
                             print(f"  Rows: {data.shape[0]}, Items: {data.shape[1]}")
                             if data.shape[1] == 2048:
                                 print("  !! FOUND THE 2048 DIMENSION SUSPECT !!")
                     else:
                         print(f"File: {full_path} (Type: {type(data)}) - Not a Tensor")
                         
                 except ImportError:
                     print("Torch not found, cannot inspect .pt files.")
                     return
                 except Exception as e:
                     print(f"Error loading {full_path}: {e}")

if __name__ == "__main__":
    verify()
    inspect_ci_scores()

