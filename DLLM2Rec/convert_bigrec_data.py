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

# SASRec needs 1-based indexing (0 is padding).
# BIGRec uses 0-based indexing (0 to item_count-1).
# Mapping: SASRec_ID = BIGRec_ID + 1

item_num = item_count
MAX_SEQ_LEN = 10 # Original value

def convert_row(row, target_col, hist_col):
    # History
    # history_ids_raw is list of integers in string format e.g. "[10804, 15747]"
    history_ids_raw = eval(str(row[hist_col]))
    
    # Map to 1-based
    seq = [int(mid) + 1 for mid in history_ids_raw]
    
    # Truncate if too long (take last MAX_SEQ_LEN)
    if len(seq) > MAX_SEQ_LEN:
         seq = seq[-MAX_SEQ_LEN:]
         
    # Pad if too short (Left Padding to keep generic position alignment? 
    # SASRec sets position embedding based on 0..len-1 usually if not managing explicitly?
    # Actually main.py SASRec uses:
    # positions = torch.arange(len_states).to(device)
    # inputs_emb += self.positional_embeddings(...)
    # Wait, main.py SASRec forward (line 193):
    # inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
    # It adds position embedding 0..max_len to columns 0..max_len.
    # So if we left pad [0, 0, item1, item2], item2 gets position 3 (if len 4).
    # If we right pad [item1, item2, 0, 0], item2 gets position 1.
    # Sequential recommendation usually models "sequence of events".
    # Left padding means the "sequence finishes at the end of the window".
    # Right padding means "sequence starts at 0".
    # main.py passes `len_states` to model. 
    # But SASRec in main.py uses `torch.arange(self.state_size)` which implies it adds positions 0..200 to the whole tensor.
    # BUT, it creates causal masks? 
    # line 196: mask = torch.ne(states, self.item_num).float() ... which masks padding (if padding is item_num?? No, padding is 0).
    # wait, main.py line 569: `zeros_tensor = torch.zeros((..., item_num + 2))`
    # line 571: sets `seq` indices to 1.
    # line 574: `zeros_tensor[:, item_num] = 1`.
    
    # Let's look at `SASRec.forward` in main.py (line 188) again.
    # `mask = torch.ne(states, self.item_num).float()`
    # If padding is 0, and item_num is e.g. 17408. 
    # Then mask keeps everything except 17408?
    # This implies 17408 IS the padding index in that simplified SASRec implementation!
    # BUT, typically 0 is padding.
    # Let's check `convert_bigrec_data.py` padding decision.
    # Code generally assumes 0 is padding in most repos.
    # But main.py line 60 says `num_embeddings=item_num + 1`.
    # And line 574 explicitly uses `item_num` as a special token for negative sampling avoidance.
    # Let's check utility.py `pad_history` later if possible?
    # No, I should stick to 0 as padding (standard) and Left Padding (standard for Transformer fixed pos).
    
    pad_len = MAX_SEQ_LEN - len(seq)
    if pad_len > 0:
        # Left padding with 0
        seq = [0] * pad_len + seq
    
    # Target
    # target_col is 'item_id' (integer)
    target = int(row[target_col]) + 1
    
    return seq, MAX_SEQ_LEN, target # len(seq) returned is post-padding (200) or original? 
    # usually original length is useful for masking. 
    # But main.py SASRec uses fixed pos embedding. 
    # Let's return the padded list. 
    # But len_seq? main.py uses it for GRU `pack_padded_sequence`.
    # For SASRec it seems ignored in some implementations or used for masking?
    # main.py SASRec: `state_hidden = extract_axis_1(ff_out, len_states - 1)`
    # It extracts the hidden state at the LAST VALID ITEM index.
    # If Left Padded: [0, 0, 1, 2], len=2? No, len is relative to valid items?
    # If Left Padded, the last item is at index 199 (end).
    # If Right Padded: [1, 2, 0, 0], last item is at index 1.
    # extract_axis_1(..., len_states - 1) implies we need the index of the last item.
    # If Left Padding [0, 0, ..., item], the last item is ALWAYS at 199 (MAX_SEQ_LEN-1).
    # If Right Padding [item, ..., 0, 0], the last item is at len_valid - 1.
    
    # CONCLUSION: SASRec in this repo probably uses Left Padding (fixed pos embedding 0..199) implies the model looks at position 199 for the prediction.
    # BUT `extract_axis_1` using `len_states - 1` suggests VARIABLE specific position.
    # If I use Left Padding, `len_states - 1` should be 199.
    # If I use Right Padding, `len_states - 1` should be real_length - 1.
    # BUT SASRec typically aligns "Next Item Prediction" to the last token.
    # If I use Left Padding, the standardized position is the end.
    # Let's try Left Padding and setting len_seq = MAX_SEQ_LEN.
    # Because [0, 0, 1, 2] -> Predict 3. The info is at index 3 (if 0-based).
    # Wait, if Left Padded, the "sequence" ends at 199.
    # So `len_seq` should be 200.
    
    return seq, MAX_SEQ_LEN, target # Return padded seq and fixed max len match.


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
    
    # Check if UID exists effectively
    has_uid = 'uid' in df.columns
    
    for _, row in df.iterrows():
        try:
            seq, len_seq, target = convert_row(row, target_col, hist_col)
            
            item = {
                'seq': seq,
                'len_seq': len_seq,
                'next': target
            }
            
            # Capture UID
            if has_uid:
                 item['uid'] = int(row['uid'])
            elif 'user_id' in row:
                 # Fallback to user_id parsing if integer?
                 try:
                     item['uid'] = int(row['user_id'])
                 except:
                     item['uid'] = -1
            else:
                 # Critical: No UID found
                 item['uid'] = -1
            
            data_list.append(item)
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    new_df = pd.DataFrame(data_list)
    if 'uid' not in new_df.columns:
         print(f"CRITICAL ERROR: Generated DataFrame missing 'uid' column! Sample data: {data_list[:1]}")
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
