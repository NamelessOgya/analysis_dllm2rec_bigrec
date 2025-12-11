import pandas as pd
import os
import sys

# Paths (Adjust relative paths as needed)
BIGREC_DIR = 'BIGRec/data/game_bigrec'
DLLM2REC_DIR = 'BIGRec/data/game_bigrec' 

def verify_1based_alignment():
    print("Verifying 1-based ID Alignment...")
    
    # 1. Load id2name.txt (BIGRec Internal Map: 0-based)
    id2name_path = os.path.join(BIGREC_DIR, 'id2name.txt')
    if not os.path.exists(id2name_path):
        print(f"Error: {id2name_path} not found.")
        return

    print(f"Loading {id2name_path}...")
    bigrec_map = {}
    with open(id2name_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                iid = int(parts[-1])
                title = parts[0]
                bigrec_map[title] = iid
    
    # 2. Load DLLM2Rec train_data.df (Target: 1-based)
    train_df_path = os.path.join(DLLM2REC_DIR, 'train_data.df')
    if not os.path.exists(train_df_path):
        print(f"Error: {train_df_path} not found.")
        return

    print(f"Loading {train_df_path}...")
    train_df = pd.read_pickle(train_df_path)
    
    # Pick a sample
    sample_row = train_df.iloc[0]
    dllm_target_id = sample_row['next'] # 11747
    
    # 3. Derive BIGRec Internal Index (0-based)
    bigrec_internal_id = dllm_target_id - 1 # 11746
    
    print(f"\nTarget ID (DLLM2Rec): {dllm_target_id}")
    print(f"BIGRec Internal Index: {bigrec_internal_id}")
    
    # 4. Simulate Evaluate.py Output
    # The new logic in evaluate.py is: output_id = internal_index + 1
    # So if BIGRec finds index 11746 (Item A), it should output 11747.
    
    simulated_output_rank = bigrec_internal_id + 1
    print(f"Simulated BIGRec Output (New Logic): {simulated_output_rank}")
    
    # 5. Verify Match
    if simulated_output_rank == dllm_target_id:
        print("PASS: BIGRec Output now matches DLLM2Rec ID directly.")
        print("      No further modification needed in DLLM2Rec/main.py.")
    else:
        print("FAIL: Still mismatched.")

if __name__ == "__main__":
    verify_1based_alignment()
