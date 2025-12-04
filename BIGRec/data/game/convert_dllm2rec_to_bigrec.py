import pandas as pd
import json
import os
import pickle
from tqdm import tqdm
import ast

def load_dllm2rec_data(data_dir):
    print(f"Loading DLLM2Rec data from {data_dir}...")
    train_df = pd.read_pickle(os.path.join(data_dir, 'train_data.df'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val_data.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_data.csv'))
    return train_df, val_df, test_df

def load_metadata(meta_path):
    print(f"Loading metadata from {meta_path}...")
    id2title = {}
    # We need to map item IDs (int) to Titles (str).
    # But DLLM2Rec data only has int IDs.
    # We need a mapping from int ID -> Title.
    # Usually there is a map file or we need to infer it from raw data if we have the mapping.
    # The user might have `id2name.txt` or similar?
    # Let's check if `id2name.txt` exists and if it matches the IDs.
    # If not, we might need to load raw data and try to reconstruct, OR just use dummy titles if we can't map.
    # But for BIGRec, titles are crucial for the prompt.
    
    # Let's assume we can load raw meta data and `item2id` mapping if it exists.
    # If `item2id` doesn't exist, we have a problem: we don't know which int ID corresponds to which ASIN/Title.
    
    # Let's check if `item2id` or `map` file exists in DLLM2Rec/data/game.
    # If not, we might have to rely on `BIGRec/data/game/id2name.txt` IF it was generated from the same source/logic.
    # But I just overwrote `id2name.txt` with my previous script!
    # I should check if `id2name.txt` is also tracked by git.
    
    return id2title

def convert_to_bigrec_format(df, output_path, id2title, split_type='train'):
    print(f"Converting {split_type} to {output_path}...")
    json_list = []
    
    # train_df has 'seq' (list of ints) and 'next' (int)
    # val/test csv has 'seq' (string representation of list) and 'next' (int)
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        seq = row['seq']
        target = row['next']
        
        if isinstance(seq, str):
            seq = ast.literal_eval(seq)
            
        # Filter out padding (0)
        seq = [x for x in seq if x != 0]
        
        # Convert IDs to Titles
        # Note: IDs in DLLM2Rec are usually 1-based (0 is padding).
        # We need to handle this.
        
        history_titles = []
        for item_id in seq:
            title = id2title.get(item_id, f"Unknown Game {item_id}")
            history_titles.append(title)
            
        target_title = id2title.get(target, f"Unknown Game {target}")
        
        if not history_titles:
            continue # Skip empty history?
            
        history_str = "The user has played the following video games before:"
        for i, title in enumerate(history_titles):
            if i == 0:
                history_str += "\"" + str(title) + "\""
            else:
                history_str += ", \"" + str(title) + "\""
                
        target_str = "\"" + str(target_title) + "\""
        
        json_list.append({
            "instruction": "Given a list of video games the user has played before, please recommend a new video game that the user likes to the user.",
            "input": f"{history_str}\n ",
            "output": target_str,
        })
        
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)

def main():
    dllm2rec_dir = '../../../DLLM2Rec/data/game'
    bigrec_dir = '.'
    
    train_df, val_df, test_df = load_dllm2rec_data(dllm2rec_dir)
    
    # Load original ID mapping
    id2title = {}
    if os.path.exists('id2name.txt'):
        print("Loading id2name.txt...")
        with open('id2name.txt') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    title = parts[0]
                    # The file format is usually "Title\tID"
                    # BIGRec IDs are 0-based.
                    # DLLM2Rec IDs are 1-based (0 is padding).
                    # So we need to map: (ID in file) + 1 -> Title
                    iid = int(parts[-1])
                    id2title[iid + 1] = title
    else:
        print("Warning: id2name.txt not found! Using dummy titles.")
    
    convert_to_bigrec_format(train_df, 'train.json', id2title, 'train')
    convert_to_bigrec_format(val_df, 'valid.json', id2title, 'valid')
    convert_to_bigrec_format(test_df, 'test.json', id2title, 'test')
    
    # Sample for 5000
    val_sample = val_df.sample(n=min(5000, len(val_df)), random_state=42)
    convert_to_bigrec_format(val_sample, 'valid_5000.json', id2title, 'valid_sample')
    
    test_sample = test_df.sample(n=min(5000, len(test_df)), random_state=42)
    convert_to_bigrec_format(test_sample, 'test_5000.json', id2title, 'test_sample')

if __name__ == '__main__':
    main()
