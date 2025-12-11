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

def convert_to_bigrec_format(df, output_path, id2title, split_type='train', start_id=0):
    print(f"Converting {split_type} to {output_path} with start_id={start_id}...")
    json_list = []
    
    # train_df has 'seq' (list of ints) and 'next' (int)
    # val/test csv has 'seq' (string representation of list) and 'next' (int)
    
    current_id = start_id
    
    for index, row in df.iterrows():
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
            "id": current_id,
            "instruction": "Given a list of video games the user has played before, please recommend a new video game that the user likes to the user.",
            "input": f"{history_str}\n ",
            "output": target_str,
            "meta": {
                "uid": current_id
            }
        })
        current_id += 1
        
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)
    
    return current_id

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
    
    # Use continuous IDs across splits
    # train -> valid -> test
    print("Generating train.json...")
    next_id = convert_to_bigrec_format(train_df, 'train.json', id2title, 'train', start_id=1)
    
    print(f"Generating valid.json starting from id {next_id}...")
    next_id = convert_to_bigrec_format(val_df, 'valid.json', id2title, 'valid', start_id=next_id)
    
    print(f"Generating test.json starting from id {next_id}...")
    next_id = convert_to_bigrec_format(test_df, 'test.json', id2title, 'test', start_id=next_id)
    
    # Sample for 5000 (preserving their original ID is tricky if we want them to be a subset of valid/test)
    # BUT the request asks for "small dataset creation".
    # If we regenerate validation/test sample separately, their IDs will conflict if we start from 1.
    # If we want them to be "subsets", we should sample from the generated JSONs?
    # Or just assign new disjoint IDs for these "small" datasets?
    # Usually "valid_5000" is a subset of valid.
    # The requirement says: "game_bigrec形式のデータに対して、簡易的な検証を行えるようなsmall dataset作成をお願いします。"
    # And: "verify ID consistency".
    # So ideally, small dataset should have IDs that DO NOT overlap if used together, OR correspond to the full set?
    # If it's a separate "small dataset", it probably counts as a separate "run".
    # BUT, to test the "consistency", maybe we just want subsamples.
    # Let's create train_5000 as well.
    # And let's keep IDs consistent with the full set if possible?
    # Actually, `convert_to_bigrec_format` currently iterates and assigns fresh IDs.
    # If we sample `val_df` before passing to it, it gets new IDs.
    # If we want the valid_5000 to be VALID SUBSET of `valid.json`, we should load `valid.json` and sample it.
    # However, existing code was generating `valid_5000` from `val_df.sample`.
    # Let's assume for `train_5000`, `valid_5000`, `test_5000`, we can just assign them fresh IDs starting from 1 or something,
    # OR we treat them as a "small dataset suite" that is internally consistent.
    # Let's create a "small dataset suite" with disjoint IDs: train_small -> valid_small -> test_small.
    
    print("Generating small dataset suite (train_5000, valid_5000, test_5000)...")
    
    train_sample = train_df.sample(n=min(5000, len(train_df)), random_state=42)
    val_sample = val_df.sample(n=min(5000, len(val_df)), random_state=42)
    test_sample = test_df.sample(n=min(5000, len(test_df)), random_state=42)
    
    # Reset IDs for the small suite? Or keep them continuous?
    # If we run verify_ids only on small suite, they must be unique.
    # Let's make them unique within the suite.
    small_start_id = 1
    small_start_id = convert_to_bigrec_format(train_sample, 'train_5000.json', id2title, 'train_sample', start_id=small_start_id)
    small_start_id = convert_to_bigrec_format(val_sample, 'valid_5000.json', id2title, 'valid_sample', start_id=small_start_id)
    small_start_id = convert_to_bigrec_format(test_sample, 'test_5000.json', id2title, 'test_sample', start_id=small_start_id)

    # Save sampled DataFrames for consistency with game_bigrec
    print("Saving small dataset DataFrames...")
    train_sample.to_pickle('train_data_5000.df')
    val_sample.to_csv('val_data_5000.csv', index=False)
    test_sample.to_csv('test_data_5000.csv', index=False)
if __name__ == '__main__':
    main()
