import pandas as pd
import os
import re

def check_mapping_quality():
    print("Checking ID mapping quality...")
    
    # 1. Load ID Mapping
    id2title = {}
    title2id = {}
    mapping_path = 'BIGRec/data/game/id2name.txt'
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    title = parts[0]
                    try:
                        iid = int(parts[-1])
                        # Adjust for 1-based indexing if necessary, but let's check raw IDs first
                        # In convert script we did: id2title[iid + 1] = title
                        # So the file has 0-based IDs.
                        id2title[iid] = title
                    except ValueError:
                        continue
    else:
        print(f"Error: {mapping_path} not found.")
        return

    print(f"Total mapped items in file: {len(id2title)}")

    # 2. Load Used IDs from DLLM2Rec Data
    data_dir = 'DLLM2Rec/data/game'
    train_df = pd.read_pickle(os.path.join(data_dir, 'train_data.df'))
    
    used_ids = set()
    # train_df has 'seq' (list of ints) and 'next' (int)
    # These are 1-based IDs (0 is padding).
    
    for index, row in train_df.iterrows():
        seq = row['seq']
        if isinstance(seq, list):
            for x in seq:
                if x != 0:
                    used_ids.add(x - 1) # Convert to 0-based to match file
        
        if row['next'] != 0:
            used_ids.add(row['next'] - 1)

    print(f"Total unique items used in Train: {len(used_ids)}")
    
    # 3. Check Coverage
    missing_ids = used_ids - set(id2title.keys())
    print(f"Missing IDs (used but not in map): {len(missing_ids)}")
    if len(missing_ids) > 0:
        print(f"Sample missing IDs: {list(missing_ids)[:5]}")

    # 4. Check Title Quality
    # Criteria for "bad" title:
    # - Looks like ASIN (starts with B0 and length 10, alphanumeric)
    # - Very short (< 3 chars)
    # - "Unknown" or "nan"
    
    asin_pattern = re.compile(r'^B0[0-9A-Z]{8}$')
    
    bad_titles = []
    asin_titles = []
    short_titles = []
    
    for iid in used_ids:
        if iid in id2title:
            title = id2title[iid]
            
            if asin_pattern.match(title):
                asin_titles.append((iid, title))
            elif len(title) < 3:
                short_titles.append((iid, title))
            elif title.lower() == 'nan' or title.lower() == 'unknown':
                bad_titles.append((iid, title))
                
    print("-" * 30)
    print(f"Quality Analysis for {len(used_ids)} used items:")
    print(f"  - ASIN-like titles (e.g., 'B00...'): {len(asin_titles)} ({len(asin_titles)/len(used_ids)*100:.2f}%)")
    print(f"  - Very short titles (< 3 chars): {len(short_titles)} ({len(short_titles)/len(used_ids)*100:.2f}%)")
    print(f"  - 'nan' or 'unknown': {len(bad_titles)}")
    
    if len(asin_titles) > 0:
        print(f"Sample ASIN titles: {[t for i, t in asin_titles[:5]]}")
    if len(short_titles) > 0:
        print(f"Sample short titles: {[t for i, t in short_titles[:5]]}")

if __name__ == "__main__":
    check_mapping_quality()
