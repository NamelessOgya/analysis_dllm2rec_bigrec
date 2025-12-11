import json
import os
import argparse
from tqdm import tqdm

def fix_uids(source_path, target_path, generate=False):
    if generate:
        print(f"Generating sequential IDs for {target_path}...")
        with open(target_path, 'r') as f:
            target_data = json.load(f)
            
        fixed_count = 0
        for i, tgt_item in tqdm(enumerate(target_data)):
            if 'meta' not in tgt_item:
                tgt_item['meta'] = {}
            
            # Assign sequential ID
            tgt_item['meta']['uid'] = i
            fixed_count += 1
            
        print(f"Generated IDs for {fixed_count} items.")
        
        print(f"Saving fixed data to {target_path}...")
        with open(target_path, 'w') as f:
            json.dump(target_data, f, indent=4)
        print("Done.")
        return True

    print(f"Loading source data from {source_path}...")
    with open(source_path, 'r') as f:
        source_data = json.load(f)
        
    print(f"Loading target data from {target_path}...")
    with open(target_path, 'r') as f:
        target_data = json.load(f)
        
    if len(source_data) != len(target_data):
        print(f"ERROR: Length mismatch! Source: {len(source_data)}, Target: {len(target_data)}")
        return False
        
    print("Injecting UIDs...")
    fixed_count = 0
    for i, (src_item, tgt_item) in tqdm(enumerate(zip(source_data, target_data)), total=len(source_data)):
        if 'meta' in src_item and 'uid' in src_item['meta']:
            uid = src_item['meta']['uid']
            
            if 'meta' not in tgt_item:
                tgt_item['meta'] = {}
            
            # Check if we are overwriting or adding
            if 'uid' not in tgt_item['meta'] or tgt_item['meta']['uid'] == -1:
                tgt_item['meta']['uid'] = uid
                fixed_count += 1
            elif tgt_item['meta']['uid'] != uid:
                print(f"WARNING: ID mismatch at index {i}. Source: {uid}, Target: {tgt_item['meta']['uid']}")
        else:
            print(f"WARNING: Source item at index {i} missing UID meta.")

    print(f"Fixed {fixed_count} items.")
    
    print(f"Saving fixed data to {target_path}...")
    with open(target_path, 'w') as f:
        json.dump(target_data, f, indent=4)
        
    print("Done.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="Path to source JSON with correct UIDs (optional if --generate is used)")
    parser.add_argument("--target", type=str, required=True, help="Path to target JSON to fix (e.g., BIGRec/results/.../train.json)")
    parser.add_argument("--generate", action="store_true", help="Generate sequential IDs instead of copying from source")
    args = parser.parse_args()
    
    if not args.generate and not args.source:
        parser.error("--source is required unless --generate is set")
        
    fix_uids(args.source, args.target, args.generate)
