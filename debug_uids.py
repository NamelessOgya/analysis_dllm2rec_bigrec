import torch
import json
import os

print("--- Debugging UIDs ---")

# 1. Check train_uids.pt
pt_path = 'experiments/game_bigrec/sasrec/seed_0/alpha_0.5/train_uids.pt'
if os.path.exists(pt_path):
    try:
        uids_pt = torch.load(pt_path)
        if isinstance(uids_pt, torch.Tensor):
            uids_pt = uids_pt.tolist()
        print(f"PT Path: {pt_path}")
        print(f"PT UIDs Count: {len(uids_pt)}")
        print(f"PT UIDs Sample: {uids_pt[:10]}")
        print(f"PT UIDs Type: {type(uids_pt[0])}")
    except Exception as e:
        print(f"PT Error: {e}")
else:
    print(f"PT Path not found: {pt_path}")

# 2. Check train.json
json_path = 'BIGRec/data/game_bigrec/train.json'
if os.path.exists(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        json_uids_sample = [x.get('meta', {}).get('uid', -1) for x in data[:10]]
        all_json_uids = set(x.get('meta', {}).get('uid', -1) for x in data)
        
        print(f"JSON Path: {json_path}")
        print(f"JSON Total Items: {len(data)}")
        print(f"JSON Unique UIDs: {len(all_json_uids)}")
        print(f"JSON UIDs Sample: {json_uids_sample}")
        print(f"JSON UIDs Type: {type(json_uids_sample[0])}")
        
    except Exception as e:
        print(f"JSON Error: {e}")
else:
    print(f"JSON Path not found: {json_path}")
