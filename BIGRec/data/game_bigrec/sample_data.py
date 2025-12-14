import argparse
import json
import pandas as pd
import torch
import numpy as np
import random
import os
from tqdm import tqdm
from collections import Counter
import sys

# Attempt to import sklearn, handle if missing
try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None

def parse_args():
    parser = argparse.ArgumentParser(description="Active Learning Sampling for BIGRec")
    parser.add_argument('--input_json', type=str, required=True, help='Path to BIGRec train.json')
    parser.add_argument('--input_df', type=str, required=True, help='Path to DLLM2Rec train_data.df')
    parser.add_argument('--dros_score', type=str, default=None, help='Path to DROS train.pt (Logits)')
    parser.add_argument('--dros_uid', type=str, default=None, help='Path to DROS train_uids.pt')
    parser.add_argument('--item_emb', type=str, default=None, help='Path to item embeddings (for clustering)')
    parser.add_argument('--method', type=str, required=True, 
                        choices=['random', 'pop_inverse', 'diversity', 'loss', 'entropy', 'error_rank', 'clustering'],
                        help='Sampling method')
    parser.add_argument('--sample_num', type=int, required=True, help='Number of samples to select')
    parser.add_argument('--al_ratio', type=float, default=1.0, help='Ratio of samples to select via Active Learning (0.0 - 1.0). Remainder is random.')
    parser.add_argument('--output_json', type=str, required=True, help='Output JSON path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for score calculation')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_json(path):
    print(f"Loading JSON from {path}...")
    with open(path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records.")
    # Map UID to entry
    uid2entry = {}
    for entry in data:
        if 'meta' in entry and 'uid' in entry['meta']:
            uid2entry[int(entry['meta']['uid'])] = entry
    return uid2entry

def load_df(path):
    print(f"Loading DataFrame from {path}...")
    df = pd.read_pickle(path)
    print(f"Loaded {len(df)} rows.")
    # Ensure uid is int
    df['uid'] = df['uid'].astype(int)
    # Map UID to row index and target
    uid2idx = {uid: idx for idx, uid in zip(df.index, df['uid'])}
    uid2target = {uid: target for uid, target in zip(df['uid'], df['next'])}
    return df, uid2idx, uid2target

def calculate_entropy(logits):
    # logits: [B, NumClasses]
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def calculate_loss(logits, targets):
    # logits: [B, NumClasses]
    # targets: [B]
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fn(logits, targets)
    return loss

def calculate_rank(logits, targets):
    # logits: [B, NumClasses]
    # targets: [B]
    # Higher logit = Better rank. Sort descending.
    # We want rank of target. 
    # argsort is slow on large items.
    # Instead, count how many items have score > target_score
    target_scores = logits.gather(1, targets.unsqueeze(1)) # [B, 1]
    # Optimization: If ItemNum is huge, this comparison is heavy.
    # But usually ItemNum ~ 20k. [1024, 20000] compare is doable.
    ranks = (logits > target_scores).sum(dim=1) + 1
    return ranks

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # 1. Load Data
    uid2entry = load_json(args.input_json)
    df, uid_to_df_idx, uid_to_target = load_df(args.input_df)
    
    # Identify Common UIDs
    common_uids = list(set(uid2entry.keys()) & set(uid_to_df_idx.keys()))
    if not common_uids:
        raise ValueError("No common UIDs found between JSON and DF!")
    print(f"Found {len(common_uids)} common UIDs.")
    
    # Limit to common UIDs
    common_uids.sort() # Ensure deterministic order
    
    scores = {} # uid -> score
    
    # 2. Logic per method
    if args.method in ['loss', 'entropy', 'error_rank']:
        if not args.dros_score or not args.dros_uid:
            raise ValueError(f"Method {args.method} requires --dros_score and --dros_uid")
        
        print(f"Loading DROS scores from {args.dros_score} (CPU)...")
        # Load map_location='cpu' to save memory
        all_logits = torch.load(args.dros_score, map_location='cpu')
        all_dros_uids = torch.load(args.dros_uid, map_location='cpu')
        
        # Create map from UID to Tensor Index
        # all_dros_uids might be tensor
        if isinstance(all_dros_uids, torch.Tensor):
            all_dros_uids = all_dros_uids.tolist()
        
        dros_uid2idx = {uid: i for i, uid in enumerate(all_dros_uids)}
        
        # Filter common_uids that exist in DROS output
        valid_uids = [u for u in common_uids if u in dros_uid2idx]
        print(f"Processing scores for {len(valid_uids)} UIDs in batches of {args.batch_size}...")
        
        # Process in batches
        for i in tqdm(range(0, len(valid_uids), args.batch_size)):
            batch_uids = valid_uids[i : i + args.batch_size]
            
            # Get indices in DROS tensor
            dros_indices = [dros_uid2idx[u] for u in batch_uids]
            
            # Load batch to GPU (or keep CPU if no GPU)
            # We assume CPU processing for safety if memory is tight, or GPU if available?
            # User warned about 3GB memory. If we have GPU, 3GB fits.
            # But let's stick to CPU or minimal GPU usage.
            # Let's use CPU for calculation to be safe unless specified.
            # Actually, torch operations are faster on GPU. If we load 1024 batch, it's small.
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            batch_logits = all_logits[dros_indices].float().to(device) # Ensure float32 for calc
            
            if args.method == 'entropy':
                batch_scores = calculate_entropy(batch_logits)
            elif args.method in ['loss', 'error_rank']:
                # Need targets
                targets = [uid_to_target[u] for u in batch_uids]
                batch_targets = torch.LongTensor(targets).to(device)
                
                if args.method == 'loss':
                    batch_scores = calculate_loss(batch_logits, batch_targets)
                else: # error_rank
                    batch_scores = calculate_rank(batch_logits, batch_targets)
            
            # Move back to CPU and store
            batch_scores = batch_scores.cpu().tolist()
            for u, s in zip(batch_uids, batch_scores):
                scores[u] = s
            
            # Cleanup for memory
            del batch_logits
            if 'batch_targets' in locals(): del batch_targets
            torch.cuda.empty_cache()

    elif args.method == 'clustering':
        if not args.item_emb:
            raise ValueError("Method clustering requires --item_emb")
        if KMeans is None:
            raise ImportError("scikit-learn is required for clustering. Please install it.")
        
        print(f"Loading embeddings from {args.item_emb}...")
        embeddings = torch.load(args.item_emb, map_location='cpu').numpy()
        
        print("Running K-Means...")
        # Number of clusters? Heuristic: sqrt(N)/2 or fixed small number?
        # Proposal: "Semantic Diversity". Maybe 50-100 clusters?
        n_clusters = 50
        kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed).fit(embeddings)
        item_clusters = kmeans.labels_ # [ItemNum]
        
        for u in common_uids:
            target_item = uid_to_target[u]
            if target_item < len(item_clusters):
                scores[u] = item_clusters[target_item] # Store cluster ID
            else:
                scores[u] = -1 # Padding or invalid
                
    elif args.method == 'pop_inverse':
        # Calculate item freq
        all_targets = df['next'].tolist()
        freq = Counter(all_targets)
        
        for u in common_uids:
            target_item = uid_to_target[u]
            f = freq.get(target_item, 0)
            if f > 0:
                scores[u] = 1.0 / f
            else:
                scores[u] = 0.0

    elif args.method == 'random':
        # Just assign random score
        # Even for mixed strategy, if method is random, al_ratio of random + random remainder = random.
        for u in common_uids:
            scores[u] = random.random()

    elif args.method == 'diversity':
        # Coverage based (Greedy) requires special handling in Sampling phase
        # We prepare (uid, target_item) list
        user_items = []
        for u in common_uids:
            user_items.append((u, uid_to_target[u]))
        scores = user_items # Special case
        
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # 3. Sampling
    total_target_size = min(len(common_uids), args.sample_num)
    
    # Calculate Split
    n_al = int(total_target_size * args.al_ratio)
    n_random = total_target_size - n_al
    
    print(f"Sampling {total_target_size} records total.")
    print(f"  - Active Learning ({args.method}): {n_al}")
    print(f"  - Random Fill: {n_random}")
    
    selected_uids_al = []
    
    # --- Active Learning Part ---
    if n_al > 0:
        if args.method == 'diversity':
            # Greedy for item coverage
            # Prioritize items not yet covered
            random.shuffle(user_items) # Shuffle for randomness in ties
            
            covered_items = set()
            candidates_new_item = []
            candidates_redundant = []
            
            for u, item in user_items:
                if item not in covered_items:
                    candidates_new_item.append(u)
                    covered_items.add(item)
                else:
                    candidates_redundant.append(u)
            
            # Take all new coverage items first
            selected_uids_al = candidates_new_item[:n_al]
            
            # Fill remaining for AL part
            if len(selected_uids_al) < n_al:
                needed = n_al - len(selected_uids_al)
                selected_uids_al.extend(candidates_redundant[:needed])
                
        elif args.method == 'clustering':
            # scores[u] = cluster_id
            cluster_uids = {}
            for u, cid in scores.items():
                if cid not in cluster_uids:
                    cluster_uids[cid] = []
                cluster_uids[cid].append(u)
            
            all_uids_list = []
            for cid in cluster_uids:
                random.shuffle(cluster_uids[cid])
            
            keys = list(cluster_uids.keys())
            random.shuffle(keys)
            
            while len(selected_uids_al) < n_al and len(keys) > 0:
                for k in list(keys): # Iterate copy
                    if not cluster_uids[k]:
                        keys.remove(k)
                        continue
                    selected_uids_al.append(cluster_uids[k].pop())
                    if len(selected_uids_al) >= n_al:
                        break
                        
        else:
            # Score based (Top-K)
            sorted_uids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            selected_uids_al = sorted_uids[:n_al]
    
    # --- Random Fill Part ---
    selected_set = set(selected_uids_al)
    candidate_random = [u for u in common_uids if u not in selected_set]
    
    if n_random > 0:
        if len(candidate_random) < n_random:
             print("Warning: Not enough candidates for random fill! Taking all.")
             selected_random = candidate_random
        else:
            selected_random = random.sample(candidate_random, n_random)
        selected_uids_al.extend(selected_random)
        
    final_selected_uids = selected_uids_al
    
    print(f"Selected {len(final_selected_uids)} UIDs.")
    if len(final_selected_uids) == 0:
        raise ValueError("No UIDs were selected! Check if DROS scores/UIDs align with input data.")
    
    # 4. Save Output
    output_data = []
    for u in final_selected_uids:
        output_data.append(uid2entry[u])
        
    print(f"Saving to {args.output_json}...")
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print("Done.")

if __name__ == "__main__":
    main()
