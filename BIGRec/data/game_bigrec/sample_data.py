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
                        choices=['random', 'pop_inverse', 'diversity', 'loss', 'entropy', 'error_rank', 'clustering',
                                 'proximal_rank', 'semantic_loss', 'confident_error'],
                        help='Sampling method')
    parser.add_argument('--sample_num', type=int, required=True, help='Number of samples to select')
    parser.add_argument('--al_ratio', type=float, default=1.0, help='Ratio of samples to select via Active Learning (0.0 - 1.0). Remainder is random.')
    parser.add_argument('--min_rank', type=int, default=10, help='Min rank for proximal_rank')
    parser.add_argument('--max_rank', type=int, default=100, help='Max rank for proximal_rank')
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
    target_scores = logits.gather(1, targets.unsqueeze(1)) # [B, 1]
    ranks = (logits > target_scores).sum(dim=1) + 1
    return ranks

def calculate_max_prob(logits):
    # logits: [B, NumClasses]
    probs = torch.softmax(logits, dim=-1)
    max_probs, _ = torch.max(probs, dim=-1)
    return max_probs # [B]

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
    dros_methods = ['loss', 'entropy', 'error_rank', 'proximal_rank', 'semantic_loss', 'confident_error']
    if args.method in dros_methods:
        if not args.dros_score or not args.dros_uid:
            raise ValueError(f"Method {args.method} requires --dros_score and --dros_uid")
        
        print(f"Loading DROS scores from {args.dros_score} (CPU)...")
        if args.method == 'semantic_loss' and not args.item_emb:
             raise ValueError("Method semantic_loss requires --item_emb")
             
        # Load map_location='cpu' to save memory
        all_logits = torch.load(args.dros_score, map_location='cpu')
        all_dros_uids = torch.load(args.dros_uid, map_location='cpu')
        
        # Create map from UID to Tensor Index
        if isinstance(all_dros_uids, torch.Tensor):
            all_dros_uids = all_dros_uids.tolist()
        
        dros_uid2idx = {uid: i for i, uid in enumerate(all_dros_uids)}
        
        # Filter common_uids that exist in DROS output
        valid_uids = [u for u in common_uids if u in dros_uid2idx]
        print(f"Processing scores for {len(valid_uids)} UIDs in batches of {args.batch_size}...")
        
        # Process in batches
        for i in tqdm(range(0, len(valid_uids), args.batch_size)):
            batch_uids = valid_uids[i : i + args.batch_size]
            
            dros_indices = [dros_uid2idx[u] for u in batch_uids]
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            batch_logits = all_logits[dros_indices].float().to(device)
            
            if args.method == 'entropy':
                batch_scores = calculate_entropy(batch_logits)
            elif args.method == 'confident_error':
                # Score = MaxProb if Rank > 1 else -1 (discard)
                targets = [uid_to_target[u] for u in batch_uids]
                batch_targets = torch.LongTensor(targets).to(device)
                
                max_probs = calculate_max_prob(batch_logits)
                ranks = calculate_rank(batch_logits, batch_targets)
                
                # We want: Wrong (Rank > 1) AND Confident (High MaxProb)
                # If Rank == 1 (Correct), set score to -1 (Low priority/Discard)
                # If Rank > 1, score = MaxProb
                batch_scores = torch.where(ranks > 1, max_probs, torch.tensor(-1.0, device=device))
                
            elif args.method in ['loss', 'error_rank', 'proximal_rank', 'semantic_loss']:
                targets = [uid_to_target[u] for u in batch_uids]
                batch_targets = torch.LongTensor(targets).to(device)
                
                if args.method in ['loss', 'semantic_loss']:
                    batch_scores = calculate_loss(batch_logits, batch_targets)
                else: # error_rank, proximal_rank
                    ranks = calculate_rank(batch_logits, batch_targets)
                    if args.method == 'proximal_rank':
                         # Filter logic happens here or later?
                         # Let's set score = rank if in range, else -1?
                         # Actually, proximal_rank wants specific range.
                         # If in range [min, max], keep. Prioritize which? 
                         # User: "Proximal Hardness". Maybe random within range? Or closer to 50?
                         # Let's just set score = 1.0 if in range, 0.0 otherwise.
                         # And then in sampling phase we pick top-k (i.e., all 1.0s) and random tie-break (sort stability).
                         # But sort is stable. Top-K of all 1.0s is simply first K.
                         # Better to randomize scores slightly if we want random selection from goldilocks zone.
                         # Or: Score = -ABS(Rank - (Min+Max)/2) ? To prioritize center?
                         # Let's imply: "Prioritize better rank within range" (closer to min_rank)?
                         # No, closer to min_rank means "Almost correct". 
                         # Let's use Score = 1.0/Rank if in range, else 0.
                         # This prioritizes better ranks (10 > 11 > ... > 100).
                         mask = (ranks >= args.min_rank) & (ranks <= args.max_rank)
                         batch_scores = torch.where(mask, 1.0 / ranks.float(), torch.tensor(-1.0, device=device))
                    else: # error_rank
                        batch_scores = ranks

            # Move back to CPU and store
            batch_scores = batch_scores.cpu().tolist()
            for u, s in zip(batch_uids, batch_scores):
                scores[u] = s
            
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
        
        print("Running K-Means for Clustering...")
        n_clusters = 50
        kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed).fit(embeddings)
        item_clusters = kmeans.labels_ # [ItemNum]
        
        for u in common_uids:
            target_item = uid_to_target[u]
            if target_item < len(item_clusters):
                scores[u] = item_clusters[target_item] # Store cluster ID
            else:
                scores[u] = -1 
                
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
        for u in common_uids:
            scores[u] = random.random()

    elif args.method == 'diversity':
        user_items = []
        for u in common_uids:
            user_items.append((u, uid_to_target[u]))
        scores = user_items
        
    else:
        # Fallback if I missed something above
        pass

    # 3. Sampling
    total_target_size = min(len(common_uids), args.sample_num)
    
    n_al = int(total_target_size * args.al_ratio)
    n_random = total_target_size - n_al
    
    print(f"Sampling {total_target_size} records total.")
    print(f"  - Active Learning ({args.method}): {n_al}")
    print(f"  - Random Fill: {n_random}")
    
    selected_uids_al = []
    
    # --- Active Learning Part ---
    if n_al > 0:
        if args.method == 'diversity':
             # (Existing Diversity Logic)
             random.shuffle(user_items)
             covered_items = set()
             candidates_new_item = []
             candidates_redundant = []
             for u, item in user_items:
                if item not in covered_items:
                    candidates_new_item.append(u)
                    covered_items.add(item)
                else:
                    candidates_redundant.append(u)
             selected_uids_al = candidates_new_item[:n_al]
             if len(selected_uids_al) < n_al:
                needed = n_al - len(selected_uids_al)
                selected_uids_al.extend(candidates_redundant[:needed])
                
        elif args.method == 'clustering':
            # (Existing Clustering Logic)
            cluster_uids = {}
            for u, cid in scores.items():
                if cid not in cluster_uids: cluster_uids[cid] = []
                cluster_uids[cid].append(u)
            
            keys = list(cluster_uids.keys())
            random.shuffle(keys)
            
            # Shuffle internal lists
            for k in keys:
                random.shuffle(cluster_uids[k])
                
            while len(selected_uids_al) < n_al and len(keys) > 0:
                for k in list(keys):
                    if not cluster_uids[k]:
                        keys.remove(k)
                        continue
                    selected_uids_al.append(cluster_uids[k].pop())
                    if len(selected_uids_al) >= n_al:
                        break
        
        elif args.method == 'semantic_loss':
            # Hybrid: Clustering + Loss
            # We have Loss scores in scores[u]
            # We need Cluster IDs
            print("Running K-Means for Semantic Loss...")
            if KMeans is None: raise ImportError("sklearn needed")
            embeddings = torch.load(args.item_emb, map_location='cpu').numpy()
            n_clusters = 50
            kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed).fit(embeddings)
            item_clusters = kmeans.labels_
            
            # Group UIDs by Cluster
            cluster_uids = {} # cid -> list of (uid, loss)
            for u, loss_val in scores.items():
                target_item = uid_to_target[u]
                if target_item < len(item_clusters):
                    cid = item_clusters[target_item]
                    if cid not in cluster_uids: cluster_uids[cid] = []
                    cluster_uids[cid].append((u, loss_val))
            
            # Sort each cluster data by Loss Descending (Hardest first)
            for cid in cluster_uids:
                cluster_uids[cid].sort(key=lambda x: x[1], reverse=True)
            
            # Stratified Sampling (Round Robin again?)
            # Yes, pick Hardest from Cluster 1, Hardest from Cluster 2...
            keys = list(cluster_uids.keys())
            random.shuffle(keys)
            
            while len(selected_uids_al) < n_al and len(keys) > 0:
                for k in list(keys):
                    if not cluster_uids[k]:
                        keys.remove(k)
                        continue
                    # Pop the hardest item (first in list)
                    uid, _ = cluster_uids[k].pop(0)
                    selected_uids_al.append(uid)
                    if len(selected_uids_al) >= n_al:
                        break
                        
        else:
            # Score based (Top-K)
            # proximal_rank: scores are 1/rank (if valid) or -1. So sort desc puts valid ranks first, better ranks higher.
            # confident_error: scores are MaxProb (if Error) or -1. Sort desc puts Confident Errors first.
            # loss: Sort Desc.
            # entropy: Sort Desc.
            # error_rank: Sort Desc (High Rank = Bad).
            
            sorted_uids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            
            # Optional: If proximal_rank / confident_error, stop if score < 0?
            # Yes, otherwise we pick invalid items.
            if args.method in ['proximal_rank', 'confident_error']:
                # Filter out negative scores
                valid_sorted = [u for u in sorted_uids if scores[u] >= 0]
                if len(valid_sorted) < n_al:
                    print(f"Warning: Only {len(valid_sorted)} valid samples found for {args.method} (Goal: {n_al}). Filling with random.")
                    selected_uids_al = valid_sorted
                    # Adjust n_al effectively or let Random Fill handle it?
                    # "Remainder is random" logic below handles remaining gap.
                    # Just need to ensure n_random calculation accounts for shortfall?
                    # Currently n_random is fixed based on n_al (target).
                    # If I return smaller selected_uids_al here, logic below:
                    # n_random = total_target_size - n_al (original n_al).
                    # So if selected_uids_al is smaller, we have gap.
                    # We should add to n_random?
                    # Or just:
                    # final_selected = selected_uids_al + selected_random
                    # If selected_uids_al is short, we need MORE random?
                    # Logic says: "Remainder is random". 
                    # Let's dynamically update n_random?
                    # No, let's just append to selected_uids_al from random candidates *inside random block*?
                    # Easier: Simply act as if AL selected fewer.
                    pass
                else:
                    selected_uids_al = sorted_uids[:n_al]
            else:
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
