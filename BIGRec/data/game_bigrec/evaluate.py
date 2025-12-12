from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import os
import math
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
from tqdm import tqdm
parse = argparse.ArgumentParser()
parse.add_argument("--input_dir",type=str, default="./", help="your model directory")
parse.add_argument("--base_model", type=str, default="Qwen/Qwen2-0.5B", help="base model path")
parse.add_argument("--embedding_path", type=str, default=None, help="path to item embeddings")
parse.add_argument("--save_results", action="store_true", help="save ranking results to txt files")
parse.add_argument("--topk", type=int, default=200, help="topk for saving results")
parse.add_argument("--batch_size", type=int, default=16, help="batch size for embedding generation")
parse.add_argument("--input_file", type=str, default=None, help="specific input file to process")
parse.add_argument("--use_embedding_model", action="store_true", help="Use dedicated embedding model (e.g. E5)")
parse.add_argument("--popularity_file", type=str, default=None, help="Path to popularity count json file")
parse.add_argument("--popularity_gamma", type=float, default=0.0, help="Gamma value for popularity adjustment")
parse.add_argument("--validation_file", type=str, default=None, help="Path to validation result json for gamma tuning")
args = parse.parse_args()
print(f"DEBUG: Parsed Arguments: {args}")

path = []
if args.input_file:
    if os.path.exists(args.input_file):
        path.append(args.input_file)
    else:
        print(f"Error: Input file {args.input_file} not found.")
        exit(1)
else:
    for root, dirs, files in os.walk(args.input_dir):
        for name in files:
            if name.endswith(".json") and "metrics.json" not in name:
                if "test" in name or "valid" in name or "train" in name:
                    path.append(os.path.join(args.input_dir, name))
print(path)
base_model = args.base_model
# Import utils
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "..", "..", "..")) # Path to BIGRec root
from BIGRec.data.utils import get_embedding_model, generate_embeddings

print(f"DEBUG: Loading model {base_model} (Use Embedding Model: {args.use_embedding_model})...")
model, tokenizer = get_embedding_model(base_model, args.use_embedding_model)
print("DEBUG: Model loaded.")

script_dir = os.path.dirname(os.path.abspath(__file__))
f = open(os.path.join(script_dir, 'id2name.txt'), 'r')
items = f.readlines()
item_names = [_.split('\t')[0].strip("\"\n").strip(" ") for _ in items]
item_ids = [_ for _ in range(len(item_names))]
item_dict = dict(zip(item_names, item_ids))
print("DEBUG: Items loaded.")
import pandas as pd

# Load Popularity Data Once if needed
pop_rank_origin = None
if args.popularity_file:
    print(f"DEBUG: Loading popularity data from {args.popularity_file}")
    with open(args.popularity_file, 'r') as f_pop:
        pop_count = json.load(f_pop)
    
    num_items = len(item_dict)
    pop_rank_origin = torch.zeros(num_items)
    
    for item_name, count in pop_count.items():
        if item_name in item_dict:
            idx = item_dict[item_name]
            pop_rank_origin[idx] = count
    
    if pop_rank_origin.sum() > 0:
            pop_rank_origin = pop_rank_origin / pop_rank_origin.sum()
            min_val = pop_rank_origin.min()
            max_val = pop_rank_origin.max()
            if max_val - min_val > 0:
                pop_rank_origin = (pop_rank_origin - min_val) / (max_val - min_val)
    
    if torch.cuda.is_available():
            pop_rank_origin = pop_rank_origin.cuda()

# Grid Search Logic
print(f"DEBUG: Checking Grid Search Conditions: validation_file={args.validation_file}, pop_rank_loaded={pop_rank_origin is not None}")
if args.validation_file and pop_rank_origin is not None:
    print(f"DEBUG: Starting Grid Search for Popularity Gamma using {args.validation_file}...")
    if not os.path.exists(args.validation_file):
        print(f"Error: Validation file {args.validation_file} not found.")
        exit(1)
        
    with open(args.validation_file, 'r') as f:
        valid_data = json.load(f)
    
    valid_text = [_["predict"][0].strip("\"") for _ in valid_data]
    print("DEBUG: Generating validation embeddings...")
    valid_embeddings = generate_embeddings(
        valid_text, model, tokenizer, batch_size=args.batch_size, use_embedding_model=args.use_embedding_model
    )
    if torch.cuda.is_available():
         valid_embeddings = valid_embeddings.cuda()
    
    # Load item embeddings (re-load for safety or reuse? reuse logic for simplicity)
    if args.embedding_path:
        embedding_file = args.embedding_path
    else:
        embedding_file = os.path.join(script_dir, "item_embedding.pt")
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    movie_embedding = torch.load(embedding_file, map_location=device)
    if device == "cuda":
        movie_embedding = movie_embedding.cuda()
        
    # Validation Distance
    valid_dist = torch.cdist(valid_embeddings, movie_embedding, p=2)
    
    # Prepare Pop Rank for broadcasting [1, M]
    pop_rank_tensor = pop_rank_origin.unsqueeze(0)
    
    best_gamma = 0.0
    best_ndcg = -1.0
    
    # Search Range: 0.0 to 100.0, step 1.0 (following adjust_pda.py logic of /100 but loop 10000)
    # adjust_pda.py: range(0, 10000, 100) -> 0, 100, 200.. -> /100 -> 0.0, 1.0, 2.0 ... 99.0
    # This seems rough. Maybe 0.1 step is better? The user asked to "find best gamma".
    # I'll stick to 1.0 step for speed as per reference, or maybe 0.5?
    # Let's use 0.0 to 10.0 step 0.1, then 10 to 100 step 1?
    # adjust_pda.py loop: `for g in range(0, 10000, 100): gamma = g/100` -> 0.0, 1.0, 2.0...
    # This implies integer steps for gamma.
    search_gammas = [g/10.0 for g in range(0, 1000, 1)] # 0.0 to 99.9 step 0.1
    # Optimization: Batched evaluation or just simple loop?
    # We do simple loop over gammas.
    
    # Helper to calc NDCG@20 only for speed
    def calc_ndcg_20(rank_indices, data, item_dict):
        S = 0
        count = 0 
        # Only calc average over the set
        # This is slow in pure python.
        # Vectorized implementation?
        # Target IDs
        target_ids = []
        for d in data:
            target_item = d['output'].strip("\"").strip(" ")
            tid = item_dict.get(target_item)
            target_ids.append(tid if tid is not None else -1)
        target_ids = torch.tensor(target_ids, device=rank_indices.device).unsqueeze(1) # [B, 1]
        
        # Get rank of target
        # rank_indices is [B, M] where value is ItemID at that rank?
        # No. dist.argsort.argsort gives Rank (0-based) for each ItemID.
        # rank_val = rank_indices.gather(1, target_ids)
        # But target_ids might have -1.
        
        valid_mask = target_ids != -1
        valid_targets = target_ids[valid_mask.squeeze()]
        if len(valid_targets) == 0: return 0.0
        
        # We need the Rank of the Target Item.
        # rank_indices[b, item_id] = rank
        ranks = torch.gather(rank_indices[valid_mask.squeeze()], 1, valid_targets.unsqueeze(1))
        
        # NDCG@20
        # if rank < 20: 1/log2(rank+2)
        # ranks are 0-based
        hits = (ranks < 20).float()
        dcg = hits * (1.0 / torch.log2(ranks + 2.0))
        # IDCG@20 is 1.0
        return dcg.mean().item()

    print("DEBUG: Searching optimal gamma (0.0 - 1.0, step 0.1)...")
    # Reduced range for safety/speed in this context, 0-100 is huge.
    # Usually around 0-2 is valid.
    search_space = [x/10.0 for x in range(0, 11)] # 0.0 to 1.0 step 0.1
    
    for gamma in search_space:
        # Apply adjustment
        # rank = (1+pop)^(-gamma) * dist
        # Note: In log space? No dist is Euclidean.
        # Optimization: (1+pop)^(-gamma) can be precomputed but gamma changes.
        adj = torch.pow((1 + pop_rank_tensor), -gamma)
        adjusted_dist = dist * adj # Broadcasting
        
        # Get Ranks
        # We only need the Rank of the Target Item
        # rank = dist.argsort().argsort()
        # This is expensive (sorting [B, M]).
        # Can we do faster?
        # For full evaluation we need sort.
        # For 1k items? M is 29k for game.
        # Validation size might be ~1k-5k.
        # Sorting 5k * 30k is heavy.
        
        # Let's keep it simple: Just do it.
        rank_indices = adjusted_dist.argsort(dim=-1).argsort(dim=-1)
        
        score = calc_ndcg_20(rank_indices, valid_data, item_dict)
        # print(f"Gamma: {gamma}, NDCG@20: {score:.4f}")
        
        if score > best_ndcg:
            best_ndcg = score
            best_gamma = gamma
            
    print(f"DEBUG: Best Gamma found: {best_gamma} (NDCG@20: {best_ndcg:.4f})")
    args.popularity_gamma = best_gamma
    
    # Save best gamma to file
    with open('best_gamma.txt', 'w') as f_g:
        f_g.write(str(best_gamma))
    print("DEBUG: Saved best_gamma.txt")


result_dict = dict()
for p in path:
    print(f"DEBUG: Processing file {p}...")
    result_dict[p] = {
        "NDCG": [],
        "HR": [],
    }
    
    print("DEBUG: Reading test data from file...")
    f = open(p, 'r')
    import json
    test_data = json.load(f)
    f.close()
    print(f"DEBUG: Loaded {len(test_data)} items from test data.")
    
    # Validation check
    if not isinstance(test_data, list):
        print(f"WARNING: Skipping {p} because it is not a list (found {type(test_data)}).")
        continue
    if len(test_data) > 0 and "predict" not in test_data[0]:
        print(f"WARNING: Skipping {p} because items do not contain 'predict' key.")
        continue

    text = [_["predict"][0].strip("\"") for _ in test_data]
    
    print("DEBUG: Generating prediction embeddings...")
    predict_embeddings = generate_embeddings(
        text, 
        model, 
        tokenizer, 
        batch_size=args.batch_size, 
        use_embedding_model=args.use_embedding_model
    )
    
    if args.embedding_path:
        embedding_file = args.embedding_path
    else:
        embedding_file = os.path.join(script_dir, "item_embedding.pt")

    print(f"DEBUG: Loading item embeddings from {embedding_file}...")
    # Load to GPU immediately for cdist
    if torch.cuda.is_available():
        device = "cuda" 
    else: 
        device = "cpu"
        
    movie_embedding = torch.load(embedding_file, map_location=device)
    if device == "cuda":
        movie_embedding = movie_embedding.cuda()

    # Prepare Popularity Adjustment Tensor
    pop_adjustment = None
    if pop_rank_origin is not None and args.popularity_gamma > 0:
        print(f"DEBUG: Preparing popularity adjustment with gamma={args.popularity_gamma}")
        try:
             pop_rank_tensor = pop_rank_origin.unsqueeze(0) # [1, M] already handled in loading? No loading gave [M]
             # pop_rank_origin is [M]
             
             # Pre-calculate adjustment factor: (1 + pop)^(-gamma)
             pop_adjustment = torch.pow((1 + pop_rank_origin.unsqueeze(0)), -args.popularity_gamma) # [1, M]
             print("DEBUG: Popularity adjustment tensor prepared.")
            
        except Exception as e:
            print(f"WARNING: Failed to prepare popularity adjustment: {e}")

    # Prepare for saving results
    base_name = os.path.splitext(p)[0]
    rank_file = f"{base_name}_rank.txt"
    score_file = f"{base_name}_score.txt"
    uid_file = f"{base_name}_uid.txt"
    
    f_rank = None
    f_score = None
    f_uid = None
    if args.save_results:
        f_rank = open(rank_file, 'w')
        f_score = open(score_file, 'w')
        f_uid = open(uid_file, 'w')
        print(f"DEBUG: Saving ranking results incrementally to {rank_file}, {score_file}, and {uid_file}...")

    # Evaluation Batch Size (for distance calc)
    eval_batch_size = 1024 
    
    # helper for batching tensor
    def get_batches_tensor(tensor, batch_size):
        for i in range(0, tensor.size(0), batch_size):
            yield tensor[i:i + batch_size], i

    topk_list = [1, 3, 5, 10, 20, 50]
    # Initialize accumulators
    S_list = {k: 0.0 for k in topk_list}
    SS_list = {k: 0.0 for k in topk_list}
    LL = len(test_data)
    
    print(f"DEBUG: Starting batched evaluation (Batch Size: {eval_batch_size})...")
    
    for batch_pred_emb, start_idx in tqdm(get_batches_tensor(predict_embeddings, eval_batch_size), total=(predict_embeddings.size(0) + eval_batch_size - 1) // eval_batch_size):
        # Move batch to GPU
        if device == "cuda":
            batch_pred_emb = batch_pred_emb.cuda()
            
        # Compute distance [B, N_items]
        dist = torch.cdist(batch_pred_emb, movie_embedding, p=2)

        # Apply Popularity Adjustment if enabled
        if pop_adjustment is not None:
             dist = dist * pop_adjustment
        
        # Incremental Ranking Saving
        if args.save_results:
            # We need top-k indices and values for saving
            pass
            
        # Calculate Rank for Metrics
        # rank[b][item_id] gives the rank (0-based) of item_id
        # argsort().argsort() is standard trick. 
        # dist is [B, M]
        batch_rank = dist.argsort(dim=-1).argsort(dim=-1) # [B, M]
        
        # Save results (Top-K)
        if args.save_results:
            sorted_dist_indices = dist.argsort(dim=-1) # [B, M]
            topk_indices = sorted_dist_indices[:, :args.topk] # [B, K]
            topk_values = torch.gather(dist, 1, topk_indices) # [B, K]
            
            # Move to CPU for writing
            topk_indices_cpu = topk_indices.cpu().numpy() + 1 # Convert to 1-based ID
            topk_values_cpu = topk_values.cpu().numpy()
            
            current_batch_size = batch_pred_emb.size(0)
            
            for row in topk_indices_cpu:
                line = ' '.join(map(str, row.tolist()))
                f_rank.write(line + '\n')
                
            for row in topk_values_cpu:
                line = ' '.join(map(str, row.tolist()))
                f_score.write(line + '\n')
                
            # Write UIDs for this batch
            for b in range(current_batch_size):
                global_idx = start_idx + b
                item_data = test_data[global_idx]
                # Extract UID gracefully
                uid = -1
                if 'meta' in item_data and 'uid' in item_data['meta']:
                    uid = item_data['meta']['uid']
                f_uid.write(f"{uid}\n")

        # Compute Metrics
        # batch_rank is on GPU. move to CPU to avoid item access overhead? 
        batch_rank_cpu = batch_rank.cpu()
        
        current_batch_size = batch_rank.size(0)
        for b in range(current_batch_size):
            global_idx = start_idx + b
            target_item = test_data[global_idx]['output'].strip("\"").strip(" ")
            target_id = item_dict.get(target_item)
            
            if target_id is None:
                continue
                
            minID = batch_rank_cpu[b][target_id].item()
            
            for topk in topk_list:
                if minID < topk:
                    S_list[topk] += (1 / math.log(minID + 2))
                    SS_list[topk] += 1

    if f_rank: f_rank.close()
    if f_score: f_score.close()
    if f_uid: f_uid.close()

    NDCG = []
    HR = []
    for topk in topk_list:
        NDCG.append(S_list[topk] / LL / (1.0 / math.log(2)))
        HR.append(SS_list[topk] / LL)

    print(NDCG)
    print(HR)
    print('_' * 100)
    result_dict[p]["NDCG"] = NDCG
    result_dict[p]["HR"] = HR
    
    if args.save_results:
         print(f"Saved rank to {rank_file}")
         print(f"Saved scores to {score_file}")
         print(f"Saved uids to {uid_file}")

f = open('./game_bigrec.json', 'w')    
json.dump(result_dict, f, indent=4)