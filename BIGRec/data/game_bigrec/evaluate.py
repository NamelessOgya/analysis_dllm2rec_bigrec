from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import os
import math
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse
parse = argparse.ArgumentParser()
parse.add_argument("--input_dir",type=str, default="./", help="your model directory")
parse.add_argument("--base_model", type=str, default="Qwen/Qwen2-0.5B", help="base model path")
parse.add_argument("--embedding_path", type=str, default=None, help="path to item embeddings")
parse.add_argument("--save_results", action="store_true", help="save ranking results to txt files")
parse.add_argument("--topk", type=int, default=200, help="topk for saving results")
parse.add_argument("--batch_size", type=int, default=16, help="batch size for embedding generation")
parse.add_argument("--input_file", type=str, default=None, help="specific input file to process")
args = parse.parse_args()

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
print(f"DEBUG: Loading tokenizer from {base_model}...")
tokenizer = AutoTokenizer.from_pretrained(base_model)
print("DEBUG: Tokenizer loaded. Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("DEBUG: Model loaded.")


script_dir = os.path.dirname(os.path.abspath(__file__))
f = open(os.path.join(script_dir, 'id2name.txt'), 'r')
items = f.readlines()
item_names = [_.split('\t')[0].strip("\"\n").strip(" ") for _ in items]
item_ids = [_ for _ in range(len(item_names))]
item_dict = dict(zip(item_names, item_ids))
print("DEBUG: Items loaded.")
import pandas as pd


result_dict = dict()
for p in path:
    print(f"DEBUG: Processing file {p}...")
    result_dict[p] = {
        "NDCG": [],
        "HR": [],
    }
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    print("DEBUG: Setting model to eval mode...")
    model.eval()
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
    tokenizer.padding_side = "left"

    def batch(list, batch_size=1):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]
    predict_embeddings = []
    from tqdm import tqdm
    print("DEBUG: Starting embedding generation loop...")
    batch_size = args.batch_size
    total_batches = (len(text) - 1) // batch_size + 1
    for i, batch_input in tqdm(enumerate(batch(text, batch_size)), total=total_batches):
        input = tokenizer(batch_input, return_tensors="pt", padding=True)
        input_ids = input.input_ids.to(model.device)
        attention_mask = input.attention_mask.to(model.device)
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        predict_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
    
    predict_embeddings = torch.cat(predict_embeddings, dim=0)
    
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

    # Prepare for saving results
    base_name = os.path.splitext(p)[0]
    rank_file = f"{base_name}_rank.txt"
    score_file = f"{base_name}_score.txt"
    
    f_rank = None
    f_score = None
    if args.save_results:
        f_rank = open(rank_file, 'w')
        f_score = open(score_file, 'w')
        print(f"DEBUG: Saving ranking results incrementally to {rank_file} and {score_file}...")

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
        
        # Incremental Ranking Saving
        if args.save_results:
            # We need top-k indices and values for saving
            # And also we might need full rank for metrics? 
            # Actually metrics calc below iterates items.
            # Usually metrics need the rank position of the ground truth.
            # Computing full argsort for huge items is still heavy [B, N_items].
            # But [1024, 17000] is fine.
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
            topk_indices_cpu = topk_indices.cpu().numpy()
            topk_values_cpu = topk_values.cpu().numpy()
            
            for row in topk_indices_cpu:
                line = ' '.join(map(str, row.tolist()))
                f_rank.write(line + '\n')
                
            for row in topk_values_cpu:
                line = ' '.join(map(str, row.tolist()))
                f_score.write(line + '\n')

        # Compute Metrics
        # batch_rank is on GPU. move to CPU to avoid item access overhead? 
        # or verify if item access on GPU is fast enough.
        # Actually random access like `rank[i][target_id]` on GPU tensor from CPU scalar index is slow.
        # Better move batch_rank to CPU.
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

f = open('./game_bigrec.json', 'w')    
json.dump(result_dict, f, indent=4)