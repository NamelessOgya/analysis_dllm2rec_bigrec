from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
import os
import math
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
from tqdm import tqdm
parse = argparse.ArgumentParser()
parse.add_argument("--input_dir",type=str, default="./", help="your model directory")
parse.add_argument("--base_model", type=str, default="Qwen/Qwen2-0.5B", help="base model path")
parse.add_argument("--embedding_path", type=str, default=None, help="path to item embeddings")
parse.add_argument("--save_results", action="store_true", help="save ranking results to txt files")
parse.add_argument("--topk", type=int, default=200, help="topk for saving results")
parse.add_argument("--batch_size", type=int, default=16, help="batch size for embedding generation")
parse.add_argument("--use_embedding_model", action="store_true", help="Use dedicated embedding model (e.g. E5)")
args = parse.parse_args()
print(f"DEBUG: evaluate.py started")
print(f"DEBUG: Arguments: {args}")
print(f"DEBUG: Input Directory: {args.input_dir}")

path = []
for root, dirs, files in os.walk(args.input_dir):
    for name in files:
        print(f"DEBUG: Checking file: {name}")
        if name.endswith(".json") and "metrics.json" not in name:
            print(f"DEBUG: Adding file to process: {name}")
            if "test" in name or "valid" in name or "train" in name:
                path.append(os.path.join(args.input_dir, name))
print(f"DEBUG: Files to process: {path}")

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
f = open(os.path.join(script_dir, 'movies.dat'), 'r', encoding='ISO-8859-1')
movies = f.readlines()
movie_names = [_.split('::')[1].strip("\"") for _ in movies]
movie_ids = [_ for _ in range(len(movie_names))]
movie_dict = dict(zip(movie_names, movie_ids))
print("DEBUG: Movies loaded.")
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
    text = [_["predict"][0].strip("\"") for _ in test_data]
    
    print("DEBUG: Generating prediction embeddings...")
    predict_embeddings = generate_embeddings(
        text, 
        model, 
        tokenizer, 
        batch_size=args.batch_size, 
        use_embedding_model=args.use_embedding_model
    ).cuda()
    if args.embedding_path:
        embedding_file = args.embedding_path
    else:
        embedding_file = os.path.join(script_dir, "movie_embedding.pt")
    
    print(f"DEBUG: Loading item embeddings from {embedding_file}...")
    movie_embedding = torch.load(embedding_file).cuda()
    dist = torch.cdist(predict_embeddings, movie_embedding, p=2)
        
    
    rank = dist
    rank = rank.argsort(dim = -1).argsort(dim = -1)

    topk_list = [1, 3, 5, 10, 20, 50]
    NDCG = []
    for topk in topk_list:
        S = 0
        for i in range(len(test_data)):
            target_movie = test_data[i]['output'].strip("\"")
            target_movie_id = movie_dict[target_movie]
            rankId = rank[i][target_movie_id].item()
            if rankId < topk:
                S = S + (1 / math.log(rankId + 2))
        NDCG.append(S / len(test_data) / (1 / math.log(2)))
    HR = []
    for topk in topk_list:
        S = 0
        for i in range(len(test_data)):
            target_movie = test_data[i]['output'].strip("\"")
            target_movie_id = movie_dict[target_movie]
            rankId = rank[i][target_movie_id].item()
            if rankId < topk:
                S = S + 1
        HR.append(S / len(test_data))
    print(NDCG)
    print(HR)
    print('_' * 100)
    result_dict[p]["NDCG"] = NDCG
    result_dict[p]["HR"] = HR

    print(f"DEBUG: save_results flag is: {args.save_results}")
    if args.save_results:
        print(f"DEBUG: Saving ranking results (top {args.topk}) to files...")
        # Save rank (indices)
        # rank is [num_users, num_items], sorted by distance (ascending? check dist calculation)
        # dist is L2 distance. Smaller is better.
        # rank = dist.argsort().argsort() -> this gives the rank (0-based) of each item.
        # We want the actual item IDs of the top-K items.
        # Wait, rank variable in code is: rank = dist.argsort(dim=-1).argsort(dim=-1)
        # This means rank[i][j] is the rank of item j for user i.
        # To get top-K item IDs, we need dist.argsort(dim=-1)[:, :topk]
        
        # Re-calculate sort to get indices
        # dist is [num_users, num_items]
        sorted_indices = dist.argsort(dim=-1) # [num_users, num_items], values are item indices
        topk_indices = sorted_indices[:, :args.topk] # [num_users, topk]
        
        # Save indices to file
        base_name = os.path.splitext(p)[0]
        rank_file = f"{base_name}_rank.txt"
        with open(rank_file, 'w') as f_rank:
            for row in topk_indices:
                # Convert indices to strings and join
                line = ' '.join(map(str, row.tolist()))
                f_rank.write(line + '\n')
        print(f"Saved rank to {rank_file}")
        
        # Save scores (confidence)
        # We need scores for these top-K items.
        # dist is distance (smaller is better).
        # DLLM2Rec expects "confidence". Usually higher is better?
        # The README says "confidence_train.txt" : refer to "dist".
        # If it refers to "dist", maybe it expects the raw distance?
        # Or maybe converted to similarity?
        # Let's save the raw distance for the top-K items for now, as "dist" implies distance.
        # But usually "confidence" implies probability or similarity.
        # However, since we are using L2 distance, let's stick to what we have or maybe negate it?
        # Given "refer to dist", I will save the values from dist for the top-K items.
        
        # Get values from dist using gather or just indexing
        # dist is on GPU, topk_indices is on GPU
        topk_values = torch.gather(dist, 1, topk_indices)
        
        score_file = f"{base_name}_score.txt"
        with open(score_file, 'w') as f_score:
            for row in topk_values:
                line = ' '.join(map(str, row.tolist()))
                f_score.write(line + '\n')
        print(f"Saved scores to {score_file}")

f = open('./movie.json', 'w')    
json.dump(result_dict, f, indent=4)
