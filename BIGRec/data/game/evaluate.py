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
parse.add_argument("--use_embedding_model", action="store_true", help="Use dedicated embedding model (e.g. E5)")
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
    ).cuda()
    if args.embedding_path:
        embedding_file = args.embedding_path
    else:
        embedding_file = os.path.join(script_dir, "item_embedding.pt")

    print(f"DEBUG: Loading item embeddings from {embedding_file}...")
    movie_embedding = torch.load(embedding_file).cuda()
    dist = torch.cdist(predict_embeddings, movie_embedding, p=2)

        
    rank = dist
    rank = rank.argsort(dim = -1).argsort(dim = -1)
    topk_list = [1, 3, 5, 10, 20, 50]
    NDCG = []
    HR = []
    for topk in topk_list:
        S = 0
        SS = 0
        LL = len(test_data)
        for i in range(len(test_data)):
            target_item = test_data[i]['output'].strip("\"").strip(" ")
            minID = 20000
            target_id = item_dict.get(target_item)
            if target_id is None:
                continue
            minID = rank[i][target_id].item()
            if minID < topk:
                S= S+ (1 / math.log(minID + 2))
                SS = SS + 1
        temp_NDCG = []
        temp_HR = []
        NDCG.append(S / LL / (1.0 / math.log(2)))
        HR.append(SS / LL)

    print(NDCG)
    print(HR)
    print('_' * 100)
    result_dict[p]["NDCG"] = NDCG
    result_dict[p]["HR"] = HR

    if args.save_results:
        print(f"DEBUG: Saving ranking results (top {args.topk}) to files...")
        sorted_indices = dist.argsort(dim=-1)
        topk_indices = sorted_indices[:, :args.topk]
        
        base_name = os.path.splitext(p)[0]
        rank_file = f"{base_name}_rank.txt"
        with open(rank_file, 'w') as f_rank:
            for row in topk_indices:
                line = ' '.join(map(str, row.tolist()))
                f_rank.write(line + '\n')
        print(f"Saved rank to {rank_file}")
        
        topk_values = torch.gather(dist, 1, topk_indices)
        score_file = f"{base_name}_score.txt"
        with open(score_file, 'w') as f_score:
            for row in topk_values:
                line = ' '.join(map(str, row.tolist()))
                f_score.write(line + '\n')
        print(f"Saved scores to {score_file}")

f = open('./game.json', 'w')    
json.dump(result_dict, f, indent=4)