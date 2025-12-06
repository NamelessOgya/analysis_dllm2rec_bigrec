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
                if "test" in name or "valid" in name:
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
    
    predict_embeddings = torch.cat(predict_embeddings, dim=0).cuda()
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
            target_id = item_dict[target_item]
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