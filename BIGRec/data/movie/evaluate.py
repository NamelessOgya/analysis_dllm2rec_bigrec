from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import transformers
import torch
import os
import math
import json
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import argparse
parse = argparse.ArgumentParser()
parse.add_argument("--input_dir",type=str, default="./", help="your model directory")
parse.add_argument("--base_model", type=str, default="Qwen/Qwen2-0.5B", help="base model path")
parse.add_argument("--embedding_path", type=str, default=None, help="path to item embeddings")
parse.add_argument("--save_results", action="store_true", help="save ranking results to txt files")
parse.add_argument("--topk", type=int, default=200, help="topk for saving results")
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
            path.append(os.path.join(args.input_dir, name))
print(f"DEBUG: Files to process: {path}")

base_model = args.base_model
print(f"DEBUG: Loading tokenizer from {base_model}...")
tokenizer = LlamaTokenizer.from_pretrained(base_model)
print("DEBUG: Tokenizer loaded. Loading model...")
model = LlamaForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    device_map="auto",
)
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
    text = [_["predict"][0].strip("\"") for _ in test_data]
    tokenizer.padding_side = "left"

    def batch(list, batch_size=1):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]
    predict_embeddings = []
    from tqdm import tqdm
    print("DEBUG: Starting embedding generation loop...")
    batch_size = 16
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
