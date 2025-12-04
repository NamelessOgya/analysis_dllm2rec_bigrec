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
args = parse.parse_args()

path = []
for root, dirs, files in os.walk(args.input_dir):
    for name in files:
            path.append(os.path.join(args.input_dir, name))
print(path)

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
    for i, batch_input in tqdm(enumerate(batch(text, 16))):
        input = tokenizer(batch_input, return_tensors="pt", padding=True)
        input_ids = input.input_ids.to(model.device)
        attention_mask = input.attention_mask.to(model.device)
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        predict_embeddings.append(hidden_states[-1][:, -1, :].detach().cpu())
    
    predict_embeddings = torch.cat(predict_embeddings, dim=0).cuda()
    movie_embedding = torch.load(os.path.join(script_dir, "movie_embedding.pt")).cuda()
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

f = open('./movie.json', 'w')    
json.dump(result_dict, f, indent=4)
