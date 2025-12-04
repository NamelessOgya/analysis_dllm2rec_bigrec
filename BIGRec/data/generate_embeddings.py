import argparse
import os
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate item embeddings for BIGRec evaluation")
    parser.add_argument("--dataset", type=str, required=True, choices=["movie", "game"], help="Dataset name")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for embeddings")
    return parser.parse_args()

def batch(list, batch_size=1):
    chunk_size = (len(list) - 1) // batch_size + 1
    for i in range(chunk_size):
        yield list[batch_size * i: batch_size * (i + 1)]

def main():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine paths based on dataset
    if args.dataset == "movie":
        input_file = os.path.join(script_dir, "movie", "movies.dat")
    elif args.dataset == "game":
        input_file = os.path.join(script_dir, "game", "id2name.txt")
    
    output_file = args.output_path
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    print(f"Generating embeddings for {args.dataset} dataset...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Base model: {args.base_model}")
    
    # Load items
    items = []
    if args.dataset == "movie":
        with open(input_file, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()
            # Format: MovieID::Title::Genres
            # We extract Title. strip("\"") removes surrounding quotes if present.
            items = [line.split('::')[1].strip("\"") for line in lines]
    elif args.dataset == "game":
        with open(input_file, 'r') as f:
            lines = f.readlines()
            # Format: Title\tID
            items = [line.split('\t')[0].strip("\"\n").strip(" ") for line in lines]
            
    print(f"Loaded {len(items)} items.")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "left"
    
    model = LlamaForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.eval()
    
    # Generate embeddings
    item_embeddings = []
    print("Starting embedding generation...")
    
    with torch.no_grad():
        for i, batch_input in tqdm(enumerate(batch(items, args.batch_size))):
            input = tokenizer(batch_input, return_tensors="pt", padding=True)
            input_ids = input.input_ids.to(model.device)
            attention_mask = input.attention_mask.to(model.device)
            
            outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            # Use the last hidden state of the last token
            embedding = hidden_states[-1][:, -1, :].detach().cpu()
            item_embeddings.append(embedding)
            
    item_embeddings = torch.cat(item_embeddings, dim=0)
    print(f"Generated embeddings shape: {item_embeddings.shape}")
    
    # Save embeddings
    torch.save(item_embeddings, output_file)
    print(f"Saved embeddings to {output_file}")

if __name__ == "__main__":
    main()
