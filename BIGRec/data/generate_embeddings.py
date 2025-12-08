import argparse
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate item embeddings for BIGRec evaluation")
    parser.add_argument("--dataset", type=str, required=True, choices=["movie", "game", "game_bigrec"], help="Dataset name")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for embeddings")
    parser.add_argument("--use_embedding_model", action="store_true", help="Use dedicated embedding model (e.g. E5)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()

def main():
    args = parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(script_dir, "..", "..")) # Add root project dir 
    from BIGRec.data.utils import get_embedding_model, generate_embeddings
    
    # Determine paths based on dataset
    if args.dataset == "movie":
        input_file = os.path.join(script_dir, "movie", "movies.dat")
    elif args.dataset == "game":
        input_file = os.path.join(script_dir, "game", "id2name.txt")
    elif args.dataset == "game_bigrec":
        input_file = os.path.join(script_dir, "game_bigrec", "id2name.txt")
    
    output_file = args.output_path
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    print(f"Generating embeddings for {args.dataset} dataset...")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Base model: {args.base_model}")
    print(f"Use embedding model: {args.use_embedding_model}")
    
    # Load items
    items = []
    if args.dataset == "movie":
        with open(input_file, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()
            # Format: MovieID::Title::Genres
            items = [line.split('::')[1].strip("\"") for line in lines]
    elif args.dataset in ["game", "game_bigrec"]:
        with open(input_file, 'r') as f:
            lines = f.readlines()
            # Format: Title\tID
            items = [line.split('\t')[0].strip("\"\n").strip(" ") for line in lines]
            
    print(f"Loaded {len(items)} items.")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = get_embedding_model(args.base_model, args.use_embedding_model)
    
    # Generate embeddings
    item_embeddings = generate_embeddings(
        items, 
        model, 
        tokenizer, 
        batch_size=args.batch_size, 
        use_embedding_model=args.use_embedding_model
    )
    
    print(f"Generated embeddings shape: {item_embeddings.shape}")
    
    # Save embeddings
    torch.save(item_embeddings, output_file)
    print(f"Saved embeddings to {output_file}")

if __name__ == "__main__":
    main()
