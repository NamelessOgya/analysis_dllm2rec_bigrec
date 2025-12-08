
import json
import argparse
import os
from collections import Counter

def create_pop_file(train_file, output_file):
    print(f"Reading training data from {train_file}...")
    with open(train_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} records.")
    
    # Extract output items (ground truth)
    items = []
    for entry in data:
        if 'output' in entry:
            # Assuming output is the item name, e.g. "Item Name"
            # We strip quotes as done in evaluate.py
            item = entry['output'].strip("\"").strip(" ")
            items.append(item)
    
    # Count frequencies
    counter = Counter(items)
    print(f"Found {len(counter)} unique items.")
    
    # Convert to dictionary
    pop_dict = dict(counter)
    
    # Save to file
    print(f"Saving popularity data to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(pop_dict, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create popularity file from training data")
    parser.add_argument("--train_file", type=str, required=True, help="Path to train.json")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output json file")
    
    args = parser.parse_args()
    
    create_pop_file(args.train_file, args.output_file)
