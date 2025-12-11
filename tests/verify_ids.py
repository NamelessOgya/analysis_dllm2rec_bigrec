
import json
import os
import argparse
import sys

def load_ids(path):
    if not os.path.exists(path):
        print(f"File not found: {path} - Skipping")
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
        
    ids = []
    for item in data:
        if 'id' not in item:
            print(f"Error: Item missing 'id' field in {path}")
            # print example
            print(item)
            sys.exit(1)
        ids.append(item['id'])
    return set(ids), len(data)

def verify_dataset(data_dir, suffix=""):
    print(f"\nVerifying dataset in {data_dir} with suffix '{suffix}'...")
    
    files = {
        'train': os.path.join(data_dir, f'train{suffix}.json'),
        'valid': os.path.join(data_dir, f'valid{suffix}.json'),
        'test': os.path.join(data_dir, f'test{suffix}.json')
    }
    
    id_sets = {}
    
    for name, path in files.items():
        res = load_ids(path)
        if res is not None:
            ids, count = res
            print(f"  {name}: {count} items. ID Range: {min(ids)} - {max(ids)}")
            id_sets[name] = ids
            
            # Check for duplicates within file
            if len(ids) != count:
                print(f"  ERROR: Duplicate IDs found in {name}! Unique: {len(ids)}, Total: {count}")
                sys.exit(1)
        else:
            id_sets[name] = set()

    # Check intersections
    splits = list(files.keys())
    has_error = False
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            name1 = splits[i]
            name2 = splits[j]
            s1 = id_sets[name1]
            s2 = id_sets[name2]
            
            intersection = s1.intersection(s2)
            if intersection:
                print(f"  ERROR: Intersection found between {name1} and {name2}! {len(intersection)} common IDs.")
                print(f"  First 5 common: {list(intersection)[:5]}")
                has_error = True
            else:
                print(f"  OK: No intersection between {name1} and {name2}.")
                
    if has_error:
        print("Verification FAILED.")
        sys.exit(1)
    else:
        print("Verification PASSED.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help="Path to BIGRec/data/{dataset} directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Directory not found: {args.dataset_path}")
        sys.exit(1)
        
    # Verify main split
    verify_dataset(args.dataset_path, suffix="")
    
    # Verify small split if exists
    # Check if train_5000.json exists
    if os.path.exists(os.path.join(args.dataset_path, "train_5000.json")):
        verify_dataset(args.dataset_path, suffix="_5000")
    else:
        print("\nSmall dataset (train_5000.json) not found, skipping small dataset verification.")

if __name__ == "__main__":
    main()
