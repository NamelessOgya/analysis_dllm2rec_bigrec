import torch
import os
import sys

def inspect_file(path):
    print(f"--- Inspecting: {path} ---")
    try:
        data = torch.load(path, map_location='cpu')
        if isinstance(data, torch.Tensor):
            print(f"  Type: Tensor")
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
        else:
            print(f"  Type: {type(data)}")
            if hasattr(data, 'keys'):
                 print(f"  Keys: {list(data.keys())}")
    except Exception as e:
        print(f"  Error loading file: {e}")

def main():
    if len(sys.argv) > 1:
        # Check specific files provided as args
        for path in sys.argv[1:]:
            inspect_file(path)
    else:
        # Recursive search in current directory
        print("No args provided. Searching for .pt files in current directory...")
        for root, dirs, files in os.walk("."):
             for file in files:
                  if file.endswith(".pt") and ("train" in file or "val" in file or "test" in file):
                       inspect_file(os.path.join(root, file))

if __name__ == "__main__":
    main()
