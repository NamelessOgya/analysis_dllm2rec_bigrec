
import pandas as pd
import sys

path = 'BIGRec/data/game_bigrec/train_data.df'
print(f"Checking {path}...")

try:
    df = pd.read_pickle(path)
except FileNotFoundError:
    print(f"File not found: {path}")
    sys.exit(1)

print(f"Loaded {len(df)} rows.")

lengths = df['seq'].apply(len)
mismatch = lengths != 10
mismatch_count = mismatch.sum()

if mismatch_count > 0:
    print(f"ERROR: Found {mismatch_count} sequences with length != 10!")
    print("Example mismatch:")
    print(df[mismatch].head())
    print("Lengths:", lengths[mismatch].head())
else:
    print("SUCCESS: All sequences are length 10.")
    
# Also check if it's running on the right file copy
dllm_path = 'DLLM2Rec/data/game_bigrec/train_data.df'
print(f"\nChecking copy at {dllm_path}...")
try:
    df2 = pd.read_pickle(dllm_path)
    lengths2 = df2['seq'].apply(len)
    if (lengths2 != 10).sum() == 0:
         print("SUCCESS: Copy is also correct.")
    else:
         print(f"ERROR: Copy has {(lengths2 != 10).sum()} faulty sequences!")
except FileNotFoundError:
    print("Copy not found.")
