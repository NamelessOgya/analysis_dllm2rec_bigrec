import pandas as pd
import os

path = 'BIGRec/data/game_bigrec/test.csv'
print(f"Reading {path}...")
df = pd.read_csv(path)
print("Columns:", df.columns.tolist())
print("First row keys:", df.iloc[0].keys().tolist())
if 'uid' in df.columns:
    print("'uid' found in columns.")
else:
    print("'uid' NOT found in columns.")
    # Check for whitespace matches
    for col in df.columns:
        if 'uid' in col:
            print(f"Found partial match: '{col}'")
