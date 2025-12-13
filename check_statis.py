import pandas as pd
import os

paths = [
    "DLLM2Rec/data/game_bigrec/data_statis.df",
    "DLLM2Rec/data/game/data_statis.df"
]

for p in paths:
    if os.path.exists(p):
        print(f"--- {p} ---")
        try:
            df = pd.read_pickle(p)
            print(df)
            item_num = df['item_num'][0]
            print(f"item_num: {item_num}")
            if item_num == 2048:
                 print("!!! MATCH FOUND: 2048 !!!")
        except Exception as e:
            print(f"Error reading {p}: {e}")
    else:
        print(f"--- {p} NOT FOUND ---")
