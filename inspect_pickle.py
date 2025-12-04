import pandas as pd
import os

path = 'DLLM2Rec/data/game/train_data.df'
try:
    df = pd.read_pickle(path)
    print("Columns:", df.columns)
    print("First row:", df.iloc[0])
    print("Data types:", df.dtypes)
    # Check if there are any other attributes on the df object (unlikely for DataFrame but possible if it's a custom class)
except Exception as e:
    print(f"Error: {e}")
