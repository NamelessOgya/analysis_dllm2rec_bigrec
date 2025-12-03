import pandas as pd
import os

data_dir = './DLLM2Rec/data/movie'
statis_path = os.path.join(data_dir, 'data_statis.df')
val_path = os.path.join(data_dir, 'val_data.csv')

print("--- Data Statistics ---")
if os.path.exists(statis_path):
    statis_df = pd.read_pickle(statis_path)
    print(statis_df)
else:
    print("data_statis.df not found")

print("\n--- Validation Data Head ---")
if os.path.exists(val_path):
    val_df = pd.read_csv(val_path)
    print(val_df.head())
else:
    print("val_data.csv not found")
