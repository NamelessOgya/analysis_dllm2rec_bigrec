import pandas as pd
import os

data_dir = './DLLM2Rec/data/game'
train_path = os.path.join(data_dir, 'train_data.df')
val_path = os.path.join(data_dir, 'val_data.csv')
test_path = os.path.join(data_dir, 'test_data.csv')

print("--- Train Data (Pickle) ---")
try:
    train_df = pd.read_pickle(train_path)
    print(train_df.head())
    print(train_df.columns)
except Exception as e:
    print(f"Error reading train data: {e}")

print("\n--- Val Data (CSV) ---")
try:
    val_df = pd.read_csv(val_path)
    print(val_df.head())
    print(val_df.columns)
except Exception as e:
    print(f"Error reading val data: {e}")

print("\n--- Test Data (CSV) ---")
try:
    test_df = pd.read_csv(test_path)
    print(test_df.head())
    print(test_df.columns)
except Exception as e:
    print(f"Error reading test data: {e}")
