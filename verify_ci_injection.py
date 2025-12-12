import torch
import json
import os
import subprocess
import sys

# Setup dummy environment
os.makedirs("dummy_ci", exist_ok=True)

# 1. Create dummy SASRec scores
# Assuming N items = count in id2name.txt or just random large number
# test_data_5000.json has 5000 items. Let's use that count or just match what evaluate expects.
# The code expects shape [N_test, N_items].
# Let's peek at valid_5000.json size
with open("BIGRec/data/game/valid_5000.json") as f:
    valid_data = json.load(f)
N_valid = len(valid_data)

with open("BIGRec/data/game/test_5000.json") as f:
    test_data = json.load(f)
N_test = len(test_data)

# Items? id2name.txt
with open("BIGRec/data/game/id2name.txt") as f:
    lines = f.readlines()
N_items = len(lines)

print(f"Creating dummy scores for Valid: {N_valid}, Test: {N_test}, Items: {N_items}")

val_pt = torch.randn(N_valid, N_items)
test_pt = torch.randn(N_test, N_items)

torch.save(val_pt, "dummy_ci/val.pt")
torch.save(test_pt, "dummy_ci/test.pt")

print("Saved dummy val.pt and test.pt")

# 2. Run evaluate.py
# We use test_5000.json and valid_5000.json
# We assume item_embedding.pt exists or let evaluate generate it (it might take time)
# We need to make sure we don't break things.
# Use --save_results to check output.

cmd = [
    "python", "BIGRec/data/game_bigrec/evaluate.py",
    "--input_file", "BIGRec/data/game/test_5000.json",
    "--validation_file", "BIGRec/data/game/valid_5000.json",
    "--ci_score_path", "dummy_ci",
    "--enable_ci_tuning",
    "--embedding_path", "BIGRec/data/game/item_embedding.pt", # reuse existing if possible
    "--batch_size", "8" # small batch for speed
]

print(f"Running command: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)

if result.returncode == 0:
    print("SUCCESS: Evaluation ran with CI injection.")
    if "Best CI Gamma" in result.stdout:
        print("Found CI Gamma tuning output.")
    else:
        print("WARNING: CI Gamma tuning output missing.")
else:
    print("FAILURE: Evaluation failed.")
