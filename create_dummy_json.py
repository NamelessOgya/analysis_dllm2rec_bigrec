import json
import os

data_dir = 'BIGRec/data/movie'
os.makedirs(data_dir, exist_ok=True)

dummy_data = []
for i in range(20):
    dummy_data.append({
        "instruction": "Given a list of movies the user has watched before, please recommend a new movie that the user likes to the user.",
        "input": "The user has watched the following movies before: \"Toy Story (1995)\", \"GoldenEye (1995)\"\n ",
        "output": "\"Four Rooms (1995)\""
    })

with open(os.path.join(data_dir, 'train.json'), 'w') as f:
    json.dump(dummy_data, f, indent=4)

with open(os.path.join(data_dir, 'valid_5000.json'), 'w') as f:
    json.dump(dummy_data, f, indent=4)

print("Dummy JSONs created.")
