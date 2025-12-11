
f = open('ratings.dat', 'r')
data = f.readlines()
f = open('movies.dat', 'r', encoding='ISO-8859-1')
movies = f.readlines()
movie_names = [_.split('::')[1] for _ in movies]
movie_ids = [_.split('::')[0] for _ in movies]
movie_dict = dict(zip(movie_ids, movie_names))
id_mapping = dict(zip(movie_ids, range(len(movie_ids))))

interaction_dicts = dict()
for line in data:
    user_id, movie_id, rating, timestamp = line.split('::')
    if user_id not in interaction_dicts:
        interaction_dicts[user_id] = {
            'movie_id': [],
            'rating': [],
            'timestamp': [],
            'movie_title': [],
        }
    
    # Fix: Check if movie_id exists in movie_dict
    if movie_id in movie_dict:
        interaction_dicts[user_id]['movie_id'].append(movie_id)
        interaction_dicts[user_id]['rating'].append(int(float(rating) > 3.0))
        interaction_dicts[user_id]['timestamp'].append(timestamp)
        interaction_dicts[user_id]['movie_title'].append(movie_dict[movie_id])

with open('all.csv', 'w') as f:
    import csv
    writer = csv.writer(f)
    writer.writerow(['user_id', 'item_id', 'rating', 'timestamp', 'item_title'])
    for user_id, user_dict in interaction_dicts.items():
        writer.writerow([user_id, user_dict['movie_id'], user_dict['rating'], user_dict['timestamp'], user_dict['movie_title']])

sequential_interaction_list = []
seq_len = 10
for user_id in interaction_dicts:
    temp = zip(interaction_dicts[user_id]['movie_id'], interaction_dicts[user_id]['rating'], interaction_dicts[user_id]['timestamp'], interaction_dicts[user_id]['movie_title'])
    temp = sorted(temp, key=lambda x: int(x[2]))
    result = zip(*temp)
    # Handle case where user has no valid interactions
    if not result:
        continue
        
    interaction_dicts[user_id]['movie_id'], interaction_dicts[user_id]['rating'], interaction_dicts[user_id]['timestamp'], interaction_dicts[user_id]['movie_title'] = [list(_) for _ in result]
    for i in range(10, len(interaction_dicts[user_id]['movie_id'])):
        sequential_interaction_list.append(
            [user_id, interaction_dicts[user_id]['movie_title'][i - seq_len: i],interaction_dicts[user_id]['movie_id'][i-seq_len:i], interaction_dicts[user_id]['rating'][i-seq_len:i], interaction_dicts[user_id]['movie_id'][i], interaction_dicts[user_id]['rating'][i], interaction_dicts[user_id]['timestamp'][i].strip('\n')]
        )
print(len(sequential_interaction_list))

import csv
# Add IDs to the list
# We want continuous IDs across train -> valid -> test
# The list is sorted by timestamp (or at least partially sorted)
# The split happens by index.
# So if we assign IDs to the sorted list, they will be sequential in the splits.
# BUT verify the split logic: list[:0.8], list[0.8:0.9], list[0.9:]
# So yes, user ID 1 will be in train, ID 2 in train... ID N in test.
# Wait, list items are interactions? No, `sequential_interaction_list` has one entry per user?
# Line 36: `sequential_interaction_list` seems to store sequences.
# Line 47: loop `for i in range(10, len(...))`. This generates multiple sequences per user.
# So yes, each item is a sequence sample.

# Assign 'id' to each item in sequential_interaction_list
# We want continuous IDs across train -> valid -> test
# The list is sorted by timestamp (or at least partially sorted)

# Sort BEFORE adding ID to ensure deterministic order
sequential_interaction_list = sorted(sequential_interaction_list, key=lambda x: int(x[-1]))

# Now assign IDs based on sorted order
for idx, item in enumerate(sequential_interaction_list):
    item.append(idx + 1) # ID is now the last element

# Headers need to be updated.
headers = ['user_id', 'history_movie_title', 'history_movie_id', 'history_rating', 'movie_id', 'rating', 'timestamp', 'id']

with open('./train.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(sequential_interaction_list[:int(len(sequential_interaction_list)*0.8)])
with open('./valid.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(sequential_interaction_list[int(len(sequential_interaction_list)*0.8):int(len(sequential_interaction_list)*0.9)])
with open('./test.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(sequential_interaction_list[int(len(sequential_interaction_list)*0.9):])

import json
import pandas as pd
import random
import numpy as np
def csv_to_json(input_path, output_path, sample=False, sample_n=5000):
    data = pd.read_csv(input_path)
    if sample:
        data = data.sample(n=min(sample_n, len(data)), random_state=42).reset_index(drop=True)
        # For sample, we keep the original ID?
        # Requirement: "consistency". If we create a small dataset, it should be consistent.
        # So we keep the ID from the full dataset.
        data.to_csv(output_path.replace(".json", ".csv"), index=False)
        
    
    # Optimize: avoid iterrows
    # Pre-convert columns to lists for faster iteration
    history_movie_ids = data['history_movie_id'].tolist()
    history_movie_titles = data['history_movie_title'].tolist()
    movie_ids = data['movie_id'].tolist()
    ids = data['id'].tolist()
    
    json_list = []
    
    # Use tqdm if available for progress
    try:
        from tqdm import tqdm
        iterator = tqdm(zip(history_movie_ids, history_movie_titles, movie_ids, ids), total=len(data))
    except ImportError:
        iterator = zip(history_movie_ids, history_movie_titles, movie_ids, ids)
        
    for h_m_ids_str, h_m_titles_str, m_id, generated_id in iterator:
        # Strict eval is slow, but necessary for these stringified lists?
        # Maybe manual parsing if format is simple?
        # They are just "['title1', 'title2']". 
        # eval is likely the bottleneck too.
        # But let's stick to eval for correctness as titles may contain commas etc.
        # But we can try to optimize if needed.
        # For now, just removing iterrows should give 10-50x speedup.
        
        row_history_movie_id = eval(h_m_ids_str)
        row_history_movie_title = eval(h_m_titles_str)
        
        L = len(row_history_movie_id)
        history = "The user has watched the following movies before:"
        for i in range(L):
            if i == 0:
                history += "\"" + row_history_movie_title[i] + "\""
            else:
                history += ", \"" + str(row_history_movie_title[i]) + "\""
                
        target_movie_name = "\"" + movie_dict[str(m_id)] + "\""
        
        item_obj = {
            "id": int(generated_id),
            "instruction": "Given a list of movies the user has watched before, please recommend a new movie that the user likes to the user.",
            "input": f"{history}\n ",
            "output": target_movie_name,
        }
        json_list.append(item_obj)    
        
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)

csv_to_json('./train.csv', './train.json')
csv_to_json('./valid.csv', './valid.json')
csv_to_json('./test.csv', './test.json')
csv_to_json('./train.csv', './train_5000.json', sample=True)
csv_to_json('./valid.csv', './valid_5000.json', sample=True)
csv_to_json('./test.csv', './test_5000.json', sample=True)
