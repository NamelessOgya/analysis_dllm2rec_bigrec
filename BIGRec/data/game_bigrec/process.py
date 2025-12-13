import json
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import os
import csv
import pickle

# Set random seed
random.seed(42)
np.random.seed(42)

def load_data():
    # Read from ../game_v2 directory to reuse downloads
    meta_path = '../game_v2/meta_Video_Games.json'
    reviews_path = '../game_v2/Video_Games_5.json'
    
    print(f"Loading metadata from {meta_path}...")
    with open(meta_path) as f:
        metadata = [json.loads(line) for line in f]
    
    print(f"Loading reviews from {reviews_path}...")
    with open(reviews_path) as f:
        reviews = [json.loads(line) for line in f]
        
    return metadata, reviews

def filter_data(metadata, reviews):
    print("Filtering data...")
    users = set()
    items = set()
    for review in tqdm(reviews):
        users.add(review['reviewerID'])
        items.add(review['asin'])
    
    item2id = dict()
    count = 0
    for item in items:
        item2id[item] = count
        count += 1
    
    print(f"Users: {len(users)}, Items: {len(items)}, Reviews: {len(reviews)}")
    
    id_title = {}
    id_item = {}
    cnt = 0
    for meta in tqdm(metadata):
        if 'title' in meta and len(meta['title']) > 1:
            id_title[meta['asin']] = meta['title']
            
    processed_users = dict()
    for review in tqdm(reviews):
        user = review['reviewerID']
        if 'asin' not in review:
            continue
        item = review['asin']
        if item not in id_title:
            continue
        if review['asin'] not in id_item:
            id_item[review['asin']] = cnt
            cnt += 1
        if 'overall' not in review:
            continue
        if 'unixReviewTime' not in review:
            continue
            
        if user not in processed_users:
            processed_users[user] = {
                'items': [],
                'ratings': [],
                'timestamps': [],
                'reviews': []
            }
        processed_users[user]['items'].append(item)
        processed_users[user]['ratings'].append(review['overall'])
        processed_users[user]['timestamps'].append(review['unixReviewTime'])
        
    return processed_users, item2id, id_title

def create_interactions(users, item2id, id_title):
    print("Creating interactions...")
    interactions = []
    
    for key in tqdm(users.keys()):
        items = users[key]['items']
        ratings = users[key]['ratings']
        timestamps = users[key]['timestamps']
        
        all_interactions = list(zip(items, ratings, timestamps))
        res = sorted(all_interactions, key=lambda x: int(x[-1]))
        
        items, ratings, timestamps = zip(*res)
        items, ratings, timestamps = list(items), list(ratings), list(timestamps)
        
        users[key]['items'] = items
        users[key]['item_ids'] = [item2id[x] for x in items]
        users[key]['item_titles'] = [id_title[x] for x in items]
        users[key]['ratings'] = ratings
        users[key]['timestamps'] = timestamps
        
        for i in range(min(10, len(items) - 1), len(items)):
            st = max(i - 10, 0)
            interactions.append([
                key, 
                users[key]['items'][st: i], 
                users[key]['items'][i], 
                users[key]['item_ids'][st: i], 
                users[key]['item_ids'][i], 
                users[key]['item_titles'][st: i], 
                users[key]['item_titles'][i], 
                ratings[st: i], 
                ratings[i], 
                int(timestamps[i])
            ])

    return interactions

def save_csv(interactions):
    print("Saving CSV files...")
    interactions = sorted(interactions, key=lambda x: x[-1])
    
    # Add unique ID to each interaction
    # Current structure: [key, items, target, item_ids, target_id, item_titles, target_title, ratings, rating, timestamp]
    # We will append uid to the end.
    interactions_with_uid = []
    for uid, row in enumerate(interactions):
        interactions_with_uid.append(row + [uid])
    interactions = interactions_with_uid

    header = ['user_id', 'item_asins', 'item_asin', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_rating', 'rating', 'timestamp', 'uid']
    
    train_data = interactions[:int(len(interactions) * 0.8)]
    valid_data = interactions[int(len(interactions) * 0.8):int(len(interactions) * 0.9)]
    test_data = interactions[int(len(interactions) * 0.9):]
    
    with open('./train.csv', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(header)
        csvwriter.writerows(train_data)
        
    with open('./valid.csv', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(header)
        csvwriter.writerows(valid_data)
        
    with open('./test.csv', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(header)
        csvwriter.writerows(test_data)
        
    return train_data, valid_data, test_data

def save_dllm2rec_data(train, valid, test, item_num):
    print("Generating DLLM2Rec data...")
    
    def process_split(data):
        # data is list of [key, history_items, target_item, history_ids, target_id, ..., timestamp, uid]
        # index 3: history_ids (0-based list)
        # index 4: target_id (0-based int)
        # index 10: uid
        processed = []
        for row in data:
            history_ids = row[3]
            target_id = row[4]
            uid = row[10]
            
            # 0-based indexing for DLLM2Rec (Aligning with BIGRec)
            # Items 0..item_num-1. Padding = item_num
            seq = [x for x in history_ids] 
            next_item = target_id
            len_seq = len(seq)
            
            # Pad sequence to length 10
            # SASRec uses item_num as padding index (if configured correctly)
            pad_token = item_num 
            if len_seq < 10:
                seq = seq + [pad_token] * (10 - len_seq)
            else:
                seq = seq[-10:]
            
            processed.append({
                'seq': seq,
                'len_seq': len_seq,
                'next': next_item,
                'uid': uid
            })
        return pd.DataFrame(processed)

    train_df = process_split(train)
    valid_df = process_split(valid)
    test_df = process_split(test)
    
    # Save train as pickle
    train_df.to_pickle('./train_data.df')
    
    # Save val/test as CSV
    # Note: Lists in CSVs need to be stringified to match typical pandas to_csv behavior for objects,
    # or just saving normally works but they become string representations "[1, 2, 3]".
    # The inspection showed they are strings in the CSV.
    valid_df.to_csv('./val_data.csv', index=False)
    test_df.to_csv('./test_data.csv', index=False)
    
    # Save statistics
    # seq_size is window size (10)
    # item_num is max_id + 1 (since 0 is padding)
    statis = pd.DataFrame([{'seq_size': 10, 'item_num': item_num + 1}])
    statis.to_pickle('./data_statis.df')
    
    print("Saved DLLM2Rec data to .")

def csv_to_json(input_path, output_path, sample=False):
    print(f"Converting {input_path} to {output_path}...")
    data = pd.read_csv(input_path)
    if sample:
        sample_size = min(5000, len(data))
        data = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
        # data.to_csv(output_path.replace('.json', '.csv'), index=False) 
        
    json_list = []
    for index, row in tqdm(data.iterrows(), total=len(data)):
        try:
            history_titles = eval(row['history_item_title'])
        except:
            continue
            
        L = len(history_titles)
        history = "The user has played the following video games before:"
        for i in range(L):
            if i == 0:
                history += "\"" + history_titles[i] + "\""
            else:
                history += ", \"" + history_titles[i] + "\""
        
        target_item = str(row['item_title'])
        target_item_str = "\"" + target_item + "\""
        uid = int(row['uid'])
        
        json_list.append({
            "instruction": "Given a list of video games the user has played before, please recommend a new video game that the user likes to the user.",
            "input": f"{history}\n ",
            "output": target_item_str,
            "meta": {
                "uid": uid
            }
        })        
        
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)

def main():
    metadata, reviews = load_data()
    users, item2id, id_title = filter_data(metadata, reviews)
    
    # Save id2name for BIGRec (0-based)
    # The eval script expects "Title\tID" ?? 
    # Let's check BIGRec logic. In process.py I converted id2name.txt using:
    # f.write(f"{title}\t{item2id[asin]}\n")
    # This matches the evaluate.py logic if evaluated with this mapping.
    # evaluate.py logic: item_dict = dict(zip(item_names, item_ids))
    # where item_names are lines split by '\t'[0]. item_ids are range(len).
    # So the order in id2name.txt matters! It must be sorted by ID 0, 1, 2...
    
    # Let's sort item2id by ID to ensure line N corresponds to ID N
    sorted_items = sorted(item2id.items(), key=lambda x: x[1])
    
    with open('id2name.txt', 'w') as f:
        for asin, iid in sorted_items:
            # We need the Title.
            # If asin is not in id_title, we filtered it out?
            # Actually filter_data checks `if item not in id_title: continue` for reviews.
            # But item2id was built from ALL items in reviews.
            # So some items in item2id might not have titles?
            # If so, they won't be in interactions.
            # But they will be in id2name.txt if we just dump item2id.
            # If they are not in interactions, they won't be targeted?
            
            # evaluate.py loads id2name.txt and creates ID list 0..N.
            # If our item2id goes 0..M, and some are unused, it's fine as long as used IDs match.
            
            # BUT, we should use the title if available.
            title = id_title.get(asin, asin) # Fallback to ASIN if title missing but somehow kept?
            f.write(f"{title}\t{iid}\n")
                
    interactions = create_interactions(users, item2id, id_title)
    train_data, valid_data, test_data = save_csv(interactions)
    
    csv_to_json('./train.csv', './train.json')
    csv_to_json('./valid.csv', './valid.json')
    csv_to_json('./test.csv', './test.json')
    csv_to_json('./valid.csv', './valid_5000.json', sample=True)
    csv_to_json('./test.csv', './test_5000.json', sample=True)
    
    # Generate DLLM2Rec data
    save_dllm2rec_data(train_data, valid_data, test_data, len(item2id))

if __name__ == '__main__':
    main()
