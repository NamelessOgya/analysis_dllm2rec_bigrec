import json
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
import os
import csv
import pickle

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def load_data():
    print("Loading metadata...")
    with open('meta_Video_Games.json') as f:
        metadata = [eval(line) for line in f]
    
    print("Loading reviews...")
    with open('Video_Games_5.json') as f:
        reviews = [eval(line) for line in f]
        
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
    print(f"Sparsity: {len(reviews) / (len(users) * len(items))}")
    
    id_title = {}
    id_item = {}
    cnt = 0
    for meta in tqdm(metadata):
        # Use title if available, else use ASIN
        if 'title' in meta and len(meta['title']) > 0:
            id_title[meta['asin']] = meta['title']
        else:
            id_title[meta['asin']] = meta['asin'] # Fallback
            
    print(f"Items with titles (or fallback): {len(id_title)}")
            
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
    
    # For DLLM2Rec compatibility
    user_map = {}
    item_map = {} # This should align with item2id if we are careful, but let's be explicit
    
    # We need to ensure item IDs are consistent. 
    # BIGRec uses item2id based on 'asin'.
    # DLLM2Rec needs integer IDs.
    
    # Let's create a consistent mapping
    # Re-map items to 1..N range for DLLM2Rec (0 is usually padding)
    # But BIGRec might use 0-indexed.
    # Let's check BIGRec logic. It uses item2id which is 0-indexed.
    
    # To ensure consistency, we will use the same integer IDs for both.
    
    # Filter users with enough history? The original notebook logic:
    # for i in range(min(10, len(items) - 1), len(items)):
    # This implies users with < 2 items might be skipped or just have no targets?
    # Actually the loop range(min(10, len(items) - 1), len(items)) means:
    # If len=1: range(0, 1) -> i=0. st=0. items[0:0] is empty. 
    # Wait, min(10, 0) is 0. range(0, 1).
    # If len=5: range(4, 5). i=4. st=0. items[0:4] history, items[4] target.
    
    valid_users = {}
    
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
        
        # Logic from notebook
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
            
        valid_users[key] = users[key]

    return interactions, valid_users

def save_csv(interactions):
    print("Saving CSV files...")
    interactions = sorted(interactions, key=lambda x: x[-1])
    
    header = ['user_id', 'item_asins', 'item_asin', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_rating', 'rating', 'timestamp']
    
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

def csv_to_json(input_path, output_path, sample=False):
    print(f"Converting {input_path} to {output_path}...")
    data = pd.read_csv(input_path)
    if sample:
        sample_size = min(5000, len(data))
        data = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
        # Save sampled CSV for reference/debugging if needed, though original code overwrites extension
        # data.to_csv(output_path.replace('.json', '.csv'), index=False) 
        
    json_list = []
    for index, row in tqdm(data.iterrows()):
        try:
            history_titles = eval(row['history_item_title'])
            # history_ratings = eval(row['history_rating']) # Not used in prompt
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
        
        json_list.append({
            "instruction": "Given a list of video games the user has played before, please recommend a new video game that the user likes to the user.",
            "input": f"{history}\n ",
            "output": target_item_str,
        })        
        
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)

def save_dllm2rec_data(users, item2id, id_title):
    print("Generating DLLM2Rec data...")
    # DLLM2Rec expects:
    # train_data.df (pickle): user sequences
    # val_data.csv: user_id, item_id, ...
    # test_data.csv
    
    # We need to map ASINs to 1-based integers for SASRec usually (0 is padding)
    # But BIGRec used 0-based index in item2id.
    # Let's check SASRecModules_ori.py or utility.py for padding.
    # Usually 0 is padding. So we should shift IDs by +1 if they are 0-based.
    
    # Let's create a mapping that is consistent.
    # item2id has all items.
    # We will use (id + 1) as the item ID for SASRec.
    
    # Prepare Train Data (Sequences)
    # Format: DataFrame with 'user_id' and 'item_id_list' (list of ints)
    
    train_rows = []
    val_rows = []
    test_rows = []
    
    # We need to split by user for SASRec usually (leave-one-out or similar)
    # But BIGRec split by interaction timestamp globally.
    # This is a discrepancy. BIGRec's split is global time split (80/10/10).
    # SASRec usually expects user-based split (e.g. last item test).
    
    # However, the user wants to "reproduce BIGRec and DLLM2Rec".
    # DLLM2Rec is supposed to distill BIGRec.
    # So DLLM2Rec should be trained on the same data as BIGRec?
    # Or at least evaluated on the same target?
    
    # If we look at DLLM2Rec/data/game/train_data.df, it likely contains sequences.
    # Let's try to follow the standard SASRec data format but using the data we have.
    
    # We will create a user-centric view for SASRec.
    # But wait, if we split interactions globally, a user might have items in train, val, and test.
    # For SASRec training, we usually feed the full sequence up to the split point.
    
    # Let's construct the sequences for each user based on the global split.
    # Actually, to keep it simple and consistent with the "Transfer" step:
    # The transfer step will generate rankings for the TRAIN set of BIGRec?
    # Or does it generate for the whole dataset?
    
    # Let's look at the goal: "DLLM2Rec uses BIGRec's knowledge".
    # Usually this means we train SASRec on the same training data, 
    # but with soft labels from BIGRec.
    
    # So we need `train_data.df` to contain the training sequences.
    
    # Let's iterate through the `interactions` list again or use `users` dict but filter by split.
    # Since we already saved train.csv, let's use that to define what is "Train".
    
    # Re-reading train.csv is one way, or just using the list `train_data` from save_csv.
    # But `train_data` is a list of interactions.
    # SASRec `train_data.df` typically has one row per user, with a list of items.
    
    # Let's reconstruct user sequences from the global train split.
    train_user_seqs = {}
    
    # We need to map user IDs to integers too for SASRec.
    user2id = {}
    u_cnt = 1 # 1-based user ID?
    
    # Load the splits we just created/defined
    # We can't easily access the local variables from `save_csv` inside here unless passed.
    # Let's assume we do this after `save_csv` and pass the data or re-read.
    # For simplicity, let's process the `users` dict and cut sequences based on timestamps?
    # No, global split is by timestamp across ALL users.
    
    # So, an interaction (u, i, t) is in Train if t < T1.
    # It is in Val if T1 <= t < T2.
    # It is in Test if t >= T2.
    
    # Let's define T1 and T2 from the sorted interactions.
    # But we already did the split by index in `save_csv`.
    
    pass

def generate_dllm2rec_files(train_data, valid_data, test_data, item2id):
    print("Generating DLLM2Rec data...")
    
    # DLLM2Rec expects:
    # train_data.df (pickle): seq (list), len_seq (int), next (int)
    # val_data.csv: seq, len_seq, next
    # test_data.csv: seq, len_seq, next
    # data_statis.df: seq_size, item_num
    
    # BIGRec uses seq_len=10 (from notebook logic: range(min(10, ...)))
    SEQ_LEN = 10
    
    # Map item IDs to 1-based for SASRec (0 is padding)
    # item2id is 0-based. So we add 1.
    ITEM_NUM = len(item2id)
    
    output_dir = '../../../DLLM2Rec/data/game'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def process_split(data_list, is_csv=False):
        rows = []
        for row in data_list:
            # row format from save_csv:
            # [user_id, item_asins, item_asin, history_item_id, item_id, history_item_title, item_title, history_rating, rating, timestamp]
            # history_item_id is a list of ints (0-based)
            # item_id is int (0-based)
            
            history_ids = row[3]
            target_id = row[4]
            
            # Pad history
            # SASRec usually expects fixed length sequences in the dataframe if using torch.LongTensor(list(series))
            # We pad with 0 on the left (or right? utility.py pad_history pads on right? Let's check utility.py)
            # utility.py: pad_history(itemlist, length, pad_item): if len<length: temp=[pad]*...; itemlist.extend(temp) -> Right padding.
            # But SASRec usually uses left padding for history?
            # Let's check main.py again. It uses pack_padded_sequence for GRU.
            # For SASRec, it uses masking.
            # Let's stick to utility.py's pad_history which is right padding.
            # Wait, utility.py is imported in main.py.
            
            # Actually, let's implement padding here to be safe.
            # If utility.py pads on right, we should follow that or change it.
            # Usually sequential models pad on left so the last item is the most recent.
            # But if main.py uses len_seq, it might handle it.
            
            # Let's look at BIGRec's history. It is [st: i].
            # If i < 10, it is short.
            
            seq = [x + 1 for x in history_ids] # 1-based
            target = target_id + 1 # 1-based
            
            len_seq = len(seq)
            
            # Pad to SEQ_LEN
            if len_seq < SEQ_LEN:
                seq = seq + [0] * (SEQ_LEN - len_seq) # Right padding
            else:
                seq = seq[-SEQ_LEN:] # Truncate to last 10
                len_seq = SEQ_LEN # Cap length
                
            rows.append({
                'seq': seq,
                'len_seq': len_seq,
                'next': target
            })
        return rows

    train_rows = process_split(train_data)
    val_rows = process_split(valid_data)
    test_rows = process_split(test_data)
    
    # Save Train (Pickle)
    train_df = pd.DataFrame(train_rows)
    train_df.to_pickle(os.path.join(output_dir, 'train_data.df'))
    
    # Save Val/Test (CSV)
    # CSV stores lists as strings. main.py uses eval() to read them.
    pd.DataFrame(val_rows).to_csv(os.path.join(output_dir, 'val_data.csv'), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)
    
    # Save Statistics
    statis_df = pd.DataFrame([{'seq_size': SEQ_LEN, 'item_num': ITEM_NUM}])
    statis_df.to_pickle(os.path.join(output_dir, 'data_statis.df'))
    
    print(f"Saved DLLM2Rec data to {output_dir}")


def main():
    metadata, reviews = load_data()
    users, item2id, id_title = filter_data(metadata, reviews)
    
    # Save id2name for BIGRec
    with open('id2name.txt', 'w') as f:
        for asin, title in id_title.items():
            if asin in item2id:
                f.write(f"{title}\t{item2id[asin]}\n")
                
    interactions, valid_users = create_interactions(users, item2id, id_title)
    train_data, valid_data, test_data = save_csv(interactions)
    
    csv_to_json('./train.csv', './train.json')
    csv_to_json('./valid.csv', './valid.json')
    csv_to_json('./test.csv', './test.json')
    csv_to_json('./valid.csv', './valid_5000.json', sample=True)
    csv_to_json('./test.csv', './test_5000.json', sample=True)
    
    generate_dllm2rec_files(train_data, valid_data, test_data, item2id)

if __name__ == '__main__':
    main()
