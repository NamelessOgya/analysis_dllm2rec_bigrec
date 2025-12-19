import json
import pandas as pd
import random
import numpy as np
import os
import csv
from tqdm import tqdm

# Set random seed
random.seed(0)
np.random.seed(0)

def load_data():
    # Read from ../movie directory to reuse downloads
    # MovieLens 10M format
    ratings_path = '../movie/ratings.dat'
    movies_path = '../movie/movies.dat'
    
    print(f"Loading movies from {movies_path}...")
    # movies.dat: MovieID::Title::Genres
    # Using '::' separator requires python engine or specific parsing
    movies_df = pd.read_csv(movies_path, sep='::', engine='python', names=['MovieID', 'Title', 'Genres'], encoding='ISO-8859-1')
    
    print(f"Loading ratings from {ratings_path}...")
    # ratings.dat: UserID::MovieID::Rating::Timestamp
    ratings_df = pd.read_csv(ratings_path, sep='::', engine='python', names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    
    return movies_df, ratings_df

def filter_data(movies_df, ratings_df):
    print("Filtering data...")
    
    # Create id_title mapping
    id_title = dict(zip(movies_df['MovieID'], movies_df['Title']))
    
    # Filter ratings consistent with movies
    ratings_df = ratings_df[ratings_df['MovieID'].isin(movies_df['MovieID'])]
    
    users = set(ratings_df['UserID'])
    items = set(ratings_df['MovieID'])
    
    # Create item2id map (0-based index)
    # Sort items to ensure deterministic ID assignment
    item2id = dict()
    count = 0
    for item in sorted(list(items)):
        item2id[item] = count
        count += 1
        
    print(f"Users: {len(users)}, Items: {len(items)}, Ratings: {len(ratings_df)}")
    
    # Process users into dictionary
    processed_users = dict()
    
    # Group by UserID for faster processing
    # ratings_df is strictly UserID, MovieID, Rating, Timestamp
    grouped = ratings_df.groupby('UserID')
    
    for user_id, group in tqdm(grouped):
        # MovieLens has high quality data, but let's ensure sorted by timestamp
        group = group.sort_values('Timestamp')
        
        user_items = group['MovieID'].tolist()
        user_ratings = group['Rating'].tolist()
        user_timestamps = group['Timestamp'].tolist()
        
        processed_users[user_id] = {
            'items': user_items,
            'ratings': user_ratings,
            'timestamps': user_timestamps,
            # Pre-compute IDs and Titles
            'item_ids': [item2id[x] for x in user_items],
            'item_titles': [id_title[x] for x in user_items]
        }
        
    return processed_users, item2id, id_title

def create_interactions(users, item2id, id_title):
    print("Creating interactions...")
    interactions = []
    
    for key in tqdm(users.keys()):
        user_data = users[key]
        items = user_data['items']
        item_ids = user_data['item_ids']
        item_titles = user_data['item_titles']
        ratings = user_data['ratings']
        timestamps = user_data['timestamps']
        
        # Game Bigrec Logic:
        # History window size 10
        # for i in range(min(10, len(items) - 1), len(items)):
        # If len=5. min(10, 4) = 4. range(4, 5). i=4.
        
        # Check if user has enough interactions?
        # Game Bigrec does not strictly filter min length here, loop handles it.
        # But if len(items) < 2, range(min(10, 1), 2) -> range(1, 2) -> i=1.
        # history=[0:1] (1 item). target=1.
        
        length = len(items)
        if length < 2:
            continue

        for i in range(min(10, length - 1), length):
            st = max(i - 10, 0)
            
            interactions.append([
                key, # user_id
                items[st: i], 
                items[i], 
                item_ids[st: i], 
                item_ids[i], 
                item_titles[st: i], 
                item_titles[i], 
                ratings[st: i], 
                ratings[i], 
                int(timestamps[i])
            ])

    return interactions

def save_csv(interactions):
    print("Saving CSV files...")
    # Sort by timestamp
    interactions = sorted(interactions, key=lambda x: x[-1])
    
    # Add unique ID to each interaction
    interactions_with_uid = []
    for uid, row in enumerate(interactions):
        interactions_with_uid.append(row + [uid])
    interactions = interactions_with_uid

    header = ['user_id', 'item_asins', 'item_asin', 'history_item_id', 'item_id', 'history_item_title', 'item_title', 'history_rating', 'rating', 'timestamp', 'uid']
    
    # 80/10/10 Split
    n = len(interactions)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.9) # Cumulative 90%
    
    train_data = interactions[:n_train]
    valid_data = interactions[n_train:n_valid]
    test_data = interactions[n_valid:]
    
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
        
    json_list = []
    for index, row in tqdm(data.iterrows(), total=len(data)):
        try:
            # history_item_title is stored as string representation of list
            # e.g. "['Movie A', 'Movie B']"
            # In CSV it might be quoted.
            history_titles = eval(row['history_item_title'])
        except Exception as e:
            print(f"Error parsing history titles row {index}: {e}")
            continue
            
        L = len(history_titles)
        history = "The user has watched the following movies before:" # Changed "played ... video games" to "watched ... movies"
        for i in range(L):
            if i == 0:
                history += "\"" + str(history_titles[i]) + "\""
            else:
                history += ", \"" + str(history_titles[i]) + "\""
        
        target_item = str(row['item_title'])
        target_item_str = "\"" + target_item + "\""
        uid = int(row['uid'])
        
        json_list.append({
            "instruction": "Given a list of movies the user has watched before, please recommend a new movie that the user likes to the user.",
            "input": f"{history}\n ",
            "output": target_item_str,
            "meta": {
                "uid": uid
            }
        })        
        
    with open(output_path, 'w') as f:
        json.dump(json_list, f, indent=4)

def main():
    movies_df, ratings_df = load_data()
    users, item2id, id_title = filter_data(movies_df, ratings_df)
    
    # Save id2name.txt for BIGRec
    # Format: Title\tID
    # Sorted by ID
    
    sorted_items = sorted(item2id.items(), key=lambda x: x[1])
    
    print("Saving id2name.txt...")
    with open('id2name.txt', 'w') as f:
        for movie_id, iid in sorted_items:
            title = id_title.get(movie_id, str(movie_id))
            f.write(f"{title}\t{iid}\n")
                
    interactions = create_interactions(users, item2id, id_title)
    
    train_data, valid_data, test_data = save_csv(interactions)
    
    csv_to_json('./train.csv', './train.json')
    csv_to_json('./valid.csv', './valid.json')
    csv_to_json('./test.csv', './test.json')
    csv_to_json('./valid.csv', './valid_5000.json', sample=True)
    csv_to_json('./test.csv', './test_5000.json', sample=True)
    
    # DLLM2Rec conversion is handled by external script

if __name__ == '__main__':
    main()
