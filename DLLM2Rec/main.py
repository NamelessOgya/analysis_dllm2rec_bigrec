import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import os
from utility import pad_history, calculate_hit, extract_axis_1
from collections import Counter
from SASRecModules_ori import *
import time
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Run supervised GRU.")
    # setting
    parser.add_argument('--model_name', type=str, default='SASRec', help='model name.')
    parser.add_argument('--data', nargs='?', default='game',  help='movie, game, toy')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout ')
    parser.add_argument('--cuda', type=int, default=3, help='cuda device.')                    
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')        
    parser.add_argument('--epoch', type=int, default=1000, help='Number of max epochs.')
    parser.add_argument('--hidden_factor', type=int, default=64, help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--num_negtive_items', type=int, default=1, help='neg sample')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--l2_decay', type=float, default=0, help='weight decay')
    # caser
    parser.add_argument('--num_filters', type=int, default=16, help='num_filters')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]', help='Specify the filter_size')
    # dro
    parser.add_argument('--alpha', type=float, default=1.0, help='weight for dros loss')
    parser.add_argument('--beta', type=float, default=1.0, help='for robust radius')
    # dllm2rec
    parser.add_argument('--ed_weight', type=float, default=0.2, help='weight for collaborative embedding distillation')
    parser.add_argument('--lam', type=float, default=0.8, help='weight for importance-aware ranking distillation') 
    parser.add_argument('--candidate_topk', type=int, default=10, help='top k items from llm')
    parser.add_argument('--gamma_position', type=float, default=0.3, help='weight for ranking position-aware')
    parser.add_argument('--gamma_confidence', type=float, default=0.5, help='weight for ranking importance-aware')
    parser.add_argument('--gamma_consistency', type=float, default=0.1, help='weight for ranking consistency-aware')
    parser.add_argument('--beta2', type=float, default=1.0, help='weight for importance-aware ranking distillation')
    # Custom paths for BIGRec data
    parser.add_argument('--embedding_path', type=str, default=None, help='Path to LLM embeddings file (.pt)')
    parser.add_argument('--ranking_path', type=str, default=None, help='Path to ranking file (.txt)')
    parser.add_argument('--confidence_path', type=str, default=None, help='Path to confidence file (.txt)')
    parser.add_argument('--teacher_model', type=str, default="", help='Name of the teacher model for directory naming')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed for student (and implies teacher seed)')
    parser.add_argument('--teacher_sample', type=str, default="", help='Sample size of teacher model for directory naming')
    parser.add_argument('--export_train_scores', action='store_true', help='If True, export train scores (train.pt) at the end')
    return parser.parse_args()

    
class GRU(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, gru_layers=1, llm_input_dim=4096):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.state_size = state_size
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=gru_layers,
            batch_first=True
        )
        self.s_fc = nn.Linear(self.hidden_size, self.item_num + 1)


        self.fc_llm = nn.Linear(llm_input_dim, 64)

    def forward(self, states, len_states, llm_emb=None):
        # Supervised Head
        emb = self.item_embeddings(states)
        if llm_emb != None:
            llm_emb = self.fc_llm(llm_emb.float())
            emb = emb + args.ed_weight * llm_emb
        if 0 in len_states:
            len_states = [max(1, length) for length in len_states]
        emb_packed = torch.nn.utils.rnn.pack_padded_sequence(emb, len_states, batch_first=True, enforce_sorted=False)
        emb_packed, hidden = self.gru(emb_packed)
        hidden = hidden.view(-1, hidden.shape[2])
        supervised_output = self.s_fc(hidden)
        return supervised_output

class Caser(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, num_filters, filter_sizes,
                 dropout_rate, llm_input_dim=4096):
        super(Caser, self).__init__()
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.state_size = state_size
        self.filter_sizes = eval(filter_sizes)
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=self.hidden_size,
        )

        # init embedding
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)

        # Horizontal Convolutional Layers
        self.horizontal_cnn = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (i, self.hidden_size)) for i in self.filter_sizes])
        # Initialize weights and biases
        for cnn in self.horizontal_cnn:
            nn.init.xavier_normal_(cnn.weight)
            nn.init.constant_(cnn.bias, 0.1)

        # Vertical Convolutional Layer
        self.vertical_cnn = nn.Conv2d(1, 1, (self.state_size, 1))
        nn.init.xavier_normal_(self.vertical_cnn.weight)
        nn.init.constant_(self.vertical_cnn.bias, 0.1)

        # Fully Connected Layer
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        final_dim = self.hidden_size + self.num_filters_total
        self.s_fc = nn.Linear(final_dim, item_num + 1)


        # dropout
        self.dropout = nn.Dropout(self.dropout_rate)

        self.fc_llm = nn.Linear(llm_input_dim, 64)

    def forward(self, states, len_states, llm_emb=None):
        input_emb = self.item_embeddings(states)
        if llm_emb != None:
            llm_emb = self.fc_llm(llm_emb.float())
            input_emb = input_emb + args.ed_weight * llm_emb
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1)
        input_emb *= mask
        input_emb = input_emb.unsqueeze(1)
        pooled_outputs = []
        for cnn in self.horizontal_cnn:
            h_out = nn.functional.relu(cnn(input_emb))
            h_out = h_out.squeeze()
            p_out = nn.functional.max_pool1d(h_out, h_out.shape[2])
            pooled_outputs.append(p_out)

        h_pool = torch.cat(pooled_outputs, 1)
        h_pool_flat = h_pool.view(-1, self.num_filters_total)

        v_out = nn.functional.relu(self.vertical_cnn(input_emb))
        v_flat = v_out.view(-1, self.hidden_size)

        out = torch.cat([h_pool_flat, v_flat], 1)
        out = self.dropout(out)
        supervised_output = self.s_fc(out)

        return supervised_output

class SASRec(nn.Module):
    def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1, llm_input_dim=4096):
        super(SASRec, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.item_embeddings = nn.Embedding(
            num_embeddings=item_num + 1,
            embedding_dim=hidden_size,
        )
        nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(hidden_size)
        self.ln_2 = nn.LayerNorm(hidden_size)
        self.ln_3 = nn.LayerNorm(hidden_size)
        self.mh_attn = MultiHeadAttention(hidden_size, hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
        self.s_fc = nn.Linear(hidden_size, item_num + 1)

           
        self.fc_llm = nn.Linear(llm_input_dim, 64)
        self.relu = nn.ReLU()

    def forward(self, states, len_states, llm_emb=None):
        inputs_emb = self.item_embeddings(states)
        if llm_emb != None:
            llm_emb = self.fc_llm(llm_emb.float())
            inputs_emb = inputs_emb + args.ed_weight * llm_emb
        inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        
        mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)        
        seq *= mask
        seq_normalized = self.ln_1(seq)

        mh_attn_out = self.mh_attn(seq_normalized, seq) # cost time

        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out *= mask
        ff_out = self.ln_3(ff_out)  
        state_hidden = extract_axis_1(ff_out, len_states - 1) 
        supervised_output = self.s_fc(state_hidden).squeeze()

        return supervised_output
    

def myevaluate(model, test_data, device, llm_all_emb=None):
    states = []
    len_states = []
    actions = []
    uids = []
    total_purchase = 0
    import csv
    with open(test_data, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            seq = eval(row['seq'])
            len_seq = int(row['len_seq'])
            next_item = float(row['next'])
            states.append(seq)
            len_states.append(len_seq)
            actions.append(next_item)
            if 'uid' in row:
                uids.append(int(row['uid']))
            total_purchase += 1

    states = np.array(states)
    states = states.astype(np.int64)
    states = torch.LongTensor(states)
    states = states.to(device)

    if llm_all_emb != None:
        seq = states
        llm_dim = llm_all_emb.shape[1]
        llm_emb = torch.zeros(seq.size(0), seq.size(1), llm_dim, dtype=llm_all_emb.dtype, device=device)
        mask = seq < llm_all_emb.size(0)
        llm_emb[mask] = llm_all_emb[seq[mask]]
        llm_emb = llm_emb.to(device)
    else:
        llm_emb = None

    # model.forward
    prediction = model.forward(states, np.array(len_states),llm_emb) # [num_test,num_item]
    sorted_list = torch.argsort(prediction.detach()).cpu().numpy()

    hit_purchase = [0] * len(topk)
    ndcg_purchase = [0] * len(topk)
    calculate_hit(sorted_list=sorted_list, topk=topk, true_items=actions, hit_purchase=hit_purchase,
                  ndcg_purchase=ndcg_purchase)

    print('#' * 120)
    hr_list = []
    ndcg_list = []
    
    # Dynamic header
    header = '\t'.join([f'hr@{k}\tndcg@{k}' for k in topk])
    print(header)

    for i in range(len(topk)):
        hr_purchase = hit_purchase[i] / total_purchase
        ng_purchase = ndcg_purchase[i] / total_purchase
        hr_list.append(hr_purchase)
        if ng_purchase == 0.0:
            ndcg_list.append(ng_purchase)
        else:
            ndcg_list.append(ng_purchase[0, 0])
        
        # Track @20 specifically
        if topk[i] == 20:
            hr_20 = hr_purchase
            ndcg_20 = ng_purchase
            rec_list = sorted_list[:, -topk[i]:]

    # Dynamic values print
    values_str = '\t'.join(['{:.6f}\t{:.6f}'.format(h, n) for h, n in zip(hr_list, ndcg_list)])
    print(values_str)
    print('#' * 120)
    return prediction[:, 1:], hr_list, ndcg_list, uids

def myevaluate_train(model, train_data, device, llm_all_emb=None, batch_size=256):
    print("Evaluating Training Data for Export...")
    states = []
    len_states = []
    uids = []
    
    # Process train_data (DataFrame) sequentially
    # train_data has 'seq', 'len_seq', 'next', 'uid'
    
    all_preds = []
    all_uids = []
    
    num_rows = len(train_data)
    num_batches = (num_rows + batch_size - 1) // batch_size
    
    # We can iterate directly using index slicing for sequential access
    import math
    
    model.eval() # Ensure eval mode
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_rows)
            batch = train_data.iloc[start_idx:end_idx]
            
            # Extract features
            # seq is already list of lists in DataFrame (if unpickled correctly?)
            # Wait, in training loop (line 492) it checks if string and converts.
            # Let's assume similar logic might be needed if loaded from pickle sometimes?
            # Creating tensors
            
            batch_seq = list(batch['seq'])
            batch_len = list(batch['len_seq'])
            batch_uid = list(batch['uid'])
            
            # Helper to handle string/list ambiguity if present (copy from train loop logic)
            import ast
            if len(batch_seq) > 0 and isinstance(batch_seq[0], str):
                 batch_seq = [[int(num) for num in ast.literal_eval(s)] for s in batch_seq]
            
            # Tensor conversion
            seq_tensor = torch.LongTensor(batch_seq).to(device)
            # len_tensor = torch.LongTensor(batch_len).to(device) # SASRec logic checks model name?
            # Model forward expects numpy array for len_states?
            # Check myevaluate: "np.array(len_states)" passed to forward
            # Check main loop: "len_seq = torch.LongTensor(len_seq).to(device)" passed to forward
            # SASRecModules_ori.py forward takes (log_seqs, item_indices, ...) ??
            # Wait, line 525: model.forward(seq, len_seq, llm_emb)
            # In myevaluate (line 242): model.forward(states, np.array(len_states), llm_emb)
            # Let's verify SASRec vs GRU. User is using SASRec.
            # main loop uses torch tensor. myevaluate uses numpy?
            # Let's follow main loop pattern for safety.
            
            len_tensor = torch.LongTensor(batch_len).to(device)
            
            # LLM Emb
            if llm_all_emb is not None:
                llm_dim = llm_all_emb.shape[1]
                llm_emb = torch.zeros(seq_tensor.size(0), seq_tensor.size(1), llm_dim, dtype=llm_all_emb.dtype, device=device)
                mask = seq_tensor < llm_all_emb.size(0)
                llm_emb[mask] = llm_all_emb[seq_tensor[mask]]
                llm_emb = llm_emb.to(device)
            else:
                llm_emb = None
                
            # Forward
            preds = model.forward(seq_tensor, len_tensor, llm_emb) # [B, ItemNum]
            
            # Slice padding
            preds = preds[:, 1:]
            
            # Append (convert to half to save memory: 4GB -> 2GB approx)
            all_preds.append(preds.half().cpu())
            all_uids.extend(batch_uid)
            
            if (i+1) % 100 == 0:
                print(f"Processed {i+1}/{num_batches} batches...")

    # Concatenate
    full_prediction = torch.cat(all_preds, dim=0)
    # Convert back to float32 if needed for saving? torch.save supports half.
    # evaluate.py loads it. We should check if evaluate.py handles half.
    # evaluate.py: ci_train = torch.load(...) -> (ci_train - min)/(max-min). 
    # Half has smaller range, could overflow if raw logits are huge?
    # Logits from SASRec usually reasonable range.
    # Let's keep it half for saving too.
    return full_prediction, all_uids

def calcu_propensity_score(buffer):
    items = list(buffer['next'])
    freq = Counter(items)
    for i in range(item_num):
        if i not in freq.keys():
            freq[i] = 0
    pop = [freq[i] for i in range(item_num)]
    pop = np.array(pop)
    ps = pop + 1
    ps = ps / np.sum(ps)
    ps = np.power(ps, 0.05)
    return ps

def set_seed(seed):
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def save_metrics(args, test_hr, test_ndcg, valid_hr, valid_ndcg):
    # Determine distillation status for directory naming
    is_distilled = (args.ed_weight > 0 or args.lam > 0)
    
    is_distilled = (args.ed_weight > 0 or args.lam > 0)
    
    if is_distilled:
        if args.teacher_model:
            teacher_name = args.teacher_model.replace('/', '_')
        else:
            teacher_name = "unknown_teacher"
        
        # Parent directory: [DATASET]/[STUDENT_MODEL]_distilled_[TEACHER_MODEL]
        parent_dir_name = f"{args.model_name.lower()}_distilled_{teacher_name}"
        
        # Hyperparameter directory: ed_[ED_WEIGHT]_lam_[LAM]
        hyper_dir_name = f"ed_{args.ed_weight}_lam_{args.lam}"
        
        # Sub directory: [seed]_[TEACHER_SAMPLE]
        if args.teacher_sample:
            sub_dir_name = f"{args.seed}_{args.teacher_sample}"
        else:
             sub_dir_name = f"{args.seed}_unknown"
             
        # Order: Parent -> Seed/Sample -> Hyperparams
        output_dir = os.path.join("results", args.data, parent_dir_name, sub_dir_name, hyper_dir_name)
        
    else:
        # No distillation case
        # [DATASET]/[STUDENT_MODEL]_no_distillation/[seed]/alpha_[ALPHA]
        hyper_dir_name = f"alpha_{args.alpha}"
        output_dir = os.path.join("results", args.data, f"{args.model_name.lower()}_no_distillation", str(args.seed), hyper_dir_name)

    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics to a single file aligned with BIGRec format
    metrics_dict = {
        "test": {
            "NDCG": test_ndcg,
            "HR": test_hr
        },
        "valid": {
            "NDCG": valid_ndcg,
            "HR": valid_hr
        }
    }
    
    with open(os.path.join(output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    
    print(f"Saved metrics to {output_dir}/metrics.json")

if __name__ == '__main__':
    args = parse_args()
    s = time.time()
    set_seed(args.seed)

    print('-' * 40 + 'ARGUMENTS' + '-' * 40)
    for arg in vars(args):
        print('{:40} {}'.format(arg, getattr(args, arg)))
    print('-' * 40 + 'ARGUMENTS' + '-' * 40)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    data_directory = './data/' + args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num
    seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
    item_num = data_statis['item_num'][0]  # total number of items

    topk = [1, 3, 5, 10, 20, 50]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tocf_data_directory = './tocf/' + args.data
    if args.ed_weight != 0:
        if args.embedding_path:
            llm_all_emb_path = args.embedding_path
        else:
            llm_all_emb_path = os.path.join(tocf_data_directory, 'all_embeddings.pt')
        print(f"Loading embeddings from: {llm_all_emb_path}")
        llm_all_emb = torch.load(llm_all_emb_path) # [num_item, 4096]
        llm_all_emb = llm_all_emb.to(device)
        llm_input_dim = llm_all_emb.shape[1]
    else:
        llm_all_emb = None
        llm_input_dim = 4096 # default
        print('without using collaborative embedding distillation!')

    model_name = args.model_name
    if model_name == "GRU":           
        model = GRU(args.hidden_factor, item_num, seq_size, llm_input_dim=llm_input_dim)
    elif model_name == "SASRec":
        model = SASRec(args.hidden_factor, item_num, seq_size, args.dropout_rate, device, llm_input_dim=llm_input_dim)
    elif model_name == "Caser":
        model = Caser(args.hidden_factor,item_num, seq_size, args.num_filters, args.filter_sizes, args.dropout_rate, llm_input_dim=llm_input_dim)
    else:
        print("check model name!")
        exit(-1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    bce_loss = nn.BCEWithLogitsLoss()
    model.to(device)

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    if args.alpha == 0:
        print('without using dros!')
    else:
        ps = calcu_propensity_score(train_data)
        ps = torch.tensor(ps)
        ps = ps.to(device)

    if args.lam != 0:       
        if args.ranking_path:
            candidate_path = args.ranking_path
        else:
            candidate_path = os.path.join(tocf_data_directory, 'myrank_train.txt')
        print(f"Loading ranking from: {candidate_path}")
        all_candidate = np.loadtxt(candidate_path)
        all_candidate = torch.LongTensor(all_candidate).to(device) # [train_data_num, k]
        
        if args.confidence_path:
            llm_confidence_path = args.confidence_path
        else:
            llm_confidence_path = os.path.join(tocf_data_directory, 'confidence_train.txt')
        print(f"Loading confidence from: {llm_confidence_path}")
        llm_confidence = np.loadtxt(llm_confidence_path)
        llm_confidence = torch.tensor(llm_confidence, dtype=torch.float).to(device) # [train_data_num, k]
    else:
        print('without using importance-aware ranking distillation!')


    total_step = 0
    best_ndcg20 = -1.0
    best_hr20 = 0
    best_step = 0
    patient = 0
    best_hr_list_result = []
    best_ndcg_list_result = []
    best_val_hr_list_result = []
    best_val_ndcg_list_result = []
    best_accuracy = {}
    best_prediction = 0
    num_rows = train_data.shape[0]
    num_batches = int(num_rows / args.batch_size)

    # Determine output directory early to save .pt files
    is_distilled = (args.ed_weight > 0 or args.lam > 0)
    if is_distilled:
        if args.teacher_model:
            teacher_name = args.teacher_model.replace('/', '_')
        else:
            teacher_name = "unknown_teacher"
        
        # Parent directory: [DATASET]/[STUDENT_MODEL]_distilled_[TEACHER_MODEL]
        parent_dir_name = f"{args.model_name.lower()}_distilled_{teacher_name}"
        
        # Hyperparameter directory: ed_[ED_WEIGHT]_lam_[LAM]
        hyper_dir_name = f"ed_{args.ed_weight}_lam_{args.lam}"
        
        # Sub directory: [seed]_[TEACHER_SAMPLE]
        if args.teacher_sample:
            sub_dir_name = f"{args.seed}_{args.teacher_sample}"
        else:
             sub_dir_name = f"{args.seed}_unknown"
             
        # Order: Parent -> Seed/Sample -> Hyperparams
        output_dir = os.path.join("results", args.data, parent_dir_name, sub_dir_name, hyper_dir_name)
        
    else:
        # No distillation case
        # [DATASET]/[STUDENT_MODEL]_no_distillation/[seed]/alpha_[ALPHA]
        hyper_dir_name = f"alpha_{args.alpha}"
        output_dir = os.path.join("results", args.data, f"{args.model_name.lower()}_no_distillation", str(args.seed), hyper_dir_name)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    for i in range(args.epoch):
        s_epoch = time.time()
        for j in range(num_batches):
            batch = train_data.sample(n=args.batch_size)
            sample = batch.index
            batch = batch.to_dict()
            
            seq = list(batch['seq'].values()) # [batch_size, 10]
            len_seq = list(batch['len_seq'].values())
            target = list(batch['next'].values())

            optimizer.zero_grad()

            import ast
            if type(seq[0]) == str:
                seq = [[int(num) for num in ast.literal_eval(s)] for s in seq]
            if type(target[0]) == str:
                target = [ast.literal_eval(s) for s in target]
            target = [int(s) for s in target]
            seq = torch.LongTensor(seq).to(device)
            if model_name == "SASRec":
                len_seq = torch.LongTensor(len_seq).to(device)
            target = torch.LongTensor(target).to(device)
            
            # negtive item sampling 
            real_batch_size = args.batch_size
            num_negtive_items = args.num_negtive_items
            zeros_tensor = torch.zeros((real_batch_size, item_num+1), device=device)
            zeros_tensor[torch.arange(real_batch_size).unsqueeze(1).repeat(1, 10), seq] = 1
            zeros_tensor[torch.arange(real_batch_size), target] = 1
            zeros_tensor = zeros_tensor[:,:-1]
            neg_tensor = 1 - zeros_tensor
            batch_neg = torch.multinomial(
                neg_tensor, num_negtive_items, replacement=True
            )
            target_neg = batch_neg.to(device)

            # llm_emb getting
            if llm_all_emb != None:
                llm_emb = torch.zeros(seq.size(0), seq.size(1), llm_input_dim, dtype=llm_all_emb.dtype, device=device)
                mask = seq < llm_all_emb.size(0)
                llm_emb[mask] = llm_all_emb[seq[mask]]
                llm_emb = llm_emb.to(device)  
            else:
                llm_emb = None

            # model forward          
            model_output = model.forward(seq, len_seq, llm_emb)

            # bce loss
            target = target.view(args.batch_size, 1)
            target_neg = target_neg.view(args.batch_size, 1)
            pos_scores = torch.gather(model_output, 1, target)
            neg_scores = torch.gather(model_output, 1, target_neg)
            pos_labels = torch.ones((args.batch_size, 1))
            neg_labels = torch.zeros((args.batch_size, 1))
            scores = torch.cat((pos_scores, neg_scores), 0)
            labels = torch.cat((pos_labels, neg_labels), 0)
            scores = scores.to(device)
            labels = labels.to(device)
            loss = bce_loss(scores, labels)

            # dros loss
            if args.alpha != 0:
                pos_scores_dro = torch.gather(torch.mul(model_output * model_output, ps), 1, target)
                pos_scores_dro = torch.squeeze(pos_scores_dro)
                pos_loss_dro = torch.gather(torch.mul((model_output - 1) * (model_output - 1), ps), 1, target)
                pos_loss_dro = torch.squeeze(pos_loss_dro)
                inner_dro = (torch.sum(torch.exp((torch.mul(model_output * model_output, ps) / args.beta)), 1)
                            - torch.exp((pos_scores_dro / args.beta)) 
                            + torch.exp((pos_loss_dro / args.beta)))
                loss_dro = torch.log(inner_dro + 1e-24)
                loss_all = loss + args.alpha * torch.mean(loss_dro)
            else:
                loss_all = loss

            if args.lam != 0:
                candidate = all_candidate[sample] # [1024,k]
                candidate = candidate[:,:args.candidate_topk]   
                # weight_rank
                _lambda = 1
                _K = args.candidate_topk
                weight_static = torch.arange(1, _K + 1, dtype=torch.float32)
                weight_static = torch.exp(-weight_static / _lambda) # 1/exp(r)
                weight_static = weight_static.unsqueeze(0) # [1, k]
                weight_static = weight_static.repeat(args.batch_size, 1)
                weight_rank = weight_static / torch.sum(weight_static, dim=1).unsqueeze(1)
                weight_rank = weight_rank.to(device)               
                # weight_com
                cf_rank_top = (-model_output).argsort(dim=1)[:, :_K].to(device) # candidate [1024, k]
                common_tensor = torch.zeros_like(candidate).to(device)
                common_mask = candidate.unsqueeze(2) == cf_rank_top.unsqueeze(1)
                common_tensor = common_mask.any(dim=2).int() + 1e-8
                weight_com = common_tensor.to(device)
                # weight_confidence
                candidate_confidence = llm_confidence[sample] # [1024,k]
                candidate_confidence = candidate_confidence[:,:_K]
                weight_confidence = torch.exp(-candidate_confidence) + 1e-8
                weight_confidence = weight_confidence / torch.sum(weight_confidence, dim=1).unsqueeze(1)
                weight_confidence = weight_confidence.to(device)
                # weight_fin
                weight_fin = args.gamma_position*weight_rank + args.gamma_confidence*weight_confidence + args.gamma_consistency*weight_com
                weight = weight_fin / torch.sum(weight_fin, dim=1).unsqueeze(1) # [1024,k]
                # distillation loss
                loss_all_rd = 0
                num_candidate = candidate.size(1)
                for i_ in range(num_candidate):
                    target = candidate[:, i_:i_+1] # [1024,1]
                    # bce loss
                    pos_scores = torch.gather(model_output, 1, target) # [1024,1]
                    neg_scores = torch.gather(model_output, 1, target_neg) # [1024,1]
                    pos_labels = torch.ones((args.batch_size, 1)).to(device) # [1024,1]
                    neg_labels = torch.zeros((args.batch_size, 1)).to(device) # [1024,1]
                    loss_bce_rd = -(pos_labels*torch.log(torch.sigmoid(pos_scores)) + (1-neg_labels)*torch.log(torch.sigmoid(1-neg_scores)))
                    if args.alpha != 0:
                        # dro loss
                        pos_scores_dro = torch.gather(torch.mul(model_output * model_output, ps), 1, target)
                        pos_scores_dro = torch.squeeze(pos_scores_dro) # [1024,1]
                        pos_loss_dro = torch.gather(torch.mul((model_output - 1) * (model_output - 1), ps), 1, target)
                        pos_loss_dro = torch.squeeze(pos_loss_dro) # [1024,1]
                        A = torch.sum(torch.exp((torch.mul(model_output * model_output, ps) / args.beta)), 1)
                        B = torch.exp((pos_scores_dro / args.beta))
                        C = torch.exp((pos_loss_dro / args.beta))
                        inner_dro_rd =  A - B + C
                        loss_dro_rd = torch.log(inner_dro_rd + 1e-24) # [1024]
                        # all loss
                        loss_all_rd +=  (weight[:, i_:i_+1]*loss_bce_rd).mean() + args.alpha * (weight[:, i_:i_+1]*loss_dro_rd).mean() 
                    else:
                        loss_all_rd +=  (weight[:, i_:i_+1]*loss_bce_rd).mean()
                loss_all = loss_all + args.lam * (loss_all_rd)

            if torch.isnan(loss_all).any():
                print('loss is nan!!!')
                exit(-1)
            loss_all.backward()
            optimizer.step()
            
        if True:
            step = i + 1
            e_epoch = time.time()
            print(f"the loss in {step}th step is: {loss_all}, train this epoch cost {e_epoch-s_epoch} s")            
            if step % 1 == 0:
                print('VAL PHRASE:')
                val_path = os.path.join(data_directory, 'val_data.csv')
                val_prediction, val_hr_list, val_ndcg_list, val_uids = myevaluate(model, val_path, device, llm_all_emb)
                print('TEST PHRASE:')
                test_path = os.path.join(data_directory, 'test_data.csv')
                s_test = time.time()
                prediction, hr_list, ndcg_list, test_uids = myevaluate(model, test_path, device, llm_all_emb)
                e_test = time.time()
                # Assuming topk=[1, 3, 5, 10, 20, 50], index 4 is @20
                if ndcg_list[4] > best_ndcg20:
                    patient = 0 
                    best_ndcg20 = ndcg_list[4]
                    best_hr20 = hr_list[4]
                    best_hr_list_result = hr_list
                    best_ndcg_list_result = ndcg_list
                    best_val_hr_list_result = val_hr_list
                    best_val_ndcg_list_result = val_ndcg_list
                    best_step = step
                    best_prediction = prediction
                    
                    # Save best predictions (CI scores)
                    print(f"New best model found! Saving scores to {output_dir}...")
                    torch.save(val_prediction, os.path.join(output_dir, "val.pt"))
                    torch.save(val_prediction, os.path.join(output_dir, "val.pt"))
                    torch.save(prediction, os.path.join(output_dir, "test.pt"))
                    torch.save(torch.LongTensor(val_uids), os.path.join(output_dir, "val_uids.pt"))
                    torch.save(torch.LongTensor(test_uids), os.path.join(output_dir, "test_uids.pt"))
                    # Save model state dict for reloading
                    torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
                else:
                    patient += 1 
                print(f'patient={patient}, BEST STEP:{best_step}, BEST NDCG@20:{best_ndcg20}, BEST HR@20:{best_hr20}, test cost:{e_test-s_test}s')
                
                if patient >= 50: # Default patience usually 10-20? User said early stopping not working in BIGRec, here it works?
                    print("Early stopping triggered (Patient hit limit).")
                    break
        
        if patient >= 50:
             break

    # TRAINING FINISHED
    # Export train scores if requested
    if args.export_train_scores:
        print("="*30)
        print("Starting Export of Training Scores (for Distillation)...")
        
        # Load best model state
        best_model_path = os.path.join(output_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}...")
            model.load_state_dict(torch.load(best_model_path))
        else:
            print("WARNING: Best model not found. Using current model state (might be suboptimal).")
        
        train_pred, train_uids = myevaluate_train(model, train_data, device, llm_all_emb, batch_size=args.batch_size)
        
        print(f"Saving train scores to {output_dir}...")
        torch.save(train_pred, os.path.join(output_dir, "train.pt"))
        torch.save(torch.LongTensor(train_uids), os.path.join(output_dir, "train_uids.pt"))
        print("Export Complete.")
        if patient >= 10:
            e = time.time()
            cost = (e - s)/60
            print(f'=============early stop=============')
            print(f'cost {cost} min')
            save_metrics(args, best_hr_list_result, best_ndcg_list_result, best_val_hr_list_result, best_val_ndcg_list_result)
            exit(0)
