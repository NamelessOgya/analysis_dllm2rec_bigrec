import os
import json
import torch
import pandas as pd
import numpy as np
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer

def create_dummy_qwen(output_dir):
    print(f"Creating dummy Qwen model in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    config = Qwen2Config(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(output_dir)
    
    # Try to save a simple tokenizer. 
    # We use gpt2 as a fallback if Qwen is not available or too large, 
    # but ideally we want something compatible-ish. 
    # Since we are just testing execution, any tokenizer that saves to the dir is fine.
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.save_pretrained(output_dir)
    except Exception as e:
        print(f"Warning: Could not save tokenizer: {e}")

def create_dummy_bigrec_data(data_dir):
    print(f"Creating dummy BIGRec data in {data_dir}")
    os.makedirs(data_dir, exist_ok=True)
    data = [
        {"instruction": "ins", "input": "inp", "output": "out"}
        for _ in range(10)
    ]
    with open(os.path.join(data_dir, "train.json"), "w") as f:
        json.dump(data, f)
    with open(os.path.join(data_dir, "valid_5000.json"), "w") as f:
        json.dump(data, f)

def create_dummy_dllm2rec_data(data_dir, tocf_dir):
    print(f"Creating dummy DLLM2Rec data in {data_dir} and {tocf_dir}")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(tocf_dir, exist_ok=True)

    # data_statis.df
    statis = pd.DataFrame({"seq_size": [10], "item_num": [100]})
    statis.to_pickle(os.path.join(data_dir, "data_statis.df"))

    # train_data.df
    # seq: list of ints, len_seq: int, next: int
    train_data = pd.DataFrame({
        "seq": [[1, 2, 3] for _ in range(100)],
        "len_seq": [3 for _ in range(100)],
        "next": [4 for _ in range(100)]
    })
    train_data.to_pickle(os.path.join(data_dir, "train_data.df"))

    # val_data.csv / test_data.csv
    # format: seq,len_seq,next (csv headers)
    val_df = pd.DataFrame({
        "seq": ["[1, 2, 3]" for _ in range(10)],
        "len_seq": [3 for _ in range(10)],
        "next": [4 for _ in range(10)]
    })
    val_df.to_csv(os.path.join(data_dir, "val_data.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "test_data.csv"), index=False)

    # TOCF data
    # myrank_train.txt: [train_data_num, k]
    np.savetxt(os.path.join(tocf_dir, "myrank_train.txt"), np.zeros((100, 10)), fmt="%d")
    # confidence_train.txt: [train_data_num, k]
    np.savetxt(os.path.join(tocf_dir, "confidence_train.txt"), np.zeros((100, 10)), fmt="%.4f")
    # all_embeddings.pt: [num_item, 4096]
    # item_num is 100.
    torch.save(torch.randn(101, 4096), os.path.join(tocf_dir, "all_embeddings.pt"))

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Dummy Model
    dummy_model_dir = os.path.join(base_dir, "dummy_models/qwen")
    create_dummy_qwen(dummy_model_dir)

    # BIGRec Data
    # Assuming script is in test/integration
    project_root = os.path.abspath(os.path.join(base_dir, "../../"))
    
    bigrec_data_dir = os.path.join(project_root, "BIGRec/data/dummy_data")
    create_dummy_bigrec_data(bigrec_data_dir)

    # DLLM2Rec Data
    dllm2rec_data_dir = os.path.join(project_root, "DLLM2Rec/data/dummy_data")
    dllm2rec_tocf_dir = os.path.join(project_root, "DLLM2Rec/tocf/dummy_data")
    create_dummy_dllm2rec_data(dllm2rec_data_dir, dllm2rec_tocf_dir)
    # SASRec Dummy
    sasrec_dir = os.path.join(base_dir, "dummy_models")
    create_dummy_sasrec(sasrec_dir, dllm2rec_tocf_dir)

def create_dummy_sasrec(output_dir, tocf_dir):
    print(f"Creating dummy SASRec checkpoint in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # We need to match the architecture in DLLM2Rec/main.py
    # hidden_size=64, item_num=100, state_size=10
    # We can just create a state_dict manually or try to instantiate the model.
    # Instantiating is safer to get correct keys.
    
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../DLLM2Rec")))
    
    # Mock args if needed by imports (though main.py seems safe with if __name__)
    try:
        from main import SASRec
    except ImportError:
        # Fallback if imports fail (e.g. missing deps in environment, though we are in container)
        print("Could not import SASRec, creating generic state dict")
        state_dict = {}
        # ... fill with guesses if needed, but let's try import first
        return

    # Params matching dummy data
    hidden_size = 64
    item_num = 100
    state_size = 10
    dropout = 0.1
    device = "cpu"
    
    model = SASRec(hidden_size, item_num, state_size, dropout, device)
    
    checkpoint_path = os.path.join(output_dir, "sasrec.pth")
    torch.save(model.state_dict(), checkpoint_path)
