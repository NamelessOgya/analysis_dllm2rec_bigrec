import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch.nn.functional as F

def get_embedding_model(model_name, use_embedding_model=False):
    """
    Loads the model and tokenizer.
    If use_embedding_model is True, loads as an Encoder (AutoModel) for E5.
    Otherwise loads as CausalLM (AutoModelForCausalLM).
    """
    if use_embedding_model:
        # Load E5 or similar Encoder model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    else:
        # Load Decoder model (CausalLM)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(model_name)
        
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2

    model.eval()
    return model, tokenizer

def average_pool(last_hidden_states, attention_mask):
    """
    Mean pooling for E5 models.
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def generate_embeddings(text_list, model, tokenizer, device="cuda", batch_size=16, use_embedding_model=False):
    """
    Generates embeddings for a list of texts.
    Handles E5 specific prefix and pooling if use_embedding_model is True.
    """
    
    # Add prefix for E5 if used
    if use_embedding_model:
        # "query: " is recommended for symmetric tasks in multilingual-e5 documentation
        formatted_texts = [f"query: {t}" for t in text_list]
    else:
        formatted_texts = text_list

    embeddings = []
    
    # Helper for batching
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    from tqdm import tqdm
    total_batches = (len(formatted_texts) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_texts in tqdm(batch(formatted_texts, batch_size), total=total_batches, desc="Generating Embeddings"):
            # Tokenize
            encoded_input = tokenizer(
                batch_texts, 
                max_length=512, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )
            
            input_ids = encoded_input['input_ids'].to(model.device)
            attention_mask = encoded_input['attention_mask'].to(model.device)
            
            if use_embedding_model:
                # Encoder Model (E5)
                outputs = model(input_ids, attention_mask=attention_mask)
                # Normalize embeddings
                emb = average_pool(outputs.last_hidden_state, attention_mask)
                emb = F.normalize(emb, p=2, dim=1)
            else:
                # Decoder Model (CausalLM)
                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                # Use the last hidden state of the last token
                emb = hidden_states[-1][:, -1, :]
            
            embeddings.append(emb.detach().cpu())
            
    return torch.cat(embeddings, dim=0)
