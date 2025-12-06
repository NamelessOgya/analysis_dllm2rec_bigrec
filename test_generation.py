
import os
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

def main():
    base_model = "Qwen/Qwen2-0.5B"
    # Dummy lora path, likely won't load if not exists but we test base model generation first
    # or we can skip peft for now to isolate.
    # The user script definitely loads PeftModel. 
    # Let's try to mimic the script EXACTLY but without the loop.
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    print("Loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Simulate the hardcoded config in inference.py
    print("Applying hardcoded config...", flush=True)
    model.config.pad_token_id = tokenizer.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    
    # print("Loading Peft...", flush=True)
    # model = PeftModel.from_pretrained(model, "tloen/alpaca-lora-7b", torch_dtype=torch.float16, device_map={'': 0})
    # We skip Peft for now to see if base model hangs. If it works, we add Peft.
    
    model.eval()
    if torch.__version__ >= "2":
        # model = torch.compile(model)
        pass

    prompt = ["Test prompt for generation."]
    print("Tokenizing...", flush=True)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    
    generation_config = GenerationConfig(
        temperature=0,
        top_p=0.9,
        top_k=40,
        num_beams=4,
        num_return_sequences=4,
    )
    
    print("Starting generation...", flush=True)
    with torch.no_grad():
        generation_output = model.generate(
            **inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=32,
        )
    print("Generation done.", flush=True)

if __name__ == "__main__":
    main()
