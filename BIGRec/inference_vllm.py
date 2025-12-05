import fire
import json
import os
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from tqdm import tqdm

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

### Instruction:
{instruction}

### Response:
"""

def main(
    base_model: str = "",
    lora_weights: str = "",
    test_data_path: str = "data/test.json",
    result_json_data: str = "temp.json",
    batch_size: int = 8, # Unused in vLLM but kept for compatibility
    max_new_tokens: int = 128,
    temperature: float = 0,
    top_p: float = 0.9,
    top_k: int = 40,
    num_beams: int = 4,
    tensor_parallel_size: int = 1,
):
    assert base_model, "Please specify a --base_model"

    print(f"DEBUG: Loading vLLM model {base_model}...")
    # Initialize vLLM
    # enable_lora=True if lora_weights is provided
    enable_lora = bool(lora_weights)
    
    llm = LLM(
        model=base_model,
        enable_lora=enable_lora,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_lora_rank=64, # Adjust if needed, default is usually sufficient but sometimes needs increase
        gpu_memory_utilization=0.9,
    )
    print("DEBUG: vLLM model loaded.")

    # Sampling parameters
    # Matching original script: num_beams=4, num_return_sequences=4
    # Load test data
    with open(test_data_path, 'r') as f:
        print(f"DEBUG: Loading test data from {test_data_path}...")
        test_data = json.load(f)
        print(f"DEBUG: Loaded {len(test_data)} items.")

    prompts = []
    for item in test_data:
        instruction = item['instruction']
        input_text = item['input']
        prompts.append(generate_prompt(instruction, input_text))
    
    lora_request = None
    if enable_lora:
        # Name can be anything unique
        lora_request = LoRARequest("bigrec_adapter", 1, lora_weights)
        print(f"DEBUG: Using LoRA adapter from {lora_weights}")

    # vLLM 0.12.0 has a dedicated beam_search method in LLM class.
    # We use that for beam search, and generate() for sampling.
    
    # Configure sampling parameters
    if num_beams > 1:
        print(f"DEBUG: Using beam search with beam_width={num_beams}")
        sampling_params = SamplingParams(
            temperature=0, # Temperature must be 0 for beam search
            max_tokens=max_new_tokens,
            best_of=num_beams,
            n=num_beams, # Return all beams
        )
    else:
        print(f"DEBUG: Using sampling with temperature={temperature}")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            n=num_beams, # Usually 1
            best_of=num_beams,
        )

    print(f"DEBUG: Starting generation for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    
    for i, output in enumerate(outputs):
        # output is RequestOutput, has outputs: list[CompletionOutput]
        generated_texts = [o.text for o in output.outputs]
        test_data[i]['predict'] = generated_texts

    print("DEBUG: Generation complete.")

    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)
    print(f"DEBUG: Results saved to {result_json_data}")

if __name__ == "__main__":
    fire.Fire(main)
