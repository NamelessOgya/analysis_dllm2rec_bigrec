import fire
import json
import os
from vllm import LLM, SamplingParams
try:
    from vllm import BeamSearchParams
except ImportError:
    try:
        from vllm.sampling_params import BeamSearchParams
    except ImportError:
        BeamSearchParams = None

from vllm.lora.request import LoRARequest
from tqdm import tqdm

def generate_prompt(instruction, input=None):
    # Read templates from files - aligned with BIGRec/inference.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "templates", "prompt_template.txt")
    template_no_input_path = os.path.join(script_dir, "templates", "prompt_template_no_input.txt")

    if input:
        with open(template_path, "r") as f:
            template = f.read()
        # For inference, output matches inference.py logic (empty string format)
        return template.format(
            instruction=instruction,
            input=input,
            output=""
        )
    else:
        with open(template_no_input_path, "r") as f:
            template = f.read()
        return template.format(
            instruction=instruction,
            output=""
        )

def main(
    base_model: str = "",
    lora_weights: str = "",
    test_data_path: str = "data/test.json",
    result_json_data: str = "temp.json",
    batch_size: int = 8, # Unused in vLLM but kept for compatibility
    max_new_tokens: int = 32,
    temperature: float = 0,
    top_p: float = 0.9,
    top_k: int = 40,
    num_beams: int = 4,
    tensor_parallel_size: int = 1,
    dataset: str = None,
    limit: int = -1,
):
    assert base_model, "Please specify a --base_model"

    print(f"DEBUG: Loading vLLM model {base_model}...")
    enable_lora = bool(lora_weights)
    
    # vLLM 0.12.0 optimization settings
    llm = LLM(
        model=base_model,
        enable_lora=enable_lora,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_lora_rank=64,
        gpu_memory_utilization=0.9,
        max_model_len=512,
        max_num_batched_tokens=8192,
        enable_prefix_caching=True,
    )
    print("DEBUG: vLLM model loaded with optimization settings.")

    lora_request = None
    if enable_lora:
        lora_request = LoRARequest("bigrec_adapter", 1, lora_weights)
        print(f"DEBUG: Using LoRA adapter from {lora_weights}")

    # Determine files to process
    files_to_process = []
    
    if test_data_path == "valid_test":
        if not dataset:
            raise ValueError("Argument --dataset is required when using 'valid_test'")
        files_to_process.append(("valid.json", "valid.json"))
        files_to_process.append(("test.json", "test.json"))
    elif test_data_path == "all":
        if not dataset:
            raise ValueError("Argument --dataset is required when using 'all'")
        files_to_process.append(("train.json", "train.json"))
        files_to_process.append(("valid.json", "valid.json"))
        files_to_process.append(("test.json", "test.json"))
    else:
        # Single file mode
        files_to_process.append((test_data_path, result_json_data))

    # Base directory for data if using keywords
    data_root = f"BIGRec/data/{dataset}" if dataset else ""
    
    # Process each file
    for input_name, output_name in files_to_process:
        # Resolve paths
        if test_data_path in ["valid_test", "all"]:
            # Logic: Input is relative to data root. Output is relative to result_json_data (treated as dir)
            # Check if result_json_data looks like a file (has extension)
            if result_json_data.endswith('.json'):
                # If valid_test passed but result_json_data is "res.json", that's ambiguous.
                # Usually assume result_json_data is output DIR in this mode.
                # But to be safe, unles user passes dir output, we might overwrite.
                # Let's assume result_json_data IS output_dir
                out_dir = os.path.dirname(result_json_data)
            else:
                out_dir = result_json_data
                
            os.makedirs(out_dir, exist_ok=True)
            
            real_input_path = os.path.join(data_root, input_name)
            real_output_path = os.path.join(out_dir, output_name)
        else:
            real_input_path = input_name
            real_output_path = output_name

        process_file(
            llm, 
            real_input_path, 
            real_output_path, 
            lora_request, 
            num_beams, 
            temperature, 
            top_p, 
            top_k, 
            max_new_tokens,
            limit
        )

def process_file(llm, input_path, output_path, lora_request, num_beams, temperature, top_p, top_k, max_new_tokens, limit):
    print(f"DEBUG: Processing {input_path} -> {output_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found. Skipping.")
        return

    with open(input_path, 'r') as f:
        test_data = json.load(f)
        if limit > 0:
            print(f"DEBUG: Limiting data to first {limit} items.")
            test_data = test_data[:limit]
        print(f"DEBUG: Loaded {len(test_data)} items.")

    prompts = []
    for item in test_data:
        instruction = item['instruction']
        input_text = item['input']
        prompts.append(generate_prompt(instruction, input_text))
    
    print(f"DEBUG: Starting generation...")
    
    if num_beams > 1:
        if BeamSearchParams is None:
            raise ImportError("BeamSearchParams not found. Please check vLLM version (0.12.0+ required for this implementation).")
        
        print(f"DEBUG: Using Beam Search with BeamSearchParams (width={num_beams})")
        beam_params = BeamSearchParams(
            beam_width=num_beams,
            max_tokens=max_new_tokens,
            temperature=temperature,
            ignore_eos=False 
        )
        
        # Use beam_search method
        # vLLM beam_search expects dict with "prompt" key or tokens
        beam_prompts = [{"prompt": p} for p in prompts]
        outputs = llm.beam_search(beam_prompts, beam_params)
        
        for i, output in enumerate(outputs):
            # output is BeamSearchOutput, contains 'sequences' which is List[BeamSearchSequence]
            # BeamSearchSequence has 'text'
            generated_texts = ["the recommended game is " + seq.text for seq in output.sequences]
            test_data[i]['predict'] = generated_texts
            
    else:
        print(f"DEBUG: Using Sampling (greedy/sampling)")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
        )

        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        
        for i, output in enumerate(outputs):
            generated_texts = ["the recommended game is " + o.text for o in output.outputs]
            test_data[i]['predict'] = generated_texts

    print("DEBUG: Generation complete.")

    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=4)
    print(f"DEBUG: Results saved to {output_path}")

if __name__ == "__main__":
    fire.Fire(main)
