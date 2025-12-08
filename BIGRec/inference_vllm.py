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

def generate_prompt(instruction, input=None, prompt_file=None):
    # Read templates from files - aligned with BIGRec/inference.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if prompt_file:
         template_path = prompt_file
    else:
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
    num_beams: int = 1,
    tensor_parallel_size: int = 1,
    dataset: str = None,
    limit: int = -1,
    prompt_file: str = None,
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
        max_model_len=1024,
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
            limit,
            batch_size,
            prompt_file
        )

def process_file(llm, input_path, output_path, lora_request, num_beams, temperature, top_p, top_k, max_new_tokens, limit, batch_size, prompt_file=None):
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
        prompts.append(generate_prompt(instruction, input_text, prompt_file))
    
    print(f"DEBUG: Starting generation...")
    
    # --- Deep LoRA Verification ---
    if lora_request and len(prompts) > 0:
        print("DEBUG: performing LoRA effectiveness check on first prompt...")
        try:
            # Create a localized SamplingParams for this test
            # We import SamplingParams at top level, assuming it is available here
            # We need logprobs to compare distributions
            test_params = SamplingParams(temperature=0, max_tokens=1, logprobs=10)
            
            # 1. Generate without LoRA
            output_base = llm.generate([prompts[0]], test_params, lora_request=None, use_tqdm=False)
            
            # 2. Generate with LoRA
            output_lora = llm.generate([prompts[0]], test_params, lora_request=lora_request, use_tqdm=False)
            
            # Compare first token logprobs
            # output is RequestOutput object
            # output.outputs[0] is CompletionOutput
            # .logprobs[0] is Dict[int, Logprob] for the first token
            
            base_logprobs = output_base[0].outputs[0].logprobs[0]
            lora_logprobs = output_lora[0].outputs[0].logprobs[0]
            
            # Extract simple dict for comparison: {token_id: logprob_float}
            base_dist = {k: v.logprob for k, v in base_logprobs.items()}
            lora_dist = {k: v.logprob for k, v in lora_logprobs.items()}
            
            # Check for exact equality
            is_identical = (base_dist == lora_dist)
            
            if is_identical:
                print("\n" + "!"*80)
                print("CRITICAL WARNING: LoRA output is EXACTLY IDENTICAL to Base model output.")
                print("The LoRA adapter weights are NOT having any effect.")
                print(f"Base Top 1: {list(base_dist.items())[0]}")
                print(f"LoRA Top 1: {list(lora_dist.items())[0]}")
                print("Possible causes:\n1. Adapter weights are zero/empty.\n2. Target modules mismatch.\n3. Adapter not loaded correctly.")
                print("!"*80 + "\n")
            else:
                print("\n" + "="*80)
                print("DEBUG: LoRA output differs from Base model. LoRA IS active.")
                print(f"Base Top 1: {list(base_dist.items())[0]}")
                print(f"LoRA Top 1: {list(lora_dist.items())[0]}")
                print("="*80 + "\n")
                
        except Exception as e:
            print(f"DEBUG: Failed to run LoRA verification check: {e}")
    # ------------------------------
    
    # Helper to chunk data
    def get_batches(lst, batch_size):
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]
    
    # Use a larger batch size for vLLM throughput if configured small
    # But for progress bar, we want frequent updates. user passed batch_size=16 probably.
    # vLLM handles internal batching, so sending 16 at a time is fine, just maybe slightly slower overhead than sending 1000.
    # We'll use the passed batch_size argument but ensure it's at least reasonable.
    # UPDATE: Using 32 causes too many sync barriers.
    # UPDATE 2: Passing all (120k) causes OOM (Killed).
    # UPDATE 3: 10k was fine, OOM was in evaluation.
    # Restore to 10000 for efficiency.
    eff_batch_size = max(batch_size, 10000) 
    
    total_batches = (len(prompts) + eff_batch_size - 1) // eff_batch_size
    
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
        
        for i, batch_prompts in tqdm(enumerate(get_batches(prompts, eff_batch_size)), total=total_batches, desc="Inference Batches"):
            # vLLM beam_search expects dict with "prompt" key or tokens
            batch_beam_prompts = [{"prompt": p} for p in batch_prompts]
            outputs = llm.beam_search(batch_beam_prompts, beam_params, lora_request=lora_request)
            
            # Map outputs back to global index
            global_start_idx = i * eff_batch_size
            for j, output in enumerate(outputs):
                # output is BeamSearchOutput, contains 'sequences' which is List[BeamSearchSequence]
                # BeamSearchSequence has 'text'
                generated_texts = [seq.text for seq in output.sequences]
                test_data[global_start_idx + j]['predict'] = generated_texts
            
    else:
        print(f"DEBUG: Using Sampling (greedy/sampling) with native progress bar")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
        )

        # Use batched generation to avoid OOM while maintaining high throughput
        if lora_request:
            print(f"DEBUG: Generating with LoRA adapter: {lora_request.lora_name}")
        else:
            print(f"DEBUG: Generating WITHOUT LoRA (lora_request is None)")

        for i, batch_prompts in tqdm(enumerate(get_batches(prompts, eff_batch_size)), total=total_batches, desc="Inference Batches"):
            outputs = llm.generate(batch_prompts, sampling_params, lora_request=lora_request)
            
            global_start_idx = i * eff_batch_size
            for j, output in enumerate(outputs):
                generated_texts = [o.text for o in output.outputs]
                test_data[global_start_idx + j]['predict'] = generated_texts

    print("DEBUG: Generation complete.")

    with open(output_path, 'w') as f:
        json.dump(test_data, f, indent=4)
    print(f"DEBUG: Results saved to {output_path}")

if __name__ == "__main__":
    fire.Fire(main)
