import fire
import json
import os
from vllm import LLM, SamplingParams
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
    limit: int = -1,
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
        if limit > 0:
            print(f"DEBUG: Limiting test data to first {limit} items.")
            test_data = test_data[:limit]
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
        # vLLM returns ONLY new tokens.
        # inference.py splits by 'Response:\n' because it decodes full sequence.
        # Since vLLM output excludes prompt, we don't need to split by 'Response:' 
        # BUT we might need to handle the prefix "the recommended game is " if the prompt forces it.
        # Our template ends with "the recommended game is {output}".
        # Actually, wait. format(output="") puts "the recommended game is " at the very end.
        # So vLLM will generate the TITLE directly.
        # inference.py's split logic: prompt...Response:\nthe recommended game is [Title]
        # split('Response:\n')[-1] -> "the recommended game is [Title]"
        # So inference.py output includes "the recommended game is ".
        # vLLM output (new tokens) will contain "[Title]".
        # We need to prepend "the recommended game is " to match inference.py's output format?
        # Let's check inference.py line 142: output = [_.split('Response:\n')[-1] for _ in output]
        # If prompt ends with "Response:\nthe recommended game is ", then split('Response:\n') gives "the recommended game is [Title]".
        # So yes, inference.py output HAS the prefix.
        # vLLM output DOES NOT.
        # So we MUST prepend "the recommended game is " to vLLM output to be equivalent.
        
        # However, checking prompt_template.txt: "the recommended game is {output}"
        # If output="" -> "the recommended game is "
        # vLLM prompt ends there.
        # vLLM generates "Super Mario Bros. 3"
        # inference.py generates "the recommended game is Super Mario Bros. 3" (as part of full text)
        # inference.py splits at "Response:\n", so it gets "the recommended game is Super Mario Bros. 3".
        
        # So I will prepend the prefix here.
        generated_texts = ["the recommended game is " + o.text for o in output.outputs]
        test_data[i]['predict'] = generated_texts

    print("DEBUG: Generation complete.")

    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)
    print(f"DEBUG: Results saved to {result_json_data}")

if __name__ == "__main__":
    fire.Fire(main)
