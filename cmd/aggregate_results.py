import os
import json
import pandas as pd
import glob
import re

def aggregate_results(root_dir="experiments", output_file="experiments/summary.csv"):
    print(f"Aggregating results from {root_dir}...")
    
    results = []
    
    # Pattern: experiments/{dataset}/{base_model}/dllm2rec_final/{al_suffix}/ed_{ed}_lam_{lam}/metrics.json
    # al_suffix: {strategy}_{sample_num}_{al_ratio}_seed_{seed}_alpha_{alpha}
    
    # We can use recursive glob
    pattern = os.path.join(root_dir, "**", "metrics.json")
    files = glob.glob(pattern, recursive=True)
    
    for f in files:
        try:
            with open(f, 'r') as json_file:
                data = json.load(json_file)
            
            # Helper to Flatten
            flat_data = {}
            for key, val in data.items():
                if isinstance(val, dict):
                    for subkey, subval in val.items():
                        flat_data[f"{key}_{subkey}"] = subval
                else:
                    flat_data[key] = val
                    
            # Parse Path for Params
            path_parts = f.split(os.sep)
            # This is fragile if path structure changes, but works for our specified structure.
            # Assuming strictly: experiments/{dataset}/{base_model}/dllm2rec_final/{al_suffix}/ed_{ed}_lam_{lam}/metrics.json
            
            # Let's try to extract known markers
            # ed_..._lam_... -> last dir
            parent = os.path.dirname(f)
            dirname = os.path.basename(parent)
            
            ed_lam_match = re.search(r"ed_([\d\.]+)_lam_([\d\.]+)", dirname)
            if ed_lam_match:
                flat_data["ed_weight"] = float(ed_lam_match.group(1))
                flat_data["lambda"] = float(ed_lam_match.group(2))
                
            # AL suffix -> parent of parent
            grandparent = os.path.dirname(parent) # dllm2rec_final usually?
            # Wait, structure: .../dllm2rec_final/{al_suffix}/ed...
            # So grandparent is {al_suffix}.
            al_dir = os.path.basename(grandparent)
            
            # al_suffix format: {strategy}_{sample_num}_{al_ratio}_seed_{seed}_alpha_{alpha}
            # This is complex to parse with regex if strategy contains underscore. 
            # But standard active learning strategies (random, loss) don't have underscores usually? 
            # Except "pop_inverse", "error_rank", "proximal_rank", "semantic_loss", "confident_error".
            # They HAVE underscores!
            # Strategies: random, pop_inverse, etc.
            # Regex approach: Look for known tail patterns. _seed_{} _alpha_{}
            # ... _seed_(\d+)_alpha_([\d\.]+)
            
            params_match = re.search(r"(.*)_(\d+)_([\d\.]+)_seed_(\d+)_alpha_([\d\.]+)", al_dir)
            if params_match:
                flat_data["sampling_strategy"] = params_match.group(1)
                flat_data["sample_num"] = int(params_match.group(2))
                flat_data["al_ratio"] = float(params_match.group(3))
                flat_data["seed"] = int(params_match.group(4))
                flat_data["alpha"] = float(params_match.group(5))
            else:
                # Fallback or different structure
                flat_data["path_params_raw"] = al_dir
                
            # Dataset and Model
            # .../experiments/{dataset}/{base_model}/...
            # We can traverse up until we hit "experiments" or root_dir
            rel_path = os.path.relpath(f, root_dir)
            rel_parts = rel_path.split(os.sep)
            if len(rel_parts) >= 2:
                flat_data["dataset_name"] = rel_parts[0]
                flat_data["base_model"] = rel_parts[1]
                
            results.append(flat_data)
            
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Aggregated results saved to {output_file}")
        print(df.head())
    else:
        print("No results found.")

if __name__ == "__main__":
    aggregate_results()
