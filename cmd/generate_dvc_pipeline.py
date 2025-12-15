import pandas as pd
import yaml
import os
import argparse

def sanitize(s):
    return str(s).replace('/', '_').replace('.', 'p')

def generate_pipeline(csv_path):
    print(f"Reading pipeline configuration from {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Split by GPUID if present
    if 'GPUID' in df.columns:
        df['GPUID'] = df['GPUID'].fillna(0).astype(int)
        gpu_groups = df.groupby('GPUID')
    else:
        df['GPUID'] = 0
        gpu_groups = [(0, df)]

    for gpu_id, group in gpu_groups:
        dvc_stages = {}
        print(f"Generating pipeline for GPU {gpu_id} with {len(group)} experiments...")
        
        for idx, row in group.iterrows():
            # Extract parameters
            dataset = str(row['dataset_name'])
            seed = int(row['seed'])
            alpha = float(row['alpha'])
            strategy = str(row['sampling_strategy'])
            sample_num = int(row['sample_num'])
            al_ratio = float(row['al_ratio'])
            base_model = str(row['base_model_name'])
            # template = row['templete'] # Not used yet in scripts? or passed as PROMPT_FILE? 
            # run_bigrec_train.sh takes PROMPT_FILE as arg 9.
            # Let's assume template maps to a file or empty.
            template = row.get('templete', '')
            if pd.isna(template): template = ""
            
            ed_weight = float(row['ed_weight'])
            lam = float(row['lambda'])
            
            # Derived names
            safe_base_model = base_model.replace('/', '_')
            alpha_str = str(alpha)
            
            # Paths (Relative to Repo Root)
            exp_root = f"experiments/{dataset}"
            
            # Helper to adjust paths for DVC deps/outs (relative to verify file location)
            # Since dvc.yaml is in pipelines/gpuX/, we need to go up 2 levels.
            def to_dvc_path(p):
                return f"../../{p}"

            # === Step 1: SASRec ===
            sasrec_dir = f"{exp_root}/sasrec/seed_{seed}/alpha_{alpha_str}"
            sasrec_stage_name = f"sasrec_{dataset}_{seed}_{sanitize(alpha)}"
            
            if sasrec_stage_name not in dvc_stages:
                dvc_stages[sasrec_stage_name] = {
                    "cmd": f"OUTPUT_DIR={sasrec_dir} ../../cmd/run_sasrec_baseline.sh {dataset} {gpu_id} 200 {seed} {alpha}",
                    "deps": [to_dvc_path("cmd/run_sasrec_baseline.sh"), to_dvc_path("DLLM2Rec/main.py")],
                    "outs": [to_dvc_path(f"{sasrec_dir}/train.pt"), to_dvc_path(f"{sasrec_dir}/train_uids.pt")],
                    "wdir": "../.."
                }

            # === Step 2: Active Learning Data ===
            # Depends on SASRec if method uses it.
            # Methods using DROS: loss, entropy, error_rank, proximal_rank, semantic_loss, confident_error
            dros_methods = ["loss", "entropy", "error_rank", "proximal_rank", "semantic_loss", "confident_error"]
            deps_al = [to_dvc_path("cmd/create_active_learning_data.sh")]
            if strategy in dros_methods:
                deps_al.append(to_dvc_path(f"{sasrec_dir}/train.pt"))
                dros_source_arg = f"{sasrec_dir}" # Script adds /train.pt
            else:
                dros_source_arg = ""

            al_suffix = f"{strategy}_{sample_num}_{al_ratio}_seed_{seed}_alpha_{alpha_str}"
            al_file = f"{exp_root}/active_learning/{al_suffix}.json"
            al_stage_name = f"al_data_{sanitize(al_suffix)}"
            
            if al_stage_name not in dvc_stages:
                dvc_stages[al_stage_name] = {
                    "cmd": f"OUTPUT_JSON={al_file} ../../cmd/create_active_learning_data.sh {dataset} {strategy} {sample_num} {al_ratio} {seed} 1024 {dros_source_arg}",
                    "deps": deps_al,
                    "outs": [to_dvc_path(al_file)],
                    "wdir": "../.."
                }
                
            # === Step 3: BIGRec Training ===
            bigrec_train_dir = f"{exp_root}/{safe_base_model}/bigrec_train/{al_suffix}"
            bigrec_train_stage_name = f"bigrec_train_{sanitize(al_suffix)}"
            
            prompt_arg = template if template else ""
            
            if bigrec_train_stage_name not in dvc_stages:
                dvc_stages[bigrec_train_stage_name] = {
                    "cmd": f"OUTPUT_DIR={bigrec_train_dir} ../../cmd/run_bigrec_train.sh {dataset} {gpu_id} {seed} -1 128 16 {base_model} 50 \"{prompt_arg}\" \"{al_file}\"",
                    "deps": [to_dvc_path("cmd/run_bigrec_train.sh"), to_dvc_path(al_file)],
                    "outs": [to_dvc_path(bigrec_train_dir)], # It's a directory
                    "wdir": "../.."
                }

            # === Step 4: BIGRec Inference ===
            bigrec_infer_dir = f"{exp_root}/{safe_base_model}/bigrec_infer_train/{al_suffix}"
            bigrec_infer_stage_name = f"bigrec_infer_{sanitize(al_suffix)}"
            
            sasrec_res_path = sasrec_dir 
            
            if bigrec_infer_stage_name not in dvc_stages:
                dvc_stages[bigrec_infer_stage_name] = {
                    "cmd": f"RESULT_DIR={bigrec_infer_dir} ../../cmd/run_bigrec_inference_vllm.sh --dataset {dataset} --gpu {gpu_id} --model {base_model} --seed {seed} --sample -1 --checkpoint best --test_data train.json --correction ci --resource {sasrec_res_path} --lora_weights {bigrec_train_dir}",
                    "deps": [to_dvc_path(bigrec_train_dir), to_dvc_path(sasrec_dir), to_dvc_path("cmd/run_bigrec_inference_vllm.sh")],
                    "outs": [to_dvc_path(f"{bigrec_infer_dir}/train_epoch_best.json")], # Script outputs .json
                    "wdir": "../.."
                }
                
            # === Step 5: DLLM2Rec Training ===
            dllm2rec_dir = f"{exp_root}/{safe_base_model}/dllm2rec_final/{al_suffix}/ed_{ed_weight}_lam_{lam}"
            dllm2rec_stage_name = f"dllm2rec_{sanitize(al_suffix)}_ed{ed_weight}_lam{lam}"
            
            embedding_path = f"BIGRec/data/{dataset}/model_embeddings/{safe_base_model}.pt" # Standard loc
            ranking_path = f"{bigrec_infer_dir}/train_epoch_best_rank.txt"
            confidence_path = f"{bigrec_infer_dir}/train_epoch_best_score.txt"
            
            if dllm2rec_stage_name not in dvc_stages:
                dvc_stages[dllm2rec_stage_name] = {
                    "cmd": f"OUTPUT_DIR={dllm2rec_dir} RANKING_PATH={ranking_path} CONFIDENCE_PATH={confidence_path} EMBEDDING_PATH={embedding_path} ../../cmd/run_dllm2rec_train.sh {dataset} SASRec {gpu_id} {ed_weight} {lam}",
                    "deps": [to_dvc_path(ranking_path), to_dvc_path(confidence_path), to_dvc_path("cmd/run_dllm2rec_train.sh")],
                    "outs": [to_dvc_path(f"{dllm2rec_dir}/metrics.json")],
                    "wdir": "../.."
                }
                
        # Write DVC file
        filename = f"pipelines/gpu{gpu_id}/dvc.yaml"
        print(f"Writing {filename}...")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            yaml.dump({"stages": dvc_stages}, f, sort_keys=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', nargs='?', default='pipeline_params.csv', help='Path to params CSV')
    args = parser.parse_args()
    generate_pipeline(args.csv_path)
