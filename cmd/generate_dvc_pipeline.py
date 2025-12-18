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

    dvc_stages = {}

    for gpu_id, group in gpu_groups:
        print(f"Generating pipeline stages for GPU {gpu_id} with {len(group)} experiments...")
        
        for idx, row in group.iterrows():
            # Extract parameters
            dataset = str(row['dataset_name'])
            seed = int(row['seed'])
            alpha = float(row['alpha'])
            strategy = str(row['sampling_strategy'])
            sample_num = int(row['sample_num'])
            al_ratio = float(row['al_ratio'])
            base_model = str(row['base_model_name'])
            template = row.get('templete', '')
            if pd.isna(template): template = ""
            
            ed_weight = float(row['ed_weight'])
            lam = float(row['lambda'])
            
            # Derived names
            safe_base_model = base_model.replace('/', '_')
            alpha_str = str(alpha)
            
            # Paths (Relative to Repo Root)
            exp_root = f"experiments/{dataset}"

            # Suffix for separation
            suffix_gpu = f"gpu{gpu_id}"

            # === Step 1: SASRec ===
            sasrec_dir = f"{exp_root}/sasrec/seed_{seed}/alpha_{alpha_str}"
            # Add GPU suffix to stage name
            sasrec_stage_name = f"sasrec_{dataset}_{seed}_{sanitize(alpha)}_{suffix_gpu}"
            
            if sasrec_stage_name not in dvc_stages:
                dvc_stages[sasrec_stage_name] = {
                    "cmd": f"OUTPUT_DIR={sasrec_dir} ./cmd/run_sasrec_baseline.sh {dataset} {gpu_id} 200 {seed} {alpha}",
                    "deps": ["cmd/run_sasrec_baseline.sh", "DLLM2Rec/main.py"],
                    "outs": [f"{sasrec_dir}/train.pt", f"{sasrec_dir}/train_uids.pt"]
                }

            # === Step 2: Active Learning Data ===
            dros_methods = ["loss", "entropy", "error_rank", "proximal_rank", "semantic_loss", "confident_error"]
            deps_al = ["cmd/create_active_learning_data.sh"]
            if strategy in dros_methods:
                deps_al.append(f"{sasrec_dir}/train.pt")
                dros_source_arg = f"{sasrec_dir}" 
            else:
                dros_source_arg = ""

            al_suffix = f"{strategy}_{sample_num}_{al_ratio}_seed_{seed}_alpha_{alpha_str}"
            al_file = f"{exp_root}/active_learning/{al_suffix}.json"
            al_stage_name = f"al_data_{sanitize(al_suffix)}_{suffix_gpu}"
            
            if al_stage_name not in dvc_stages:
                dvc_stages[al_stage_name] = {
                    "cmd": f"OUTPUT_JSON={al_file} ./cmd/create_active_learning_data.sh {dataset} {strategy} {sample_num} {al_ratio} {seed} 1024 {dros_source_arg}",
                    "deps": deps_al,
                    "outs": [al_file]
                }
                
            # === Step 3: BIGRec Training ===
            bigrec_train_dir = f"{exp_root}/{safe_base_model}/bigrec_train/{al_suffix}"
            bigrec_train_stage_name = f"bigrec_train_{sanitize(al_suffix)}_{suffix_gpu}"
            
            prompt_arg = template if template else ""
            
            if bigrec_train_stage_name not in dvc_stages:
                if sample_num == 0:
                     # Skip training
                     pass
                else:
                    dvc_stages[bigrec_train_stage_name] = {
                        "cmd": f"OUTPUT_DIR={bigrec_train_dir} ./cmd/run_bigrec_train.sh {dataset} {gpu_id} {seed} -1 128 16 {base_model} 50 \"{prompt_arg}\" \"{al_file}\"",
                        "deps": ["cmd/run_bigrec_train.sh", al_file],
                        "outs": [bigrec_train_dir] 
                    }

            # === Step 4: BIGRec Inference ===
            bigrec_infer_dir = f"{exp_root}/{safe_base_model}/bigrec_infer_train/{al_suffix}"
            bigrec_infer_stage_name = f"bigrec_infer_{sanitize(al_suffix)}_{suffix_gpu}"
            
            sasrec_res_path = sasrec_dir 
            
            if bigrec_infer_stage_name not in dvc_stages:
                if sample_num == 0:
                     # Vanilla Mode
                     bigrec_infer_out = f"{bigrec_infer_dir}/train_vanilla.json"
                     # Deps exclude train dir
                     dvc_stages[bigrec_infer_stage_name] = {
                        "cmd": f"RESULT_DIR={bigrec_infer_dir} ./cmd/run_bigrec_inference_vllm.sh --dataset {dataset} --gpu {gpu_id} --model {base_model} --seed {seed} --sample -1 --checkpoint best --test_data all --correction ci --resource {sasrec_res_path} --no_adapter",
                        "deps": [sasrec_dir, "cmd/run_bigrec_inference_vllm.sh"],
                        "outs": [bigrec_infer_out] 
                    }
                else:
                    dvc_stages[bigrec_infer_stage_name] = {
                        "cmd": f"RESULT_DIR={bigrec_infer_dir} ./cmd/run_bigrec_inference_vllm.sh --dataset {dataset} --gpu {gpu_id} --model {base_model} --seed {seed} --sample -1 --checkpoint best --test_data all --correction ci --resource {sasrec_res_path} --lora_weights {bigrec_train_dir}",
                        "deps": [bigrec_train_dir, sasrec_dir, "cmd/run_bigrec_inference_vllm.sh"],
                        "outs": [f"{bigrec_infer_dir}/train_epoch_best.json"] 
                    }
                
            # === Step 5: DLLM2Rec Training ===
            dllm2rec_dir = f"{exp_root}/{safe_base_model}/dllm2rec_final/{al_suffix}/ed_{ed_weight}_lam_{lam}"
            dllm2rec_stage_name = f"dllm2rec_{sanitize(al_suffix)}_ed{ed_weight}_lam{lam}_{suffix_gpu}"
            
            embedding_path = f"BIGRec/data/{dataset}/model_embeddings/{safe_base_model}.pt" 
            ranking_path = f"{bigrec_infer_dir}/train_epoch_best_rank.txt"
            confidence_path = f"{bigrec_infer_dir}/train_epoch_best_score.txt"
            
            if dllm2rec_stage_name not in dvc_stages:
                if sample_num == 0:
                    ranking_path = f"{bigrec_infer_dir}/train_vanilla_rank.txt"
                    confidence_path = f"{bigrec_infer_dir}/train_vanilla_score.txt"
                    epoch_arg = "vanilla"
                else:
                    ranking_path = f"{bigrec_infer_dir}/train_epoch_best_rank.txt"
                    confidence_path = f"{bigrec_infer_dir}/train_epoch_best_score.txt"
                    epoch_arg = "best"
                    
                dvc_stages[dllm2rec_stage_name] = {
                    "cmd": f"OUTPUT_DIR={dllm2rec_dir} RANKING_PATH={ranking_path} CONFIDENCE_PATH={confidence_path} EMBEDDING_PATH={embedding_path} ./cmd/run_dllm2rec_train.sh {dataset} SASRec {gpu_id} {ed_weight} {lam} '' {seed} 1024 {epoch_arg}",
                    "deps": [ranking_path, confidence_path, "cmd/run_dllm2rec_train.sh"],
                    "outs": [f"{dllm2rec_dir}/metrics.json"]
                }
                
    # Write DVC file
    filename = "dvc.yaml"
    print(f"Writing {filename} with {len(dvc_stages)} stages...")
    with open(filename, 'w') as f:
        yaml.dump({"stages": dvc_stages}, f, sort_keys=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', nargs='?', default='pipeline_params.csv', help='Path to params CSV')
    args = parser.parse_args()
    generate_pipeline(args.csv_path)
