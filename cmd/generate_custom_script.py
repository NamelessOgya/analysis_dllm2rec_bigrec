import pandas as pd
import os
import argparse
import stat

def sanitize(s):
    return str(s).replace('/', '_').replace('.', 'p')

def generate_bash_pipeline(csv_path):
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
        script_filename = f"pipeline_gpu{gpu_id}.sh"
        print(f"Generating {script_filename} with {len(group)} experiments...")
        
        with open(script_filename, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write("# Auto-generated pipeline script. Skips steps if output exists.\n")
            f.write("set -e\n\n")
            
            f.write(f"REPO_ROOT=$(pwd)\n")
            
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
                
                exp_root = f"${{REPO_ROOT}}/experiments/{dataset}"

                f.write(f"echo \"----------------------------------------------------------------\"\n")
                f.write(f"echo 'Starting experiment for {dataset} seed={seed} alpha={alpha} strategy={strategy} ...'\n")

                # === Step 1: SASRec ===
                sasrec_dir = f"{exp_root}/sasrec/seed_{seed}/alpha_{alpha_str}"
                # Primary output to check
                sasrec_out = f"{sasrec_dir}/train.pt"
                sasrec_cmd = f"OUTPUT_DIR={sasrec_dir} ./cmd/run_sasrec_baseline.sh {dataset} {gpu_id} 200 {seed} {alpha}"
                
                f.write(f"\n# --- Step 1: SASRec ---\n")
                f.write(f"if [ -e \"{sasrec_out}\" ]; then\n")
                f.write(f"    echo \"Skipping SASRec (Output exists: {sasrec_out})\"\n")
                f.write(f"else\n")
                f.write(f"    echo \"Running SASRec...\"\n")
                f.write(f"    {sasrec_cmd}\n")
                f.write(f"fi\n")

                # === Step 2: Active Learning Data ===
                dros_methods = ["loss", "entropy", "error_rank", "proximal_rank", "semantic_loss", "confident_error"]
                if strategy in dros_methods:
                    dros_source_arg = f"{sasrec_dir}" 
                else:
                    dros_source_arg = ""

                al_suffix = f"{strategy}_{sample_num}_{al_ratio}_seed_{seed}_alpha_{alpha_str}"
                al_file = f"{exp_root}/active_learning/{al_suffix}.json"
                al_cmd = f"OUTPUT_JSON={al_file} ./cmd/create_active_learning_data.sh {dataset} {strategy} {sample_num} {al_ratio} {seed} 1024 {dros_source_arg}"

                f.write(f"\n# --- Step 2: AL Data ---\n")
                f.write(f"if [ -e \"{al_file}\" ]; then\n")
                f.write(f"    echo \"Skipping AL Data Gen (Output exists: {al_file})\"\n")
                f.write(f"else\n")
                f.write(f"    echo \"Running AL Data Gen...\"\n")
                f.write(f"    {al_cmd}\n")
                f.write(f"fi\n")
                
                # === Step 3: BIGRec Training ===
                bigrec_train_dir = f"{exp_root}/{safe_base_model}/bigrec_train/{al_suffix}"
                prompt_arg = f"templates/{template}" if template else ""
                
                bigrec_train_cmd = f"OUTPUT_DIR={bigrec_train_dir} ./cmd/run_bigrec_train.sh {dataset} {gpu_id} {seed} -1 128 16 {base_model} 50 \"{prompt_arg}\" \"{al_file}\""

                f.write(f"\n# --- Step 3: BIGRec Train ---\n")
                if sample_num == 0:
                    f.write(f"echo \"Skipping BIGRec Train (Vanilla Mode: sample_num=0)\"\n")
                else:
                    f.write(f"if [ -d \"{bigrec_train_dir}\" ] && [ \"$(ls -A {bigrec_train_dir})\" ]; then\n")
                    f.write(f"    echo \"Skipping BIGRec Train (Output dir exists and not empty: {bigrec_train_dir})\"\n")
                    f.write(f"else\n")
                    f.write(f"    echo \"Running BIGRec Train...\"\n")
                    f.write(f"    {bigrec_train_cmd}\n")
                    f.write(f"fi\n")

                # === Step 4: BIGRec Inference ===
                bigrec_infer_dir = f"{exp_root}/{safe_base_model}/bigrec_infer_train/{al_suffix}"
                sasrec_res_path = sasrec_dir 
                
                if sample_num == 0:
                     # Vanilla Mode: No adapter, result filename changes
                     bigrec_infer_out = f"{bigrec_infer_dir}/train_vanilla.json"
                     # Add --no_adapter, Remove lora_weights (or pass empty/dummy if logic handles it, but better explicit)
                     # Shell script handles --no_adapter which implies empty lora.
                     # But we must construct command carefully.
                     bigrec_infer_cmd = f"RESULT_DIR={bigrec_infer_dir} ./cmd/run_bigrec_inference_vllm.sh --dataset {dataset} --gpu {gpu_id} --model {base_model} --seed {seed} --sample -1 --checkpoint best --test_data all --correction ci --resource {sasrec_res_path} --no_adapter"
                else:
                     bigrec_infer_out = f"{bigrec_infer_dir}/train_epoch_best.json"
                     bigrec_infer_cmd = f"RESULT_DIR={bigrec_infer_dir} ./cmd/run_bigrec_inference_vllm.sh --dataset {dataset} --gpu {gpu_id} --model {base_model} --seed {seed} --sample -1 --checkpoint best --test_data all --correction ci --resource {sasrec_res_path} --lora_weights {bigrec_train_dir}"

                f.write(f"\n# --- Step 4: BIGRec Inference ---\n")
                f.write(f"if [ -e \"{bigrec_infer_out}\" ]; then\n")
                f.write(f"    echo \"Skipping BIGRec Inference (Output exists: {bigrec_infer_out})\"\n")
                f.write(f"else\n")
                f.write(f"    echo \"Running BIGRec Inference...\"\n")
                f.write(f"    {bigrec_infer_cmd}\n")
                f.write(f"fi\n")
                
                # === Step 5: DLLM2Rec Training ===
                dllm2rec_dir = f"{exp_root}/{safe_base_model}/dllm2rec_final/{al_suffix}/ed_{ed_weight}_lam_{lam}"
                embedding_path = f"BIGRec/data/{dataset}/model_embeddings/{safe_base_model}.pt" 
                
                if sample_num == 0:
                     ranking_path = f"{bigrec_infer_dir}/train_vanilla_rank.txt"
                     confidence_path = f"{bigrec_infer_dir}/train_vanilla_score.txt"
                     epoch_arg = "vanilla"
                else:
                     ranking_path = f"{bigrec_infer_dir}/train_epoch_best_rank.txt"
                     confidence_path = f"{bigrec_infer_dir}/train_epoch_best_score.txt"
                     epoch_arg = "best"
                     
                dllm2rec_out = f"{dllm2rec_dir}/metrics.json"

                dllm2rec_cmd = f"OUTPUT_DIR={dllm2rec_dir} RANKING_PATH={ranking_path} CONFIDENCE_PATH={confidence_path} EMBEDDING_PATH={embedding_path} ./cmd/run_dllm2rec_train.sh {dataset} SASRec {gpu_id} {ed_weight} {lam} '' {seed} 1024 {epoch_arg}"

                f.write(f"\n# --- Step 5: DLLM2Rec Train ---\n")
                f.write(f"if [ -e \"{dllm2rec_out}\" ]; then\n")
                f.write(f"    echo \"Skipping DLLM2Rec Train (Output exists: {dllm2rec_out})\"\n")
                f.write(f"else\n")
                f.write(f"    echo \"Running DLLM2Rec Train...\"\n")
                f.write(f"    {dllm2rec_cmd}\n")
                f.write(f"fi\n")
                
                f.write("\n")

        # Make script executable
        st = os.stat(script_filename)
        os.chmod(script_filename, st.st_mode | stat.S_IEXEC)
        print(f"Created executable {script_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', nargs='?', default='pipeline_params.csv', help='Path to params CSV')
    args = parser.parse_args()
    generate_bash_pipeline(args.csv_path)
