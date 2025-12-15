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
            
            # === Step 1: SASRec ===
            sasrec_dir = f"{exp_root}/sasrec/seed_{seed}/alpha_{alpha_str}"
            sasrec_stage_name = f"sasrec_{dataset}_{seed}_{sanitize(alpha)}"
            
            if sasrec_stage_name not in dvc_stages:
                dvc_stages[sasrec_stage_name] = {
                    "cmd": f"OUTPUT_DIR={sasrec_dir} ./cmd/run_sasrec_baseline.sh {dataset} {gpu_id} 200 {seed} {alpha}",
                    "deps": ["cmd/run_sasrec_baseline.sh", "DLLM2Rec/main.py"],
                    "outs": [f"{sasrec_dir}/train.pt", f"{sasrec_dir}/train_uids.pt"]
                }

            # === Step 2: Active Learning Data ===
            # Depends on SASRec if method uses it.
            # Methods using DROS: loss, entropy, error_rank, proximal_rank, semantic_loss, confident_error
            dros_methods = ["loss", "entropy", "error_rank", "proximal_rank", "semantic_loss", "confident_error"]
            deps_al = ["cmd/create_active_learning_data.sh"]
            if strategy in dros_methods:
                deps_al.append(f"{sasrec_dir}/train.pt")
                dros_source_arg = f"{sasrec_dir}" # Script adds /train.pt
            else:
                dros_source_arg = ""

            al_suffix = f"{strategy}_{sample_num}_{al_ratio}_seed_{seed}_alpha_{alpha_str}"
            al_file = f"{exp_root}/active_learning/{al_suffix}.json"
            al_stage_name = f"al_data_{sanitize(al_suffix)}"
            
            if al_stage_name not in dvc_stages:
                dvc_stages[al_stage_name] = {
                    "cmd": f"OUTPUT_JSON={al_file} ./cmd/create_active_learning_data.sh {dataset} {strategy} {sample_num} {al_ratio} {seed} 1024 {dros_source_arg}",
                    "deps": deps_al,
                    "outs": [al_file]
                }
                
            # === Step 3: BIGRec Training ===
            # Output dir: experiments/{dataset}/{base_model}/bigrec_train/{al_suffix}
            # We assume bigrec training depends on the specific AL data.
            # run_bigrec_train.sh <dataset> <gpu> <seed> <sample> ... <train_file> <suffix>
            # We use sample=-1 (all) as per instructions for AL data.
            # Suffix is part of path now, so maybe empty suffix?
            # Script uses suffix to name output inside its logic, but we override output dir.
            
            bigrec_train_dir = f"{exp_root}/{safe_base_model}/bigrec_train/{al_suffix}"
            bigrec_train_stage_name = f"bigrec_train_{sanitize(al_suffix)}"
            
            # run_bigrec_train args: data, gpu, seed, sample(-1), batch(128), ubatch(4), model, epochs(50), prompt, train_file
            # Note: positions: 1=DATA, 2=GPU, 3=SEED, 4=SAMPLE, 5=BATCH, 6=UBATCH, 7=MODEL, 8=EPOCH, 9=PROMPT, 10=TRAIN_FILE
            
            prompt_arg = template if template else ""
            
            if bigrec_train_stage_name not in dvc_stages:
                dvc_stages[bigrec_train_stage_name] = {
                    "cmd": f"OUTPUT_DIR={bigrec_train_dir} ./cmd/run_bigrec_train.sh {dataset} {gpu_id} {seed} -1 128 16 {base_model} 50 \"{prompt_arg}\" \"{al_file}\"",
                    "deps": ["cmd/run_bigrec_train.sh", al_file],
                    "outs": [bigrec_train_dir] # It's a directory
                }

            # === Step 4: BIGRec Inference ===
            # Depends on BIGRec Train.
            # Output: experiments/{dataset}/{base_model}/bigrec_infer/{al_suffix}
            # Script expects CHECKPOINT, TEST_DATA, etc.
            # Default test data is test_5000.json? Or train.json for distillation?
            # The pipeline flow says: "BIGRec Inference (on Training Data) -> Step 2" in run_pipeline.py.
            # Step 12 in readme says: "BIGRec Inference (Train Data) ... for Distillation".
            # So we need TWO inferences? Or just the one for distillation?
            # User requirement: "experiments efficient... ./cmd/inference_vllm.sh".
            # Usually we need inference on Test (for metrics) and Train (for distillation).
            # But the final step is DLLM2Rec Train, which needs Train scores.
            # So we MUST run inference on Train.
            # We can also run on Test if needed.
            # Let's run on Train as primary dependency for DLLM2Rec.
            
            bigrec_infer_dir = f"{exp_root}/{safe_base_model}/bigrec_infer_train/{al_suffix}"
            bigrec_infer_stage_name = f"bigrec_infer_{sanitize(al_suffix)}"
            
            # run_bigrec_inference_vllm.sh
            # Needs --result_dir override.
            # Needs --lora_weights pointing to bigrec_train_dir
            # Needs --test_data train.json
            # Needs --correction ci? Or none? Readme says "correction ci" for distillation in step 11.
            # But wait, step 11 says "BIGRec Inference (Train Data) ... --correction ci".
            # And "resource .../sasrec...".
            # So we need SASRec results again here!
            
            sasrec_res_path = sasrec_dir # Where train.pt is
            
            # Verify if correction is needed?
            # Step 11 says "Use SASRec scores (CI) ...".
            # So we enable CI correction using SASRec output.
            
            # Note: Base model for inference should be base_model.
            # We assume "train.json" is in BIGRec/data/{dataset}/train.json (the original one)?
            # Or the AL data one?
            # Distillation usually uses the FULL original train set to label it.
            # "BIGRec Inference (Train Data) ... limit -1".
            # If we use AL data to train, do we infer on AL data or Full data?
            # Usually we want to distill predictions for the WHOLE dataset (or subset) to the student.
            # Step 11: "--test_data train.json". This is standard train.json.
            
            if bigrec_infer_stage_name not in dvc_stages:
                dvc_stages[bigrec_infer_stage_name] = {
                    "cmd": f"RESULT_DIR={bigrec_infer_dir} ./cmd/run_bigrec_inference_vllm.sh --dataset {dataset} --gpu {gpu_id} --model {base_model} --seed {seed} --sample -1 --checkpoint best --test_data train.json --correction ci --resource {sasrec_res_path} --lora_weights {bigrec_train_dir}",
                    # Note: We manually passing lora_weights via modification?
                    # Wait, run_bigrec_inference_vllm.sh calculates LORA_WEIGHTS based on defaults.
                    # We modified it to accept RESULT_DIR.
                    # But LORA_WEIGHTS calculation logic in line 87: BASE_LORA_PATH="BIGRec/model/..."
                    # We moved the model output to {bigrec_train_dir}.
                    # So the script will FAIL to find lora weights unless we override LORA_WEIGHTS env var or path.
                    # Use "LORA_WEIGHTS={bigrec_train_dir}"? The script doesn't support LORA_WEIGHTS override yet?
                    # Let's check script.
                    # "LORA_WEIGHTS=$BASE_LORA_PATH".
                    # I need to modify run_bigrec_inference_vllm.sh to accept LORA_WEIGHTS override.
                    # I will add this to "cmd" but first I must support it in script.
                    # I'll do this later or assume I can pass it.
                    # Actually, let's look at `cmd/run_bigrec_inference_vllm.sh` content again.
                    "deps": [bigrec_train_dir, sasrec_dir, "cmd/run_bigrec_inference_vllm.sh"],
                    "outs": [f"{bigrec_infer_dir}/train_epoch_best.json"] # Script outputs .json
                }
                
                # I'll assume I'll fix the script to accept LORA_WEIGHTS override.
                
            # === Step 5: DLLM2Rec Training ===
            # Depends on Inference result (ranking/score).
            # Inferred result is in {bigrec_infer_dir}.
            # Expected files: train_epoch_best_rank.txt / score.txt. (Inference script produces these?)
            # Wait, inference produces JSON. Distillation needs txt ranking/score?
            # Step 11 says: "Generated train_epoch_best_rank.txt ... use to run_dllm2rec_train.sh".
            # So inference VLLM script MUST produce .txt inputs for DLLM2Rec.
            # `cmd/run_bigrec_inference_vllm.sh` usually runs `evaluate.py`.
            # `evaluate.py` with `--save_results` and `train.json` should produce rank/score?
            # Let's assume it does.
            # Output Dir: experiments/{dataset}/{base_model}/dllm2rec_final/.../
            
            dllm2rec_dir = f"{exp_root}/{safe_base_model}/dllm2rec_final/{al_suffix}/ed_{ed_weight}_lam_{lam}"
            dllm2rec_stage_name = f"dllm2rec_{sanitize(al_suffix)}_ed{ed_weight}_lam{lam}"
            
            # run_dllm2rec_train.sh args: dataset, model(SASRec), gpu, ed, lam, teacher_model, seed, sample, epoch
            # It also sets input paths based on standard structure.
            # Since we moved files, we must pass explicit paths.
            # Script has specific logic: "if [ -n "$BIGREC_BASE_MODEL" ]; then ... use direct paths".
            # But the direct paths are hardcoded to BIGRec/results/...
            # We need to override input paths.
            # `run_dllm2rec_train.sh` supports `EXTRA_ARGS`.
            # We can pass `--embedding_path`, `--ranking_path`, `--confidence_path` via EXTRA_ARGS?
            # script sets EXTRA_ARGS.
            # We should probably pass them directly to `run_dllm2rec_train.sh` if we modify it to accept explicit input args?
            # Or just rely on passing them to `python main.py` via some way.
            # The script logic at line 27 uses `BIGREC_BASE_MODEL` to set paths.
            # If we don't pass `BIGREC_BASE_MODEL` (pass empty), we can manually clean up EXTRA_ARGS?
            # No, `run_dllm2rec_train.sh` is a wrapper.
            # I can just call python main.py directly in DVC? No, requirement says "run ./cmd/run_dllm2rec_train.sh".
            # I should modify `run_dllm2rec_train.sh` to accept `--ranking_path` etc overrides?
            # It blindly sets them.
            # I'll modify `run_dllm2rec_train.sh` to specific input override variables.
            
            # For now, let's write the DVC entry assuming I can pass args.
            # We will use variables `RANKING_PATH`, `CONFIDENCE_PATH`, `EMBEDDING_PATH`.
            
            embedding_path = f"BIGRec/data/{dataset}/model_embeddings/{safe_base_model}.pt" # Standard loc
            ranking_path = f"{bigrec_infer_dir}/train_epoch_best_rank.txt"
            confidence_path = f"{bigrec_infer_dir}/train_epoch_best_score.txt"
            
            if dllm2rec_stage_name not in dvc_stages:
                dvc_stages[dllm2rec_stage_name] = {
                    "cmd": f"OUTPUT_DIR={dllm2rec_dir} RANKING_PATH={ranking_path} CONFIDENCE_PATH={confidence_path} EMBEDDING_PATH={embedding_path} ./cmd/run_dllm2rec_train.sh {dataset} SASRec {gpu_id} {ed_weight} {lam}",
                    "deps": [ranking_path, confidence_path, "cmd/run_dllm2rec_train.sh"],
                    "outs": [f"{dllm2rec_dir}/metrics.json"]
                }
                
        # Write DVC file
        filename = f"dvc_gpu{gpu_id}.yaml"
        print(f"Writing {filename}...")
        with open(filename, 'w') as f:
            yaml.dump({"stages": dvc_stages}, f, sort_keys=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', nargs='?', default='pipeline_params.csv', help='Path to params CSV')
    args = parser.parse_args()
    generate_pipeline(args.csv_path)
