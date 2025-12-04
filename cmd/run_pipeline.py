import yaml
import subprocess
import os
import sys

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_command(command):
    print(f"Executing: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(e)
        sys.exit(1)

def main():
    config_path = 'pipeline_config.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        sys.exit(1)

    config = load_config(config_path)
    
    # Extract parameters
    common = config.get('common', {})
    dataset = common.get('dataset', 'game')
    gpu_id = common.get('gpu_id', 0)
    seed = common.get('seed', 0)
    
    bigrec = config.get('bigrec', {})
    base_model = bigrec.get('base_model', 'Qwen/Qwen2-0.5B')
    sample = bigrec.get('sample', 1024)
    batch_size = bigrec.get('batch_size', 128)
    micro_batch_size = bigrec.get('micro_batch_size', 4)
    num_epochs = bigrec.get('num_epochs', 50)
    
    inference = config.get('inference', {})
    inf_batch_size = inference.get('batch_size', 32)
    
    dllm2rec = config.get('dllm2rec', {})
    dllm2rec_model = dllm2rec.get('model_name', 'SASRec')

    print("========================================================")
    print("Starting Pipeline from YAML Configuration")
    print(f"Dataset: {dataset}")
    print(f"GPU ID: {gpu_id}")
    print("========================================================")

    # 1. BIGRec Training
    print(">>> Step 1: BIGRec Training")
    # ./cmd/run_bigrec_train.sh <dataset> <gpu_id> <seed> <sample> <batch_size> <micro_batch_size> <base_model> <num_epochs>
    cmd_train = (
        f"./cmd/run_bigrec_train.sh \"{dataset}\" {gpu_id} {seed} {sample} "
        f"{batch_size} {micro_batch_size} \"{base_model}\" {num_epochs}"
    )
    run_command(cmd_train)

    # 2. BIGRec Inference (Generating Distillation Data)
    print(">>> Step 2: BIGRec Inference (on Training Data)")
    # ./cmd/run_bigrec_inference.sh <dataset> <gpu_id> <base_model> <seed> <sample> <skip_inference> <test_data> <batch_size>
    cmd_inf = (
        f"./cmd/run_bigrec_inference.sh \"{dataset}\" {gpu_id} \"{base_model}\" "
        f"{seed} {sample} false train.json {inf_batch_size}"
    )
    run_command(cmd_inf)

    # 3. Data Transfer
    print(">>> Step 3: Data Transfer")
    # ./cmd/transfer_data.sh <dataset> <gpu_id> <base_model> <seed> <sample>
    cmd_transfer = (
        f"./cmd/transfer_data.sh \"{dataset}\" {gpu_id} \"{base_model}\" {seed} {sample}"
    )
    run_command(cmd_transfer)

    # 4. DLLM2Rec Training
    print(">>> Step 4: DLLM2Rec Training")
    # ./cmd/run_dllm2rec_train.sh <dataset> <model_name> <gpu_id>
    cmd_dllm2rec = (
        f"./cmd/run_dllm2rec_train.sh \"{dataset}\" \"{dllm2rec_model}\" {gpu_id}"
    )
    run_command(cmd_dllm2rec)

    print("========================================================")
    print("Pipeline Completed Successfully!")
    print("========================================================")

if __name__ == '__main__':
    main()
