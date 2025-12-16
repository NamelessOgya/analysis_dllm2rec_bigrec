#!/bin/bash
# Auto-generated pipeline script. Skips steps if output exists.
set -e

REPO_ROOT=$(pwd)
echo "----------------------------------------------------------------"
echo 'Starting experiment for game_bigrec seed=0 alpha=0.5 strategy=random ...'

# --- Step 1: SASRec ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_0/alpha_0.5/train.pt" ]; then
    echo "Skipping SASRec (Output exists: ${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_0/alpha_0.5/train.pt)"
else
    echo "Running SASRec..."
    OUTPUT_DIR=${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_0/alpha_0.5 ./cmd/run_sasrec_baseline.sh game_bigrec 0 200 0 0.5
fi

# --- Step 2: AL Data ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/active_learning/random_1000_0.5_seed_0_alpha_0.5.json" ]; then
    echo "Skipping AL Data Gen (Output exists: ${REPO_ROOT}/experiments/game_bigrec/active_learning/random_1000_0.5_seed_0_alpha_0.5.json)"
else
    echo "Running AL Data Gen..."
    OUTPUT_JSON=${REPO_ROOT}/experiments/game_bigrec/active_learning/random_1000_0.5_seed_0_alpha_0.5.json ./cmd/create_active_learning_data.sh game_bigrec random 1000 0.5 0 1024 
fi

# --- Step 3: BIGRec Train ---
if [ -d "${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_1000_0.5_seed_0_alpha_0.5" ] && [ "$(ls -A ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_1000_0.5_seed_0_alpha_0.5)" ]; then
    echo "Skipping BIGRec Train (Output dir exists and not empty: ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_1000_0.5_seed_0_alpha_0.5)"
else
    echo "Running BIGRec Train..."
    OUTPUT_DIR=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_1000_0.5_seed_0_alpha_0.5 ./cmd/run_bigrec_train.sh game_bigrec 0 0 -1 128 16 google/gemma-2B-it 50 "templates/prompt_template_gemma.txt" "${REPO_ROOT}/experiments/game_bigrec/active_learning/random_1000_0.5_seed_0_alpha_0.5.json"
fi

# --- Step 4: BIGRec Inference ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_1000_0.5_seed_0_alpha_0.5/train_epoch_best.json" ]; then
    echo "Skipping BIGRec Inference (Output exists: ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_1000_0.5_seed_0_alpha_0.5/train_epoch_best.json)"
else
    echo "Running BIGRec Inference..."
    RESULT_DIR=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_1000_0.5_seed_0_alpha_0.5 ./cmd/run_bigrec_inference_vllm.sh --dataset game_bigrec --gpu 0 --model google/gemma-2B-it --seed 0 --sample -1 --checkpoint best --test_data all --correction ci --resource ${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_0/alpha_0.5 --lora_weights ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_1000_0.5_seed_0_alpha_0.5
fi

# --- Step 5: DLLM2Rec Train ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_1000_0.5_seed_0_alpha_0.5/ed_0.3_lam_0.7/metrics.json" ]; then
    echo "Skipping DLLM2Rec Train (Output exists: ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_1000_0.5_seed_0_alpha_0.5/ed_0.3_lam_0.7/metrics.json)"
else
    echo "Running DLLM2Rec Train..."
    OUTPUT_DIR=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_1000_0.5_seed_0_alpha_0.5/ed_0.3_lam_0.7 RANKING_PATH=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_1000_0.5_seed_0_alpha_0.5/train_epoch_best_rank.txt CONFIDENCE_PATH=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_1000_0.5_seed_0_alpha_0.5/train_epoch_best_score.txt EMBEDDING_PATH=BIGRec/data/game_bigrec/model_embeddings/google_gemma-2B-it.pt ./cmd/run_dllm2rec_train.sh game_bigrec SASRec 0 0.3 0.7
fi

echo "----------------------------------------------------------------"
echo 'Starting experiment for game_bigrec seed=2 alpha=0.5 strategy=random ...'

# --- Step 1: SASRec ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_2/alpha_0.5/train.pt" ]; then
    echo "Skipping SASRec (Output exists: ${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_2/alpha_0.5/train.pt)"
else
    echo "Running SASRec..."
    OUTPUT_DIR=${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_2/alpha_0.5 ./cmd/run_sasrec_baseline.sh game_bigrec 0 200 2 0.5
fi

# --- Step 2: AL Data ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/active_learning/random_1000_0.5_seed_2_alpha_0.5.json" ]; then
    echo "Skipping AL Data Gen (Output exists: ${REPO_ROOT}/experiments/game_bigrec/active_learning/random_1000_0.5_seed_2_alpha_0.5.json)"
else
    echo "Running AL Data Gen..."
    OUTPUT_JSON=${REPO_ROOT}/experiments/game_bigrec/active_learning/random_1000_0.5_seed_2_alpha_0.5.json ./cmd/create_active_learning_data.sh game_bigrec random 1000 0.5 2 1024 
fi

# --- Step 3: BIGRec Train ---
if [ -d "${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_1000_0.5_seed_2_alpha_0.5" ] && [ "$(ls -A ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_1000_0.5_seed_2_alpha_0.5)" ]; then
    echo "Skipping BIGRec Train (Output dir exists and not empty: ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_1000_0.5_seed_2_alpha_0.5)"
else
    echo "Running BIGRec Train..."
    OUTPUT_DIR=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_1000_0.5_seed_2_alpha_0.5 ./cmd/run_bigrec_train.sh game_bigrec 0 2 -1 128 16 google/gemma-2B-it 50 "templates/prompt_template_gemma.txt" "${REPO_ROOT}/experiments/game_bigrec/active_learning/random_1000_0.5_seed_2_alpha_0.5.json"
fi

# --- Step 4: BIGRec Inference ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_1000_0.5_seed_2_alpha_0.5/train_epoch_best.json" ]; then
    echo "Skipping BIGRec Inference (Output exists: ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_1000_0.5_seed_2_alpha_0.5/train_epoch_best.json)"
else
    echo "Running BIGRec Inference..."
    RESULT_DIR=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_1000_0.5_seed_2_alpha_0.5 ./cmd/run_bigrec_inference_vllm.sh --dataset game_bigrec --gpu 0 --model google/gemma-2B-it --seed 2 --sample -1 --checkpoint best --test_data all --correction ci --resource ${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_2/alpha_0.5 --lora_weights ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_1000_0.5_seed_2_alpha_0.5
fi

# --- Step 5: DLLM2Rec Train ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_1000_0.5_seed_2_alpha_0.5/ed_0.3_lam_0.7/metrics.json" ]; then
    echo "Skipping DLLM2Rec Train (Output exists: ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_1000_0.5_seed_2_alpha_0.5/ed_0.3_lam_0.7/metrics.json)"
else
    echo "Running DLLM2Rec Train..."
    OUTPUT_DIR=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_1000_0.5_seed_2_alpha_0.5/ed_0.3_lam_0.7 RANKING_PATH=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_1000_0.5_seed_2_alpha_0.5/train_epoch_best_rank.txt CONFIDENCE_PATH=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_1000_0.5_seed_2_alpha_0.5/train_epoch_best_score.txt EMBEDDING_PATH=BIGRec/data/game_bigrec/model_embeddings/google_gemma-2B-it.pt ./cmd/run_dllm2rec_train.sh game_bigrec SASRec 0 0.3 0.7
fi

echo "----------------------------------------------------------------"
echo 'Starting experiment for game_bigrec seed=0 alpha=0.5 strategy=random ...'

# --- Step 1: SASRec ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_0/alpha_0.5/train.pt" ]; then
    echo "Skipping SASRec (Output exists: ${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_0/alpha_0.5/train.pt)"
else
    echo "Running SASRec..."
    OUTPUT_DIR=${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_0/alpha_0.5 ./cmd/run_sasrec_baseline.sh game_bigrec 0 200 0 0.5
fi

# --- Step 2: AL Data ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/active_learning/random_100_0.5_seed_0_alpha_0.5.json" ]; then
    echo "Skipping AL Data Gen (Output exists: ${REPO_ROOT}/experiments/game_bigrec/active_learning/random_100_0.5_seed_0_alpha_0.5.json)"
else
    echo "Running AL Data Gen..."
    OUTPUT_JSON=${REPO_ROOT}/experiments/game_bigrec/active_learning/random_100_0.5_seed_0_alpha_0.5.json ./cmd/create_active_learning_data.sh game_bigrec random 100 0.5 0 1024 
fi

# --- Step 3: BIGRec Train ---
if [ -d "${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_100_0.5_seed_0_alpha_0.5" ] && [ "$(ls -A ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_100_0.5_seed_0_alpha_0.5)" ]; then
    echo "Skipping BIGRec Train (Output dir exists and not empty: ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_100_0.5_seed_0_alpha_0.5)"
else
    echo "Running BIGRec Train..."
    OUTPUT_DIR=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_100_0.5_seed_0_alpha_0.5 ./cmd/run_bigrec_train.sh game_bigrec 0 0 -1 128 16 google/gemma-2B-it 50 "templates/prompt_template_gemma.txt" "${REPO_ROOT}/experiments/game_bigrec/active_learning/random_100_0.5_seed_0_alpha_0.5.json"
fi

# --- Step 4: BIGRec Inference ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_100_0.5_seed_0_alpha_0.5/train_epoch_best.json" ]; then
    echo "Skipping BIGRec Inference (Output exists: ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_100_0.5_seed_0_alpha_0.5/train_epoch_best.json)"
else
    echo "Running BIGRec Inference..."
    RESULT_DIR=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_100_0.5_seed_0_alpha_0.5 ./cmd/run_bigrec_inference_vllm.sh --dataset game_bigrec --gpu 0 --model google/gemma-2B-it --seed 0 --sample -1 --checkpoint best --test_data all --correction ci --resource ${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_0/alpha_0.5 --lora_weights ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_100_0.5_seed_0_alpha_0.5
fi

# --- Step 5: DLLM2Rec Train ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_100_0.5_seed_0_alpha_0.5/ed_0.3_lam_0.7/metrics.json" ]; then
    echo "Skipping DLLM2Rec Train (Output exists: ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_100_0.5_seed_0_alpha_0.5/ed_0.3_lam_0.7/metrics.json)"
else
    echo "Running DLLM2Rec Train..."
    OUTPUT_DIR=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_100_0.5_seed_0_alpha_0.5/ed_0.3_lam_0.7 RANKING_PATH=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_100_0.5_seed_0_alpha_0.5/train_epoch_best_rank.txt CONFIDENCE_PATH=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_100_0.5_seed_0_alpha_0.5/train_epoch_best_score.txt EMBEDDING_PATH=BIGRec/data/game_bigrec/model_embeddings/google_gemma-2B-it.pt ./cmd/run_dllm2rec_train.sh game_bigrec SASRec 0 0.3 0.7
fi

echo "----------------------------------------------------------------"
echo 'Starting experiment for game_bigrec seed=2 alpha=0.5 strategy=random ...'

# --- Step 1: SASRec ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_2/alpha_0.5/train.pt" ]; then
    echo "Skipping SASRec (Output exists: ${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_2/alpha_0.5/train.pt)"
else
    echo "Running SASRec..."
    OUTPUT_DIR=${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_2/alpha_0.5 ./cmd/run_sasrec_baseline.sh game_bigrec 0 200 2 0.5
fi

# --- Step 2: AL Data ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/active_learning/random_100_0.5_seed_2_alpha_0.5.json" ]; then
    echo "Skipping AL Data Gen (Output exists: ${REPO_ROOT}/experiments/game_bigrec/active_learning/random_100_0.5_seed_2_alpha_0.5.json)"
else
    echo "Running AL Data Gen..."
    OUTPUT_JSON=${REPO_ROOT}/experiments/game_bigrec/active_learning/random_100_0.5_seed_2_alpha_0.5.json ./cmd/create_active_learning_data.sh game_bigrec random 100 0.5 2 1024 
fi

# --- Step 3: BIGRec Train ---
if [ -d "${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_100_0.5_seed_2_alpha_0.5" ] && [ "$(ls -A ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_100_0.5_seed_2_alpha_0.5)" ]; then
    echo "Skipping BIGRec Train (Output dir exists and not empty: ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_100_0.5_seed_2_alpha_0.5)"
else
    echo "Running BIGRec Train..."
    OUTPUT_DIR=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_100_0.5_seed_2_alpha_0.5 ./cmd/run_bigrec_train.sh game_bigrec 0 2 -1 128 16 google/gemma-2B-it 50 "templates/prompt_template_gemma.txt" "${REPO_ROOT}/experiments/game_bigrec/active_learning/random_100_0.5_seed_2_alpha_0.5.json"
fi

# --- Step 4: BIGRec Inference ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_100_0.5_seed_2_alpha_0.5/train_epoch_best.json" ]; then
    echo "Skipping BIGRec Inference (Output exists: ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_100_0.5_seed_2_alpha_0.5/train_epoch_best.json)"
else
    echo "Running BIGRec Inference..."
    RESULT_DIR=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_100_0.5_seed_2_alpha_0.5 ./cmd/run_bigrec_inference_vllm.sh --dataset game_bigrec --gpu 0 --model google/gemma-2B-it --seed 2 --sample -1 --checkpoint best --test_data all --correction ci --resource ${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_2/alpha_0.5 --lora_weights ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_train/random_100_0.5_seed_2_alpha_0.5
fi

# --- Step 5: DLLM2Rec Train ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_100_0.5_seed_2_alpha_0.5/ed_0.3_lam_0.7/metrics.json" ]; then
    echo "Skipping DLLM2Rec Train (Output exists: ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_100_0.5_seed_2_alpha_0.5/ed_0.3_lam_0.7/metrics.json)"
else
    echo "Running DLLM2Rec Train..."
    OUTPUT_DIR=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_100_0.5_seed_2_alpha_0.5/ed_0.3_lam_0.7 RANKING_PATH=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_100_0.5_seed_2_alpha_0.5/train_epoch_best_rank.txt CONFIDENCE_PATH=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_100_0.5_seed_2_alpha_0.5/train_epoch_best_score.txt EMBEDDING_PATH=BIGRec/data/game_bigrec/model_embeddings/google_gemma-2B-it.pt ./cmd/run_dllm2rec_train.sh game_bigrec SASRec 0 0.3 0.7
fi

