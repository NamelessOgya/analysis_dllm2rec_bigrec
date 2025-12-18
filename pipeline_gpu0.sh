#!/bin/bash
# Auto-generated pipeline script. Skips steps if output exists.
set -e

REPO_ROOT=$(pwd)
echo "----------------------------------------------------------------"
echo 'Starting experiment for game_bigrec seed=99 alpha=0.5 strategy=random ...'

# --- Step 1: SASRec ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_99/alpha_0.5/train.pt" ]; then
    echo "Skipping SASRec (Output exists: ${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_99/alpha_0.5/train.pt)"
else
    echo "Running SASRec..."
    OUTPUT_DIR=${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_99/alpha_0.5 ./cmd/run_sasrec_baseline.sh game_bigrec 0 200 99 0.5
fi

# --- Step 2: AL Data ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/active_learning/random_0_0.5_seed_99_alpha_0.5.json" ]; then
    echo "Skipping AL Data Gen (Output exists: ${REPO_ROOT}/experiments/game_bigrec/active_learning/random_0_0.5_seed_99_alpha_0.5.json)"
else
    echo "Running AL Data Gen..."
    OUTPUT_JSON=${REPO_ROOT}/experiments/game_bigrec/active_learning/random_0_0.5_seed_99_alpha_0.5.json ./cmd/create_active_learning_data.sh game_bigrec random 0 0.5 99 1024 
fi

# --- Step 3: BIGRec Train ---
echo "Skipping BIGRec Train (Vanilla Mode: sample_num=0)"

# --- Step 4: BIGRec Inference ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_0_0.5_seed_99_alpha_0.5/train_vanilla.json" ]; then
    echo "Skipping BIGRec Inference (Output exists: ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_0_0.5_seed_99_alpha_0.5/train_vanilla.json)"
else
    echo "Running BIGRec Inference..."
    RESULT_DIR=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_0_0.5_seed_99_alpha_0.5 ./cmd/run_bigrec_inference_vllm.sh --dataset game_bigrec --gpu 0 --model google/gemma-2B-it --seed 99 --sample -1 --checkpoint best --test_data all --correction ci --resource ${REPO_ROOT}/experiments/game_bigrec/sasrec/seed_99/alpha_0.5 --no_adapter
fi

# --- Step 5: DLLM2Rec Train ---
if [ -e "${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_0_0.5_seed_99_alpha_0.5/ed_0.3_lam_0.7/metrics.json" ]; then
    echo "Skipping DLLM2Rec Train (Output exists: ${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_0_0.5_seed_99_alpha_0.5/ed_0.3_lam_0.7/metrics.json)"
else
    echo "Running DLLM2Rec Train..."
    OUTPUT_DIR=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/dllm2rec_final/random_0_0.5_seed_99_alpha_0.5/ed_0.3_lam_0.7 RANKING_PATH=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_0_0.5_seed_99_alpha_0.5/train_vanilla_rank.txt CONFIDENCE_PATH=${REPO_ROOT}/experiments/game_bigrec/google_gemma-2B-it/bigrec_infer_train/random_0_0.5_seed_99_alpha_0.5/train_vanilla_score.txt EMBEDDING_PATH=BIGRec/data/game_bigrec/model_embeddings/google_gemma-2B-it.pt ./cmd/run_dllm2rec_train.sh game_bigrec SASRec 0 0.3 0.7 '' 99 1024 vanilla
fi

