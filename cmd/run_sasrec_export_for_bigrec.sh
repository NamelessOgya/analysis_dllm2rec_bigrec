#!/bin/bash

# Consolidated Script to Export ALL SASRec Scores for BIGRec Distillation
# Usage: ./cmd/run_sasrec_export_for_bigrec.sh --dataset [game|movie] --alpha [value] --gpu [id]
# This script runs SASRec training/inference and exports:
# 1. val.pt, val_uids.pt (Best Epoch)
# 2. test.pt, test_uids.pt (Best Epoch)
# 3. train.pt, train_uids.pt (Best Epoch, via --export_train_scores)

DATASET="game"
ALPHA="1.0"
GPU="0"
EPOCH="200"

while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --alpha)
      ALPHA="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --epoch)
      EPOCH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "Running SASRec Training & Full Export for BIGRec on ${DATASET} (Alpha=${ALPHA})..."

cd DLLM2Rec

# Run main.py with export flag
# The script will:
# - Train for EPOCHs (saving val.pt/test.pt/best_model.pth when new best found)
# - At the end, reload best_model.pth and export train.pt/train_uids.pt
python main.py \
    --data ${DATASET} \
    --model_name SASRec \
    --epoch ${EPOCH} \
    --alpha ${ALPHA} \
    --cuda ${GPU} \
    --export_train_scores

echo "Export Complete. All .pt files should be in results/${DATASET}/..."
