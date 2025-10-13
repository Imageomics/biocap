#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=[account]
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=biocap-eval
#SBATCH --time=1:00:00
#SBATCH --mem=400GB

export CUDA_VISIBLE_DEVICES=0

cd [path-to-BioCAP]/train_and_eval

MODEL_NAME="hf-hub:imageomics/biocap"
PRETRAINED=""

python -m evaluation.eval_rerank_with_clip \
    --split test \
    --model-name "$MODEL_NAME" \
    --pretrained "$PRETRAINED" \
