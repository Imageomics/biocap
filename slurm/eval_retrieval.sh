#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=[account]
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=biocap-eval
#SBATCH --time=4:00:00
#SBATCH --mem=400GB

export CUDA_VISIBLE_DEVICES=0

cd [path-to-BioCAP]/train_and_eval

CSV_FILES=(
    "path/to/cornell_bird.csv"
    "path/to/plantID.csv"
)

IMAGE_FOLDERS=(
    "path/to/bird_photos"
    "path/to/plant_photos"
)

# Model configuration
MODEL_NAME="hf-hub:imageomics/biocap"
PRETRAINED=""

# Run evaluations
for i in "${!CSV_FILES[@]}"; do
    CSV_FILE=${CSV_FILES[$i]}
    IMAGE_FOLDER=${IMAGE_FOLDERS[$i]}

    python -m evaluation.retrieval_openclip \
        --model-name "$MODEL_NAME" \
        --pretrained "$PRETRAINED" \
        --csv-file "$CSV_FILE" \
        --image-folder "$IMAGE_FOLDER"

done

