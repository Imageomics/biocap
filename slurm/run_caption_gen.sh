#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=[account]
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=biocap-eval
#SBATCH --time=20:00:00
#SBATCH --mem=400GB

cd [path-to-BioCAP]

# Configuration - Modify these paths before running
INPUT_TAR="path/to/input/shard-000000.tar"
OUTPUT_TAR="path/to/output/shard-000000_with_captions.tar"
FORMAT_EXAMPLES="path/to/format_examples.csv"
WIKI_DATA="path/to/wikipedia.parquet"
MODEL_NAME="OpenGVLab/InternVL3-38B-AWQ"


python -m caption_gen.main \
    --input_tars "$INPUT_TAR" \
    --output_tar "$OUTPUT_TAR" \
    --format_examples "$FORMAT_EXAMPLES" \
    --wiki_data "$WIKI_DATA" \
    --model_name "$MODEL_NAME" \


