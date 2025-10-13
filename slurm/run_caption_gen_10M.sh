#!/usr/bin/env bash
set -e

TOTAL_TARS=160
NUM_JOBS=12
BASE_TAR_PATH="path/to/ToL_10M/train"
OUTPUT_DIR="path/to/ToL_10M/train_with_captions"
FORMAT_EXAMPLES="path/to/format_examples.csv"
WIKI_DATA="path/to/wikipedia.parquet"
MODEL_NAME="OpenGVLab/InternVL3-38B-AWQ"

TARS_PER_JOB=$((TOTAL_TARS / NUM_JOBS))
REMAINDER=$((TOTAL_TARS % NUM_JOBS))


generate_tar_list() {
    local start_idx=$1
    local end_idx=$2
    local tar_list=""

    for ((i=start_idx; i<=end_idx; i++)); do
        tar_file=$(printf "%s/shard-%06d.tar" "$BASE_TAR_PATH" "$i")
        tar_list="$tar_list $tar_file"
    done

    echo "$tar_list"
}

# Submit jobs
current_tar=0

for ((job=1; job<=NUM_JOBS; job++)); do
    start_tar=$current_tar

    if [ $job -le $REMAINDER ]; then
        tars_this_job=$((TARS_PER_JOB + 1))
    else
        tars_this_job=$TARS_PER_JOB
    fi

    end_tar=$((start_tar + tars_this_job - 1))

    tar_list=$(generate_tar_list $start_tar $end_tar)

    job_script="job_${job}.sh"

    cat > "$job_script" << EOF
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=[account]
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=biocap-caption-gen-job${job}
#SBATCH --time=40:00:00
#SBATCH --mem=400GB

cd [path-to-BioCAP]

# Configuration - Job $job processing shards $start_tar to $end_tar
INPUT_TARS="$tar_list"
OUTPUT_DIR="$OUTPUT_DIR"
FORMAT_EXAMPLES="$FORMAT_EXAMPLES"
WIKI_DATA="$WIKI_DATA"
MODEL_NAME="$MODEL_NAME"


python -m caption_gen.main \\
    --input_tars \$INPUT_TARS \\
    --output_dir "\$OUTPUT_DIR" \\
    --format_examples "\$FORMAT_EXAMPLES" \\
    --wiki_data "\$WIKI_DATA" \\
    --model_name "\$MODEL_NAME" \\


EOF

    job_id=$(sbatch "$job_script" | awk '{print $4}')
    echo "Submitted Job $job (ID: $job_id) - Processing shards $start_tar to $end_tar ($tars_this_job files)"

    current_tar=$((end_tar + 1))

    rm "$job_script"
done


