#!/bin/bash
#SBATCH --nodes=2
#SBATCH --account=[account]
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --job-name=biocap_train
#SBATCH --time=50:00:00
#SBATCH --mem=800GB

cd [path-to-BioCAP]/train_and_eval

host_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo $host_node

export RDZV_HOST=$host_node
export RDZV_PORT=29322

srun torchrun --nnodes=2 --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=${RDZV_HOST}:${RDZV_PORT} \
  -m open_clip_train.main \
  --train-data '[training-dir]/shard-{000000..000159}.tar' \
  --val-data   '[val-dir]/shard-{000000..000031}.tar' \
  --dataset-type 'webdataset' \
  --pretrained 'openai' \
  --text-type 'random' \
  --warmup 500 \
  --batch-size 4096 \
  --accum-freq 1 \
  --epochs 50 \
  --dual-projector \
  --workers 8 \
  --model ViT-B-16 \
  --lr 1e-4 \
  --log-every-n-steps 20 \
  --dataset-resampled \
  --local-loss \
  --gather-with-grad \
  --grad-checkpointing \
  --save-frequency 1 \

echo "=== Training completed at $(date) ==="