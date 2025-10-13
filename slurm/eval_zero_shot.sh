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

LOG_FILEPATH="../storage/logs"
MODEL_NAME="hf-hub:imageomics/biocap"
PRETRAINED=""

TASK_TYPE="all"
TEXT_TYPE="taxon_com"

DATA_ROOTS=(
    "[path-to-BioCAP]/data/imgs/CameraTrap/images/desert-lion/"
    "[path-to-BioCAP]/data/imgs/CameraTrap/images/ENA24/"
    "[path-to-BioCAP]/data/imgs/CameraTrap/images/island/"
    "[path-to-BioCAP]/data/imgs/CameraTrap/images/orinoquia/"
    "[path-to-BioCAP]/data/imgs/CameraTrap/images/ohio-small-animals/"
)
LABEL_FILES=(
    "[path-to-BioCAP]/data/classification_annotation/CameraTrap/desert-lion-balanced.csv"
    "[path-to-BioCAP]/data/classification_annotation/CameraTrap/ENA24-balanced.csv"
    "[path-to-BioCAP]/data/classification_annotation/CameraTrap/island-balanced.csv"
    "[path-to-BioCAP]/data/classification_annotation/CameraTrap/orinoquia-balanced.csv"
    "[path-to-BioCAP]/data/classification_annotation/CameraTrap/ohio-small-animals-balanced.csv"
)

for i in "${!DATA_ROOTS[@]}"; do
    DATA_ROOT=${DATA_ROOTS[$i]}
    LABEL_FILE=${LABEL_FILES[$i]}

    python -m evaluation.zero_shot_iid \
            --model $MODEL_NAME \
            --batch-size 256 \
            --data_root $DATA_ROOT \
            --pretrained $PRETRAINED \
            --label_filename $LABEL_FILE \
            --log $LOG_FILEPATH \
            --text_type $TEXT_TYPE \
            --projector_type tax \

done

TEXT_TYPE="asis"

DATA_ROOTS=(
    "[path-to-BioCAP]/data/imgs/meta-album/set0/PLK_Mini/val"
    "[path-to-BioCAP]/data/imgs/meta-album/set2/INS_Mini/val"
    "[path-to-BioCAP]/data/imgs/meta-album/set1/INS_2_Mini/val"
    "[path-to-BioCAP]/data/imgs/meta-album/set1/PLT_NET_Mini/val"
    "[path-to-BioCAP]/data/imgs/meta-album/set2/FNG_Mini/val"
    "[path-to-BioCAP]/data/imgs/meta-album/set0/PLT_VIL_Mini/val"
    "[path-to-BioCAP]/data/imgs/meta-album/set1/MED_LF_Mini/val"
    "[path-to-BioCAP]/data/imgs/nabird/images/"
)
LABEL_FILES=(
    "[path-to-BioCAP]/data/classification_annotation/meta-album/PLK_Mini/val/metadata.csv"
    "[path-to-BioCAP]/data/classification_annotation/meta-album/INS_Mini/metadata.csv"
    "[path-to-BioCAP]/data/classification_annotation/meta-album/INS_2_Mini/metadata.csv"
    "[path-to-BioCAP]/data/classification_annotation/meta-album/PLT_NET_Mini/metadata.csv"
    "[path-to-BioCAP]/data/classification_annotation/meta-album/FNG_Mini/metadata.csv"
    "[path-to-BioCAP]/data/classification_annotation/meta-album/PLT_VIL_Mini/val/metadata.csv"
    "[path-to-BioCAP]/data/classification_annotation/meta-album/MED_LF_Mini/metadata.csv"
    "[path-to-BioCAP]/data/classification_annotation/nabird/metadata.csv"
)

for i in "${!DATA_ROOTS[@]}"; do
    DATA_ROOT=${DATA_ROOTS[$i]}
    LABEL_FILE=${LABEL_FILES[$i]}

    python -m evaluation.zero_shot_iid \
            --model $MODEL_NAME \
            --batch-size 256 \
            --data_root $DATA_ROOT \
            --pretrained $PRETRAINED \
            --label_filename $LABEL_FILE \
            --log $LOG_FILEPATH \
            --text_type $TEXT_TYPE \
            --projector_type tax \

done


TEXT_TYPE="taxon_com"
DATA_ROOT="[path-to-BioCAP]/data/imgs/rare-species/"
LABEL_FILE="[path-to-BioCAP]/data/classification_annotation/rare-species/metadata.csv"

python -m evaluation.zero_shot_iid \
        --model $MODEL_NAME \
        --batch-size 256 \
        --data_root $DATA_ROOT \
        --pretrained $PRETRAINED \
        --label_filename $LABEL_FILE \
        --log $LOG_FILEPATH \
        --text_type $TEXT_TYPE \
        --projector_type tax \

