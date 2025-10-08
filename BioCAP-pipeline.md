# BioCAP Training Pipeline

This repository contains a complete pipeline for training and evaluating the BioCAP model with enhanced caption generation and visual description filtering. The pipeline consists of four main components (one can skip to [3. Model Training](#3-model-training-train_and_evalslurmtrainsh) if the dataset is downloaded first (see dataset card)):

## 1. Wiki Data Scraping and Filtering (`wiki_scraper_filter/`)

This step relies on first downloading the [TreeOfLife-10M catalog](https://huggingface.co/datasets/imageomics/TreeOfLife-10M/blob/main/metadata/catalog.csv) to have all the required taxa. It is then filtered for unique taxa (preserving full 7-ranks to avoid issues of hemihomonymy), as described in the dataset card.

**Data Scraping:**
- `scraper_wiki.py`: Scrapes Wikipedia pages for biological species data
- Extracts content from sections containing visual descriptions (morphology, appearance, physical characteristics)
- Uses multi-threaded processing with configurable retry mechanisms
- Focuses on keywords like "description", "morphology", "appearance", "identification"

**LLM-based Filtering:**
- `filter.py`: Uses VLLM with large language models to filter and extract visual descriptions
- Multi-GPU processing for scalable text classification
- Filters biological content to extract only visual appearance descriptions
- `extract.py`: Additional utilities for data extraction and processing

## 2. VLM Caption Generation (`caption_gen/submit_all_tars.sh`)

For this step, please download the [TreeOfLife-10M dataset](https://huggingface.co/datasets/imageomics/TreeOfLife-10M); be sure to follow the [reproduction instructions](https://github.com/Imageomics/bioclip/blob/main/docs/imageomics/treeoflife10m.md#reproduce-treeoflife-10m).

**Caption Generation Pipeline:**
- Uses [InternVL3-38B](https://huggingface.co/OpenGVLab/InternVL3-38B) for automatic caption generation
- Processes webdataset tar files in parallel batches
- `main.py`: Core caption generation logic with VLLM integration
- Generates descriptive captions for biological images to enhance training data

**Usage:**
```bash
./submit_all_tars.sh
# Automatically processes 160 tar files across 12 parallel jobs
# Outputs enhanced datasets with generated captions
```

## 3. Model Training (`train_and_eval/slurm/train.sh`)

If you do not wish to reproduce steps 1 & 2, then download [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) and [TreeOfLife-10M Captions](https://huggingface.co/datasets/imageomics/TreeOfLife-10M-Captions) to reproduce the model training.

**Training Setup:**
```bash
# Multi-node distributed training
srun torchrun --nnodes=2 --nproc_per_node=4 \
  -m src.open_clip_train.main \
  --train-data '[training-dir]/shard-{000000..000159}.tar' \
  --val-data '[val-dir]/shard-{000000..000031}.tar' \
  --dataset-type 'webdataset' \
  --pretrained 'openai' \
  --batch-size 4096 \
  --epochs 50
```

## 4. Evaluation Scripts (`train_and_eval/slurm/`)

See the [Evaluation Data section](#evaluation-data) below for details on accessing the evaluation datasets and how to format them locally to run the evaluation scripts.

**INQUIRE Evaluation:**
```bash
./eval_inquire.sh
# Evaluates reranking performance on INQUIRE dataset
```

**Retrieval Evaluation:**
```bash
./eval_retrieval.sh  
# Text-image retrieval on biological datasets (bird, plant)
# Reports R@1, R@5, R@10 metrics
```

**Zero-shot Classification:**
```bash
./eval_zero_shot.sh
# Zero-shot classification on multiple biological datasets
```

## Installation Requirements

**For CLIP Training, Retrieval & Zero-shot Evaluation:**
```bash
pip install open_clip_torch
```

**For INQUIRE Evaluation:**
```bash
pip install -r train_and_eval/src/evaluation_inquire/requirements.txt
```

## Evaluation Data

### Data Directory Structure

To reproduce the reported results, please ensure that all evaluation annotation data is downloaded into your `data/` folder with the following structure. Images can be downloaded into a separate folder, as their location is passed separately to the evaluation script (just be sure to update that part of the code appropriately for your structure!).
```
data/
├── eval/
│   ├── classification_annotation/    # Metadata for zero-shot classification tasks
│   ├── inquire_annotations/          # INQUIRE dataset annotations for reranking evaluation
│   └── retrieval_annotations/        # Text-image retrieval dataset annotations
└── train/                            # See dataset card for training data details
    ├── wiki_and_format_example/
    └──uuid_caption_match/
```

### Classification Benchmarks

Annotations for the classification benchmarks are in the `classification_annotation/` directory, which contains metadata for the following biological classification datasets that were used for zero-shot evaluation.

- **nabirds/**: North American Birds [dataset](https://dl.allaboutbirds.org/nabirds).
- **meta-album/**: Meta-Album biological datasets ([8 mini-datasets](https://paperswithcode.com/dataset/meta-album)).
  - `FNG_Mini/`: Fungi classification
  - `INS_2_Mini/`: Insects classification (version 2)
  - `INS_Mini/`: Insects classification
  - `MED_LF_Mini/`: Mediterranean leaf classification
  - `PLK_Mini/`: Plankton classification
  - `PLT_DOC_Mini/`: Plant documentation classification
  - `PLT_NET_Mini/`: Plant network classification
  - `PLT_VIL_Mini/`: Plant village classification
- **camera_trap/**: Camera trap animal classification, specifically the [IDLE-OO Camera Traps dataset](https://huggingface.co/datasets/imageomics/IDLE-OO-Camera-Traps).
- **rare_species/**: Rare species image classification, specifically the [Rare Species dataset](https://huggingface.co/datasets/imageomics/rare-species).

Each subdirectory contains `metadata.csv` files with classification labels and image information. Note that the metadata CSVs in the `camera_trap/` directory are named based on the LILA BC dataset they are sourced from (e.g., `desert-lion-balanced.csv`). See the classification dataset sources themselves for more information about each benchmark.

### Image Re-ranking (Query)

INQUIRE is an iNaturalist-based [dataset](https://github.com/inquire-benchmark/INQUIRE/tree/main/data) for evaluating biological image reranking capabilities. Download the INQUIRE metadata to the `inquire_annotations/` folder.

**Files:**
- **`inquire_annotations.csv`** (4.3MB): Main annotation file containing image-query relationships
- **`inquire_queries_test.csv`**: Test set queries with categories (used for evaluation)
  - Columns: `query_id`, `query_text`, `supercategory`, `category`, `iconic_group`
  - Categories include: Behavior, Appearance, Context, Species
  - Iconic groups: Mammals, Birds, Fungi, Arachnids, Mollusks, etc.

**Query Examples:**
- "A mongoose standing upright alert" (Behavior → Defensive and Survival Behaviors)
- "A female pheasant" (Appearance → Sex identification)
- "puffins carrying food" (Behavior → Feeding and Hydration)

### Text-to-Image Retrieval Benchmarks

This task is to use the provided descriptive caption to retrieve the associated images from each source. CSVs are provided, in the `retrieval_annotations/` directory, for each with the required information to download the precise images used.
We recommend using the [cautious-robot package](https://github.com/Imageomics/cautious-robot) to automatically validate the download, since these are not packaged benchmarks:

```console
pip install cautious-robot
cautious-robot -i <path/to/benchmark-CSV> -o <path/to/images/folder> -n id -u source_url -v md5

# ex: cautious-robot -i retrieval_annotations/cornell_bird.csv -o images/cornell_bird -n id -u source_url -v md5
```

**Files:**
- **`cornell_bird.csv`** (1.4MB): Cornell bird dataset for identification and description ([data](https://www.macaulaylibrary.org/)).
- **`plantID.csv`** (1.4MB): Plant identification and description [data](https://plantid.net/).

**Format:**
- Columns: `id`, `source_url`, `captions`, `md5`
- Each row contains an image ID, source URL (typically Macaulay Library), and descriptive caption.
- The `md5` column is provided to ensure reproducibility (checks the same images are retrieved).

**Example Entries:**
```csv
id,source_url,captions,md5
22552061,https://macaulaylibrary.org/asset/22552061,"Aberts Towhee, Large, ground-dwelling sparrow with a thick bill and long tail. Brown overall with warm reddish brown undertail.", [ADD MD5]
```
