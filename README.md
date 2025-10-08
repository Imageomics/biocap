# BioCAP

This repository contains the code for [BioCAP](https://huggingface.co/imageomics/biocap) training, evaluation, caption generation, and Wikipedia scraper. We developed this repository based on [BioCLIP](https://github.com/imageomics/BioCLIP) and [OpenCLIP](https://github.com/mlfoundations/open_clip).
BioCAP is trained on the [TreeOfLife-10M dataset](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) paired with a new [TreeOfLife-10M Captions dataset](), curated for this model. The BioCAP website is hosted from the `gh-pages` branch of this repository.

[Paper]() | [Model](https://huggingface.co/imageomics/biocap) | [Data](https://huggingface.co/datasets/imageomics/TreeOfLife-10M-Captions) | [Demo]()
---

BioCAP is a CLIP model trained on the 10M-image dataset with both taxonomic labels and fine-grained synthetic captions. BioCAP achieves strong performance on biology-related tasks, including zero-shot classification and text-image retrieval.

## Table of Contents

1. [Model](#model)
2. [Training and Evaluation Commands](#commands)
3. [Paper, website, and data](#paper)
4. [Citation](#citation)

## Model

The main differences in the training implementation between BioCAP and BioCLIP are the adopted model architecture and the introduction of captions. BioCAP uses two separate visual projectors. This part of the code is [transformer.py](train_and_eval/open_clip/transformer.py). In addition, we incorporate synthetic captions as complementary supervision. Synthetic captions help bridge this gap by providing descriptive, trait-focused supervision. This part of the code is [data.py](train_and_eval/open_clip_train/data.py) and [train.py](train_and_eval/open_clip_train/train.py).
We provide the weight of BioCAP in the [BioCAP model repo](https://huggingface.co/imageomics/biocap).

## Commands

For more details on the training and evaluation processes and downloading the requisit data, please see the [BioCAP Pipeline](BioCAP-pipeline.md). A summary for training and evaluating on the different tasks is provided below.

### Training

First download the data from [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) and [TreeOfLife-10M Captions](https://huggingface.co/datasets/imageomics/TreeOfLife-10M-Captions) to reproduce the model training.

Clone this repository, then install the requirements:
```
conda env create -f requirements-training.yml
```

To train the model, run:
```bash
sbatch slurm/train.sh
```

### Evaluation

First install the evaluation environment:
```
conda env create -f environments/eval.yml
```

#### Species classification

We evaluated [BioCAP](https://huggingface.co/imageomics/bioclip-2) on zero-shot classification evaluation using the same test datasets as [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2#evaluation). The metadata used in evaluation zero-shot classification is provided in [`data/classification_annotation`](data/annotation/). All evaluation parameters are described in [src/evaluation/README.md](src/evaluation/README.md).
Please be sure to update the directories accordingly to reflect the locations of these data and metadata in `slurm/eval_zero_shot.sh`, then run:

```bash
sbatch slurm/eval_zero_shot.sh
```

### Image Re-ranking (Query)

For this task, we evaluated on [INQUIRE-Rerank](https://github.com/inquire-benchmark/INQUIRE/), which assesses a model’s ability to reorder 100 initially retrieved images per query so that relevant ones appear higher in the ranking.

This evaluation can be performed by running:

```bash
sbatch slurm/eval_inquire.sh
```

### Text-to-Image Retrieval Benchmarks

We also evaluate our model on the text-image retrieval task using datasets collected from [Cornell Lab of Ornithology, Macaulay Library](https://www.macaulaylibrary.org) and [PlantID.net](https://plantid.net/Home.aspx). The metadata used is provided in [`data/retrieval_annotations`](data/annotation/). Please be sure to update the directories accordingly to reflect the locations of these data and metadata in `slurm/eval_retrieval.sh`, then run:

```bash
sbatch slurm/eval_retrieval.sh
```

### Caption generation

Run the following to create the caption generation environment:

```
conda env create -f environments/caption.yml
```

We use [vLLM](https://github.com/vllm-project/vllm) with [InternVL-3-38B](https://huggingface.co/OpenGVLab/InternVL3-38B-AWQ) to generate fine-grained captions for images. The caption generation process enriches species images with detailed descriptions of visual traits and characteristics. With Wikipedia-derived visual information and taxon-tailored format examples as domain-specific contexts from [`data/wiki_and_format_example/`](data/wiki_and_format_example/), the model generates biologically accurate and descriptive captions.

To generate captions, configure the paths in `slurm/run_caption_gen.sh`, then run:
```bash
sbatch slurm/run_caption_gen.sh
```

### Wiki scraper

We provide scripts to scrape species descriptions from Wikipedia. The scraper extracts visual and morphological information for species based on their binomial names. Species lists are provided in [`data/wiki_species/`](data/wiki_species/), which include both unique and ambiguous species names.

To run the Wikipedia scraper:
```bash
sbatch slurm/scrape_wiki.sh
```

Note that Wikipedia is not versioned, so this process is not perfectly reproducible. This is why we provide the results of this webscraping in the [TreeOfLife-10M-Captions dataset](https://huggingface.co/datasets/imageomics/TreeOfLife-10M-Captions).

<h2 id="paper">Paper, Website, and Data</h2>

We have a preprint on [arXiv]() and a [project website](https://imageomics.github.io/biocap/).

Our data is published on Hugging Face: [TreeOfLife-10M-Captions](https://huggingface.co/datasets/imageomics/TreeOfLife-10M-Captions), as is the existing [TreeOfLife-10M](https://huggingface.co/datasets/imageomics/TreeOfLife-10M) to which the captions are applied (this is the source of the images and their associated taxonomic ranks).

## Citation

Please cite our papers and the associated repositories if you use our code or results.

```
@article{<code>,
  title = {{B}io{CAP}}, 
  author = {},
  year = {2025},
  eprint={},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={}, 
}
 ```

Our code (this repository):
```
@software{biocapcode,
  author = {Ziheng Zhang and Xinyue Ma and Elizabeth G. Campolongo and Matthew J. Thompson and Net Zhang and Jianyang Gu},
  doi = {},
  title = {{B}io{CAP}},
  version = {1.0.0},
  month = {oct},
  year = {2025}
}
```

Also consider citing OpenCLIP and BioCLIP:

```
@software{ilharco_gabriel_2021_5143773,
  author={Ilharco, Gabriel and Wortsman, Mitchell and Wightman, Ross and Gordon, Cade and Carlini, Nicholas and Taori, Rohan and Dave, Achal and Shankar, Vaishaal and Namkoong, Hongseok and Miller, John and Hajishirzi, Hannaneh and Farhadi, Ali and Schmidt, Ludwig},
  title={OpenCLIP},
  year={2021},
  doi={10.5281/zenodo.5143773},
}
```

Original BioCLIP Paper:
 ```
@inproceedings{stevens2024bioclip,
  title = {{B}io{CLIP}: A Vision Foundation Model for the Tree of Life}, 
  author = {Samuel Stevens and Jiaman Wu and Matthew J Thompson and Elizabeth G Campolongo and Chan Hee Song and David Edward Carlyn and Li Dong and Wasila M Dahdul and Charles Stewart and Tanya Berger-Wolf and Wei-Lun Chao and Yu Su},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2024},
  pages = {19412-19424}
}
```
Original Code:
```
@software{bioclip2023code,
  author = {Samuel Stevens and Jiaman Wu and Matthew J. Thompson and Elizabeth G. Campolongo and Chan Hee Song and David Edward Carlyn},
  doi = {10.5281/zenodo.10895871},
  title = {BioCLIP},
  version = {v1.0.0},
  year = {2024}
}
```
BioCLIP 2 Code:
```
@software{bioclip2code,
  author = {Jianyang Gu and Samuel Stevens and Elizabeth G. Campolongo and Matthew J. Thompson and Net Zhang and Jiaman Wu and Zheda Mai},
  doi = {10.5281/zenodo.15644363},
  title = {{B}io{CLIP} 2},
  version = {1.0.1},
  month = {sep},
  year = {2025}
}
```

## License

BioCAP is released under the MIT License. Some elements of the code are copyright by others (see [`LICENSE`](LICENSE)); detailed provenance information is provided in [`HISTORY.md`](HISTORY.md).
