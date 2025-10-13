#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --account=[account]
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=biocap-wiki-scraper
#SBATCH --time=60:00:00
#SBATCH --mem=64GB

cd [path-to-BioCAP]/wiki_scraper

# Scrape unique species names
python scraper_unique_names.py --input path/tospecies_binomial_unique.csv

# Scrape ambiguous species names
python scraper_ambiguous_names.py --input path/to/species_binomial_ambiguous.csv

# Merge results
python merge.py \
    --unique-taxa path/to/species_binomial_unique.csv \
    --unique-descriptions path/to/descriptions_unique.csv \
    --ambiguous-descriptions path/to/descriptions_ambiguous.csv \
    --output path/to/merged_descriptions.csv

