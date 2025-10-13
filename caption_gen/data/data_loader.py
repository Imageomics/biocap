import csv
import pandas as pd
from typing import Dict
from caption_gen.utils.text_utils import normalize_species_name, reduce_to_binomial


def load_wiki_captions(parquet_path: str) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    df = pd.read_parquet(parquet_path)

    species_col = None
    caption_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'species' in col_lower or 'name' in col_lower:
            species_col = col
        elif 'caption' in col_lower or 'text' in col_lower or 'description' in col_lower:
            caption_col = col

    if species_col is None or caption_col is None:
        if len(df.columns) >= 2:
            species_col = df.columns[0]
            caption_col = df.columns[-1]
        else:
            raise ValueError("Could not identify species and caption columns")

    for _, row in df.iterrows():
        species = str(row[species_col]).strip()
        caption = str(row[caption_col]).strip()

        if not species or not caption or pd.isna(row[species_col]) or pd.isna(row[caption_col]):
            continue

        key_full = normalize_species_name(species)
        key_binom = reduce_to_binomial(species)
        if key_full and key_full not in lookup:
            lookup[key_full] = caption
        if key_binom and key_binom not in lookup:
            lookup[key_binom] = caption

    return lookup


def find_wiki_caption(scientific_name: str, wiki_lookup: Dict[str, str]) -> str:
    if not scientific_name:
        return ""
    key_full = normalize_species_name(scientific_name)
    if key_full in wiki_lookup:
        return wiki_lookup[key_full]
    key_binom = reduce_to_binomial(scientific_name)
    return wiki_lookup.get(key_binom, "")


def load_class_examples(csv_path: str) -> Dict[str, Dict[str, str]]:
    class_mapping = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            key = row["class"].strip().lower()
            class_mapping[key] = {
                'description': row.get("description", "").strip()
            }
    return class_mapping
