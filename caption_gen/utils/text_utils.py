import re
from typing import Dict


def normalize_species_name(name: str) -> str:
    if not name:
        return ""
    name = name.strip().lower()
    name = re.sub(r"\([^\)]*\)", "", name)
    name = name.replace(",", " ")
    name = name.rstrip(".")
    name = re.sub(r"[^a-z\.\-\s]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def reduce_to_binomial(name: str) -> str:
    normalized = normalize_species_name(name)
    if not normalized:
        return ""
    tokens = normalized.split()
    if len(tokens) >= 2:
        return f"{tokens[0]} {tokens[1]}"
    return normalized


def safe_get_text(sample: Dict, key: str) -> str:
    value = sample.get(key, "")
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    return value.strip()


def extract_class_from_taxon(taxon_tag: str) -> str:
    """Extract class name from taxonomy tag."""
    if not taxon_tag:
        return ""
    parts = taxon_tag.split("class ")
    if len(parts) > 1:
        return parts[1].split(" order")[0].strip()
    return ""