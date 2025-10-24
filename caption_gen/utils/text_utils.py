from typing import Dict


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