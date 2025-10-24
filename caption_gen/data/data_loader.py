import pandas as pd
from typing import Dict


def load_wiki_captions(parquet_path: str) -> Dict[str, str]:
    """Load UUID to description mapping from parquet file.

    Args:
        parquet_path: Path to parquet file with columns: [uuid, caption, description]

    Returns:
        Dictionary mapping UUID to description text (only non-empty descriptions)
    """
    lookup: Dict[str, str] = {}
    df = pd.read_parquet(parquet_path)

    # Expected columns: [uuid, caption, description]
    # We use columns 0 (uuid) and 2 (description)
    uuid_col = df.columns[0]  # 'uuid'
    desc_col = df.columns[2]  # 'description'

    for _, row in df.iterrows():
        uuid = str(row[uuid_col]).strip()
        description = row[desc_col]

        # Skip if UUID is empty or description is None/NaN/empty
        if not uuid or pd.isna(description) or not str(description).strip():
            continue

        lookup[uuid] = str(description).strip()

    return lookup


def find_wiki_caption_by_uuid(uuid: str, wiki_lookup: Dict[str, str]) -> str:
    """Find description by UUID. Direct O(1) lookup.

    Args:
        uuid: The UUID key from the sample
        wiki_lookup: Dictionary mapping UUID to description

    Returns:
        Description text if found, empty string otherwise
    """
    if not uuid:
        return ""
    return wiki_lookup.get(uuid, "")


def load_class_examples(parquet_path: str) -> Dict[str, Dict[str, str]]:
    """Load class-specific examples from parquet file.

    Args:
        parquet_path: Path to parquet file with columns: [class, description]

    Returns:
        Dictionary mapping class name (lowercase) to description
    """
    class_mapping = {}
    df = pd.read_parquet(parquet_path)

    for _, row in df.iterrows():
        key = str(row["class"]).strip().lower()
        description = str(row.get("description", "")).strip()
        if key and description:
            class_mapping[key] = {
                'description': description
            }

    return class_mapping
