"""Data loading utilities for the pairwise ranking challenge."""

from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_products(data_dir: str | Path) -> pd.DataFrame:
    """Load products.csv with nutrition, ingredients, and image paths."""
    return pd.read_csv(Path(data_dir) / "products.csv")


def load_labels(labels_path: str | Path) -> pd.DataFrame:
    """Load product_labels_manually_cleaned.csv with product_type and cleaned ingredients."""
    return pd.read_csv(labels_path)


def load_pairs(data_dir: str | Path) -> pd.DataFrame:
    """Load ranking_pairs.csv with test pairs to predict."""
    return pd.read_csv(Path(data_dir) / "ranking_pairs.csv")


def load_sample_submission(data_dir: str | Path) -> pd.DataFrame:
    """Load sample_submission.csv for format reference."""
    return pd.read_csv(Path(data_dir) / "sample_submission.csv")


def get_animal_products(labels_df: pd.DataFrame) -> Dict[str, List[int]]:
    """Return {category: [animal product codes]} for animal reference products."""
    animals = labels_df[labels_df["product_type"].str.startswith("animal")]
    result = {}
    for _, row in animals.iterrows():
        cat = row["category"]
        code = int(row["product_code"])
        result.setdefault(cat, []).append(code)
    return result


def get_plant_products(labels_df: pd.DataFrame) -> Dict[str, List[int]]:
    """Return {category: [plant product codes]} for plant-based analogs."""
    plants = labels_df[labels_df["product_type"] == "plant_based"]
    result = {}
    for _, row in plants.iterrows():
        cat = row["category"]
        code = int(row["product_code"])
        result.setdefault(cat, []).append(code)
    return result
