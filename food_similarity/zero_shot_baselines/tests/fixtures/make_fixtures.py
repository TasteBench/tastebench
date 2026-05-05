"""Generate small fixtures for compound-embedding feature tests.

Produces a fake cache pickle (3 SMILES → deterministic 4-dim embeddings),
a minimal products_df, and a minimal labels_df. Designed to exercise the
aggregation pipeline without depending on FoodAtlas/SMILES resolver caches.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def fake_cache(path: Path) -> None:
    """Write a 3-SMILES → 4-dim cache pickle."""
    cache = {
        "CCO": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),      # ethanol
        "CC(=O)O": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),  # acetic acid
        "c1ccccc1": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32), # benzene
    }
    with open(path, "wb") as f:
        pickle.dump(cache, f)


def fake_products(path: Path) -> None:
    """Write a minimal products CSV with 2 products."""
    df = pd.DataFrame({
        "product_code": [1, 2],
        "product_name": ["prod_one", "prod_two"],
    })
    df.to_csv(path, index=False)


def fake_labels(path: Path) -> None:
    """Write a minimal labels CSV with 2 products and cleaned_ingredients."""
    df = pd.DataFrame({
        "category": ["TestCat", "TestCat"],
        "product_code": [1, 2],
        "cleaned_ingredients": ["water | salt", "sugar"],
        "is_reference": [False, False],
    })
    df.to_csv(path, index=False)
