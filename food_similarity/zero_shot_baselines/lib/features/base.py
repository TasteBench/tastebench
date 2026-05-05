"""Abstract base class for feature extractors."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class BaseFeature(ABC):
    """Interface that all feature extractors must implement.

    Each feature extractor takes product data and returns a fixed-size
    numpy vector per product, or None if the feature is unavailable.
    """

    def __init__(
        self,
        config: dict,
        products_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> None:
        self.config = config
        self.products_df = products_df
        self.labels_df = labels_df
        self._product_index = {
            int(row["Product code"]): idx
            for idx, row in products_df.iterrows()
        }

    @staticmethod
    def parse_ingredients_with_weights(
        labels_df: pd.DataFrame,
    ) -> Tuple[Dict[int, str], Dict[Tuple[int, str], float]]:
        """Parse ingredient lists and compute inverse-rank weights.

        FDA 21 CFR 101.4 requires ingredients in descending order by weight,
        so rank is a guaranteed proxy for relative proportion. Weights follow
        a Zipfian (1/rank) decay — see features/__init__.py for full rationale.

        Returns:
            code_to_ingredients: {product_code: "ing1 | ing2 | ..."}
            ingredient_weights: {(product_code, ingredient_name): weight}
        """
        code_to_ingredients: Dict[int, str] = {}
        ingredient_weights: Dict[Tuple[int, str], float] = {}
        for _, row in labels_df.iterrows():
            code = int(row["product_code"])
            ing_str = row.get("cleaned_ingredients", "") or ""
            code_to_ingredients[code] = ing_str
            for i, name in enumerate(
                n.strip() for n in ing_str.split(" | ") if n.strip()
            ):
                ingredient_weights[(code, name)] = 1.0 / (i + 1)
        return code_to_ingredients, ingredient_weights

    @abstractmethod
    def extract(self, product_code: int) -> Optional[np.ndarray]:
        """Return a 1-D feature vector for one product, or None if unavailable."""

    def extract_all(self, product_codes: List[int]) -> Dict[int, np.ndarray]:
        """Extract features for multiple products. Skips unavailable ones."""
        result = {}
        for code in product_codes:
            vec = self.extract(code)
            if vec is not None:
                result[code] = vec
        return result
