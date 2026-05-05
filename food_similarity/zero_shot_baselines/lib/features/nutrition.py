"""Nutrition feature extractor.

Extracts a numeric vector from the nutrition columns in products.csv,
normalized to per-100g serving size.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from .base import BaseFeature

logger = logging.getLogger(__name__)

# All available nutrition columns in products.csv.
# Note: Calories is an energy unit (kcal), not a mass unit, so convert_to_grams
# does not apply to it. StandardScaler normalizes all dimensions regardless.
ALL_NUTRITION_COLUMNS = [
    "Calories",
    "Total Fat (g)",
    "Saturated Fat (g)",
    "Trans Fat (g)",
    "Polyunsaturated Fat (g)",
    "Monounsaturated Fat (g)",
    "Cholesterol (mg)",
    "Sodium (mg)",
    "Total Carbohydrate (g)",
    "Dietary Fiber (g)",
    "Total Sugars (g)",
    "Added Sugars (g)",
    "Protein (g)",
    "Vitamin D (mcg)",
    "Calcium (mg)",
    "Iron (mg)",
    "Potassium (mg)",
]

# Unit conversions to normalize everything to grams per 100g.
# Columns already in grams need no conversion (factor=1.0).
# mg columns need *0.001, mcg columns need *0.000001.
UNIT_CONVERSION = {
    "Cholesterol (mg)": 0.001,
    "Sodium (mg)": 0.001,
    "Calcium (mg)": 0.001,
    "Iron (mg)": 0.001,
    "Potassium (mg)": 0.001,
    "Vitamin D (mcg)": 0.000001,
}


class NutritionFeature(BaseFeature):
    """Extract per-100g normalized nutrition vectors."""

    def __init__(
        self,
        config: dict,
        products_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> None:
        super().__init__(config, products_df, labels_df)

        # Which columns to use (config override or all)
        self.columns: List[str] = config.get("columns", None) or ALL_NUTRITION_COLUMNS
        self.normalize_per_100g: bool = config.get("normalize_per_100g", True)
        self.convert_units: bool = config.get("convert_to_grams", True)

        # Category filter: if specified, only extract for products in these categories
        self.categories: Optional[List[str]] = config.get("categories", None)
        if self.categories is not None:
            cat_col = "category"
            self._allowed_codes = set(
                labels_df.loc[
                    labels_df[cat_col].isin(self.categories), "product_code"
                ].astype(int)
            )
        else:
            self._allowed_codes = None

        # Pre-compute the nutrition matrix for imputation
        self._precompute()

    def _precompute(self) -> None:
        """Pre-compute nutrition vectors.

        Vectors may contain NaN for missing values. NaN imputation is handled
        by DistancePredictor.fit() after all feature vectors are collected,
        ensuring consistent column-mean computation across the full dataset.
        """
        df = self.products_df
        self._vectors = {}

        for _, row in df.iterrows():
            code = int(row["Product code"])
            # Skip products not in allowed categories
            if self._allowed_codes is not None and code not in self._allowed_codes:
                continue
            vec = self._extract_raw(row)
            if vec is not None:
                self._vectors[code] = vec

        logger.info(
            f"NutritionFeature: {len(self._vectors)} products, "
            f"{len(self.columns)} features"
        )

    def _extract_raw(self, row: pd.Series) -> Optional[np.ndarray]:
        """Extract raw nutrition vector from a product row."""
        values = []
        for col in self.columns:
            val = row.get(col, np.nan)
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = np.nan
            values.append(val)

        vec = np.array(values, dtype=np.float64)

        # Skip if all NaN
        if np.isnan(vec).all():
            return None

        # Normalize to per-100g
        if self.normalize_per_100g:
            serving_size = row.get("Serving Size (g)", np.nan)
            try:
                serving_size = float(serving_size)
            except (ValueError, TypeError):
                serving_size = np.nan

            if not np.isnan(serving_size) and serving_size > 0:
                vec = vec * (100.0 / serving_size)
            else:
                logger.debug(
                    f"Missing Serving Size (g) for product — "
                    f"nutrition returned as-is (per-serving)"
                )

        # Unit conversions (mg → g, mcg → g)
        if self.convert_units:
            for i, col in enumerate(self.columns):
                if col in UNIT_CONVERSION:
                    vec[i] *= UNIT_CONVERSION[col]

        return vec

    def extract(self, product_code: int) -> Optional[np.ndarray]:
        """Return per-100g normalized nutrition vector."""
        return self._vectors.get(product_code)
