"""Feature extraction registry.

Ingredient aggregation weighting
--------------------------------
Features that aggregate ingredient-level vectors to product level (fart,
ingredient_text) use **inverse-rank weights**: ingredient at
position i gets weight 1/(i+1). This is motivated by three observations:

1. FDA 21 CFR 101.4 requires ingredients listed in descending order by
   weight, so rank is a guaranteed proxy for relative proportion.
2. Ingredient proportions in food products empirically follow a Zipfian
   (1/rank) decay — the primary ingredient dominates, and each subsequent
   ingredient contributes progressively less (Zipf, 1949).
3. Unlike inferred proportions (QP solver), rank weights have zero failure
   modes — they require only the ingredient list ordering, not nutrition
   profile resolution.

Weights are normalized to sum to 1 before aggregation.

Adding a new feature type
-------------------------
1. Create a module in this package (e.g., fart.py)
2. Implement a class extending BaseFeature
3. Register it in FEATURE_REGISTRY below
"""

from typing import Dict, Type

import pandas as pd

from .base import BaseFeature
from .nutrition import NutritionFeature


def _get_image_feature():
    from .image import ImageFeature
    return ImageFeature


def _get_compound_feature():
    from .compound import CompoundFeature
    return CompoundFeature


def _get_ingredient_text_feature():
    from .ingredient_text import IngredientTextFeature
    return IngredientTextFeature


FEATURE_REGISTRY: Dict[str, Type[BaseFeature]] = {
    "nutrition": NutritionFeature,
    "image": _get_image_feature,           # lazy to avoid torchvision import
    "compound": _get_compound_feature,      # lazy to avoid heavy imports
    "ingredient_text": _get_ingredient_text_feature,  # lazy to avoid sentence-transformers import
}


def get_feature(
    name: str,
    config: dict,
    products_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> BaseFeature:
    """Instantiate a feature extractor by name.

    Supports prefixed names like 'nutrition_meat' or 'nutrition_dairy'
    which resolve to the base feature type (e.g., 'nutrition').
    """
    # Try exact match first, then prefix match
    registry_name = name
    if name not in FEATURE_REGISTRY:
        # Check for prefix match (e.g., 'nutrition_meat' -> 'nutrition')
        registry_name = None
        for key in FEATURE_REGISTRY:
            if name.startswith(key + "_") or name == key:
                registry_name = key
                break
        if registry_name is None:
            raise ValueError(
                f"Unknown feature '{name}'. Available: {list(FEATURE_REGISTRY.keys())} "
                f"(also supports prefixed names like 'nutrition_meat')"
            )
    cls = FEATURE_REGISTRY[registry_name]
    # Handle lazy loaders (functions that return the actual class)
    if not isinstance(cls, type):
        cls = cls()
    return cls(config, products_df, labels_df)
