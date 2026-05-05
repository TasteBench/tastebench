"""Ingredient text embedding feature extractor.

Embeds the full ingredient list as a single comma-separated string per
product using a sentence-transformers model. This preserves composition
information — which ingredients appear together, their FDA-mandated
descending order by weight, and overall formulation complexity — that is
lost when averaging individual ingredient embeddings.

Supports:
- Instruction-aware models (e.g., Qwen3-Embedding) via the 'instruction'
  config parameter, which steers embeddings toward sensory/functional
  similarity rather than lexical similarity.
- Any sentence-transformers compatible model via 'model_name'.

Pipeline:
  1. Build a comma-separated ingredient string per product from labels_df
  2. Embed each product's full ingredient string in one batch call
  3. Distance to animal centroid determines ranking
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .base import BaseFeature

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_REVISION = "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"


class IngredientTextFeature(BaseFeature):
    """Extract text embeddings from full ingredient lists."""

    def __init__(
        self,
        config: dict,
        products_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> None:
        super().__init__(config, products_df, labels_df)

        from sentence_transformers import SentenceTransformer

        self.model_name = config.get("model_name", DEFAULT_MODEL)
        self.instruction = config.get("instruction", None)
        self.model_revision = config.get("model_revision", DEFAULT_REVISION)

        # Load model
        logger.info(f"Loading text embedding model: {self.model_name}")
        if self.instruction:
            logger.info(f"Using instruction prefix: {self.instruction}")
        self.model = SentenceTransformer(
            self.model_name,
            trust_remote_code=True,
            revision=self.model_revision,
            device="cpu",  # Force CPU for deterministic embeddings (MPS is non-deterministic)
        )

        # Detect embedding dimension from model
        self._embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded: {self._embedding_dim}-dim embeddings")

        # Build code → ingredients mapping from labels.
        self._code_to_ingredients, _ = (
            self.parse_ingredients_with_weights(labels_df)
        )

        # Pre-compute all product embeddings
        self._precompute()

    def _encode_texts(self, texts: list) -> np.ndarray:
        """Encode text strings, applying instruction prefix if configured."""
        kwargs = {"show_progress_bar": False, "normalize_embeddings": True}
        if self.instruction:
            kwargs["prompt"] = self.instruction
        return self.model.encode(texts, **kwargs)

    def _precompute(self) -> None:
        """Embed full ingredient lists per product."""
        # Build per-product ingredient strings
        product_codes = []
        texts = []
        for code, ing_str in self._code_to_ingredients.items():
            if not isinstance(ing_str, str) or not ing_str or ing_str == "nan":
                continue
            # Convert pipe-separated to comma-separated for natural text
            product_codes.append(code)
            texts.append(ing_str.replace(" | ", ", "))

        logger.info(f"Embedding {len(texts)} product ingredient lists...")
        embeddings = self._encode_texts(texts)

        self._vectors: Dict[int, np.ndarray] = {
            code: embeddings[i] for i, code in enumerate(product_codes)
        }

        logger.info(
            f"IngredientTextFeature: {len(self._vectors)} products, "
            f"{self._embedding_dim}-dim embeddings"
        )

    def extract(self, product_code: int) -> Optional[np.ndarray]:
        """Return full ingredient list text embedding for a product."""
        return self._vectors.get(product_code)
