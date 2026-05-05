"""Image embedding feature extractor using DINOv3.

Loads product images and extracts feature vectors using a pre-trained
vision transformer via HuggingFace transformers.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseFeature

logger = logging.getLogger(__name__)

# DINOv3 config name → HuggingFace model ID
DINOV3_HF_MODELS = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

# Pinned revisions for reproducibility
DINOV3_REVISIONS = {
    "dinov3_vits16": "114c1379950215c8b35dfcd4e90a5c251dde0d32",
    "dinov3_vitb16": "5931719e67bbdb9737e363e781fb0c67687896bc",
    "dinov3_vitl16": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
}


class ImageFeature(BaseFeature):
    """Extract image embeddings using DINOv3."""

    def __init__(
        self,
        config: dict,
        products_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> None:
        super().__init__(config, products_df, labels_df)

        try:
            import torch
            from PIL import Image  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Image feature requires torch and Pillow. "
                "Install with: pip install torch Pillow"
            ) from e

        self._torch = torch
        self.model_name: str = config.get("model_name", "dinov3_vitb16")
        self.model_revision: Optional[str] = config.get(
            "model_revision", DINOV3_REVISIONS.get(self.model_name)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_name not in DINOV3_HF_MODELS:
            raise ValueError(
                f"Unknown model '{self.model_name}'. "
                f"Available: {list(DINOV3_HF_MODELS.keys())}"
            )

        # Resolve image base directory
        unsupervised_dir = Path(__file__).parent.parent.parent
        self.image_base_dir = unsupervised_dir / "data" / "competition"

        # Load model
        self._load_model()

        # Pre-compute all embeddings
        self._precompute()

    def _load_model(self) -> None:
        """Load DINOv3 model from HuggingFace transformers."""
        from transformers import AutoImageProcessor, AutoModel

        hf_model_id = DINOV3_HF_MODELS[self.model_name]
        logger.info(f"Loading DINOv3 model: {hf_model_id}...")
        self.processor = AutoImageProcessor.from_pretrained(
            hf_model_id, revision=self.model_revision
        )
        self.model = AutoModel.from_pretrained(
            hf_model_id, revision=self.model_revision
        )
        self.model.eval()
        self.model.to(self.device)
        logger.info(f"DINOv3 model loaded: {self.model_name} on {self.device}")

    def _precompute(self) -> None:
        """Extract embeddings for all products with images."""
        self._vectors = {}
        missing = 0

        for _, row in self.products_df.iterrows():
            code = int(row["Product code"])
            image_path_str = row.get("image_path", "")

            if pd.isna(image_path_str) or not image_path_str:
                missing += 1
                continue

            full_path = self.image_base_dir / image_path_str
            if not full_path.exists():
                missing += 1
                continue

            vec = self._extract_embedding(full_path)
            if vec is not None:
                self._vectors[code] = vec

        logger.info(
            f"ImageFeature: {len(self._vectors)} products with embeddings, "
            f"{missing} missing images"
        )

    def _extract_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """Extract a feature vector from a single image."""
        from PIL import Image

        try:
            img = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            with self._torch.no_grad():
                outputs = self.model(**inputs)
            # CLS token (first token of last hidden state)
            return outputs.last_hidden_state[:, 0].cpu().numpy().flatten()
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(f"Failed to extract embedding from {image_path}: {e}")
            return None

    def extract(self, product_code: int) -> Optional[np.ndarray]:
        """Return image embedding for a product."""
        return self._vectors.get(product_code)
