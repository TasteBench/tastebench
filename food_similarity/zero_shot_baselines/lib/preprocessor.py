"""Feature preprocessing with optional PCA and configurable scaling."""

import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    """Unified preprocessor with optional PCA and configurable scaling.

    Supports fit/transform separation to prevent data leakage.
    """

    SCALER_TYPES = {
        "standard": StandardScaler,
        "minmax": MinMaxScaler,
        "robust": RobustScaler,
        "none": None,
    }

    def __init__(
        self,
        pca_dim: Optional[int] = None,
        scaler_type: str = "standard",
        random_seed: int = 42,
    ) -> None:
        if scaler_type not in self.SCALER_TYPES:
            raise ValueError(
                f"Unknown scaler_type: {scaler_type}. "
                f"Use one of: {list(self.SCALER_TYPES.keys())}"
            )

        self.pca_dim = pca_dim
        self.scaler_type = scaler_type
        self.random_seed = random_seed
        self.pca: Optional[PCA] = None

        scaler_cls = self.SCALER_TYPES[scaler_type]
        self.scaler = scaler_cls() if scaler_cls is not None else None
        self.is_fitted: bool = False

    def fit(self, data: np.ndarray) -> "FeaturePreprocessor":
        """Fit PCA and scaler on data. Call once before transform."""
        if len(data) == 0:
            logger.warning("No data provided to fit FeaturePreprocessor.")
            return self

        if self.pca_dim is not None and self.pca_dim < data.shape[1]:
            n_components = min(self.pca_dim, len(data), data.shape[1])
            self.pca = PCA(n_components=n_components, random_state=self.random_seed)
            data = self.pca.fit_transform(data)
            logger.info(
                f"PCA: {data.shape[1]}D, explained variance: "
                f"{self.pca.explained_variance_ratio_.sum():.4f}"
            )

        if self.scaler is not None:
            self.scaler.fit(data)

        self.is_fitted = True
        return self

    def transform(self, vector: np.ndarray) -> np.ndarray:
        """Apply fitted PCA and scaling to a feature vector."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted before transforming.")

        vector = vector.reshape(1, -1)

        if self.pca is not None:
            vector = self.pca.transform(vector)

        if self.scaler is not None:
            vector = self.scaler.transform(vector)

        return vector.flatten()

    def transform_batch(self, data: np.ndarray) -> np.ndarray:
        """Apply fitted PCA and scaling to a batch of feature vectors."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePreprocessor must be fitted before transforming.")

        if self.pca is not None:
            data = self.pca.transform(data)

        if self.scaler is not None:
            data = self.scaler.transform(data)

        return data
