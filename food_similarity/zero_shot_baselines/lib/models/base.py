"""Abstract base class for ranking models."""

from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd

from ..features.base import BaseFeature


class BaseModel(ABC):
    """Interface that all ranking models must implement."""

    def __init__(self, config: dict) -> None:
        self.config = config

    @abstractmethod
    def fit(
        self,
        products_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        features: Dict[str, BaseFeature],
    ) -> "BaseModel":
        """Fit the model on product data and features."""

    @abstractmethod
    def predict_pairs(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Predict which product in each pair is more similar to animal reference.

        Returns DataFrame with columns: test_id, higher_rated_product
        """
