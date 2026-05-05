"""Ridge regression model (pointwise).

With n=215 products and p~2,178 concatenated features, Ridge's L2
regularization is essential to prevent overfitting. Coefficient
magnitudes can be aggregated by feature group (N/C/T/I) to show
which feature types the model relies on most.
"""

import numpy as np
from sklearn.linear_model import Ridge

from .base import BaseModel


class RidgeRegressor(BaseModel):
    """Ridge regression (L2-regularized linear model)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model = Ridge(alpha=self.alpha, fit_intercept=True)
        self.model.fit(X_train, y_train)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> dict:
        return {"alpha": self.alpha}
