"""LightGBM regression model (pointwise)."""

import numpy as np

from .base import BaseModel


class LightGBMRegressor(BaseModel):
    """LightGBM with regression objective (MSE)."""

    def __init__(self, **params):
        self.params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_child_samples": 5,
            "random_state": 42,
            # Determinism: avoids drift across runs/environments. Must combine
            # all three — random_state alone leaves thread-ordering and data
            # layout nondeterminism on the table.
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
        }
        self.params.update(params)
        self.model = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        import lightgbm as lgb

        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X_train, y_train)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_params(self) -> dict:
        return self.params
