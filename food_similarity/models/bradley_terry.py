"""Feature-based Bradley-Terry model (pairwise).

Classical Bradley-Terry estimates latent strength parameters from
pairwise comparison data: P(i > j) = sigma(s_i - s_j), where s_i
is the strength of item i and sigma is the logistic function.

This feature-based variant parameterizes strength as a linear
function of features: s_i = w^T x_i, so it can generalize to
unseen products. Trained via logistic regression on feature diffs.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .base import PairwiseModel


class FeatureBradleyTerry(PairwiseModel):
    """Bradley-Terry model with feature-based strength function.

    Strength(product) = w^T features. Trained by logistic regression
    on (features_A - features_B, label) pairs.
    """

    def __init__(self, C: float = 1.0):
        """
        Args:
            C: Inverse regularization strength (sklearn convention)
        """
        super().__init__()
        self.C = C
        self._lr = None

    def _fit_pairs(self, X_pairs: np.ndarray, y_pairs: np.ndarray,
                   sample_weights=None) -> None:
        self._lr = LogisticRegression(
            C=self.C,
            fit_intercept=False,  # Bradley-Terry has no intercept
            max_iter=1000,
            random_state=42,
        )
        self._lr.fit(X_pairs, y_pairs, sample_weight=sample_weights)

    def _predict_pair_proba(self, X_diff: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(X_diff)[:, 1]

    def get_params(self) -> dict:
        return {"C": self.C}
