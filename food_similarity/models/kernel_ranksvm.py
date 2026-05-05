"""Kernel RankSVM: nonlinear pairwise ranking via SVM with RBF kernel.

Applies the pairwise transform (features = x_i - x_j, label = preference)
then fits an SVM with RBF kernel. This captures nonlinear feature
interactions without explicit feature engineering.

SVMs excel in small-data regimes (n=215) due to margin maximization
and the kernel trick's implicit infinite-dimensional feature space.
"""

import numpy as np
from sklearn.svm import SVC

from .base import PairwiseModel


class KernelRankSVM(PairwiseModel):
    """RBF kernel SVM on pairwise difference features."""

    def __init__(self, C: float = 1.0, gamma: str = "scale"):
        super().__init__()
        self.C = C
        self.gamma = gamma
        self._svm = None

    def _fit_pairs(self, X_pairs: np.ndarray, y_pairs: np.ndarray,
                   sample_weights=None) -> None:
        self._svm = SVC(
            C=self.C,
            kernel="rbf",
            gamma=self.gamma,
            probability=True,
            random_state=42,
            max_iter=5000,
        )
        self._svm.fit(X_pairs, y_pairs)

    def _predict_pair_proba(self, X_diff: np.ndarray) -> np.ndarray:
        return self._svm.predict_proba(X_diff)[:, 1]

    def get_params(self) -> dict:
        return {"C": self.C, "gamma": self.gamma}
