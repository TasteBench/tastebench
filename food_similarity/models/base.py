"""Base model interface for supervised ranking models.

All models implement fit() and predict_score(), regardless of whether
they are pointwise (regression) or pairwise (ranking) internally.

Pointwise models:
    - fit(X_train, y_train): train on product features → scores
    - predict_score(X): predict scores for new products

Pairwise models:
    - fit(X_train, y_train): internally generate pairs from training data
    - predict_score(X): predict scores via tournament scoring against training products
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from scipy import stats


class BaseModel(ABC):
    """Common interface for all supervised ranking models."""

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model on training data.

        Args:
            X_train: (n_train, n_features) feature matrix
            y_train: (n_train,) target scores (mean_similarity)
        """

    @abstractmethod
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """Predict scores for one or more products.

        Args:
            X: (n_products, n_features) feature matrix

        Returns:
            (n_products,) predicted scores. Higher = more similar to animal.
        """

    def get_params(self) -> dict:
        """Return model hyperparameters (for logging)."""
        return {}


class PairwiseModel(BaseModel):
    """Base class for pairwise ranking models.

    Subclasses implement _fit_pairs() and _predict_pair_proba().
    The base class handles conversion between pairwise predictions
    and product-level scores via within-category tournament scoring.

    Pairs are only generated between products in the same category,
    matching the ranking task structure.
    """

    def __init__(self):
        self._X_train: Optional[np.ndarray] = None
        self._train_categories: Optional[List[str]] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            categories: Optional[List[str]] = None,
            pair_weighting: Optional[str] = None,
            product_stds: Optional[np.ndarray] = None,
            product_ns: Optional[np.ndarray] = None) -> None:
        """Generate within-category pairs and fit the pairwise model.

        For each pair (i, j) in the same category where y_train[i] > y_train[j],
        creates (X[i] - X[j], 1) and (X[j] - X[i], 0).

        Args:
            X_train: (n_train, n_features) feature matrix
            y_train: (n_train,) target scores
            categories: (n_train,) category labels. If None, all products
                are treated as one category (backward compatibility).
            pair_weighting: weighting strategy for pairs. Options:
                None: uniform weights (default)
                "score_diff": weight = |score_i - score_j|
                "t_stat": weight = |Welch t-statistic| for the pair
                "p_value": weight = 1 - p (two-sided Welch t-test)
            product_stds: (n_train,) per-product similarity std (for t_stat/p_value)
            product_ns: (n_train,) per-product panelist count (for t_stat/p_value)
        """
        self._X_train = X_train.copy()
        self._y_train = y_train.copy()
        self._train_categories = list(categories) if categories is not None else None

        X_pairs, y_pairs, sample_weights = [], [], []
        n = len(X_train)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Only compare within the same category
                if categories is not None and categories[i] != categories[j]:
                    continue
                if y_train[i] > y_train[j]:
                    X_pairs.append(X_train[i] - X_train[j])
                    y_pairs.append(1)
                    X_pairs.append(X_train[j] - X_train[i])
                    y_pairs.append(0)

                    # Compute pair weight (same for both directions)
                    w = 1.0
                    if pair_weighting == "score_diff":
                        w = abs(float(y_train[i] - y_train[j]))
                    elif pair_weighting in ("t_stat", "p_value") and product_stds is not None and product_ns is not None:
                        se_diff = np.sqrt(
                            product_stds[i] ** 2 / max(product_ns[i], 1)
                            + product_stds[j] ** 2 / max(product_ns[j], 1)
                        )
                        if se_diff > 0:
                            t_val = abs(float(y_train[i] - y_train[j])) / se_diff
                            if pair_weighting == "t_stat":
                                w = min(t_val, 5.0)
                            else:  # p_value
                                # Welch-Satterthwaite degrees of freedom
                                s1_sq_n1 = product_stds[i] ** 2 / max(product_ns[i], 1)
                                s2_sq_n2 = product_stds[j] ** 2 / max(product_ns[j], 1)
                                numer = (s1_sq_n1 + s2_sq_n2) ** 2
                                denom = (s1_sq_n1 ** 2 / max(product_ns[i] - 1, 1)
                                         + s2_sq_n2 ** 2 / max(product_ns[j] - 1, 1))
                                df = numer / denom if denom > 0 else 1.0
                                p = 2.0 * stats.t.sf(t_val, df)
                                w = 1.0 - p
                    sample_weights.extend([w, w])

        if not X_pairs:
            return

        X_pairs = np.array(X_pairs)
        y_pairs = np.array(y_pairs)
        weights = np.array(sample_weights) if pair_weighting else None
        self._fit_pairs(X_pairs, y_pairs, sample_weights=weights)

    @abstractmethod
    def _fit_pairs(self, X_pairs: np.ndarray, y_pairs: np.ndarray,
                   sample_weights: Optional[np.ndarray] = None) -> None:
        """Fit on pairwise difference features.

        Args:
            X_pairs: (n_pairs, n_features) feature difference vectors
            y_pairs: (n_pairs,) binary labels (1 = first product wins)
            sample_weights: (n_pairs,) optional per-pair weights
        """

    @abstractmethod
    def _predict_pair_proba(self, X_diff: np.ndarray) -> np.ndarray:
        """Predict P(product A > product B) given feature diffs.

        Args:
            X_diff: (n_pairs, n_features) feature difference vectors (A - B)

        Returns:
            (n_pairs,) probabilities
        """

    def predict_score(self, X: np.ndarray,
                      categories: Optional[List[str]] = None) -> np.ndarray:
        """Predict product scores via within-category tournament scoring.

        For each test product, count wins against training products
        in the same category. Score = expected number of wins.

        Args:
            X: (n_test, n_features) feature matrix
            categories: (n_test,) category labels for test products.
                If None, compares against all training products.
        """
        n_test = X.shape[0]
        scores = np.zeros(n_test)

        for i in range(n_test):
            # Find training products in the same category
            if categories is not None and self._train_categories is not None:
                cat = categories[i]
                mask = np.array([c == cat for c in self._train_categories])
                X_same_cat = self._X_train[mask]
            else:
                X_same_cat = self._X_train

            if len(X_same_cat) == 0:
                scores[i] = 0.0
                continue

            diffs = X[i:i+1] - X_same_cat  # (n_same_cat, n_features)
            probs = self._predict_pair_proba(diffs)
            scores[i] = probs.sum()

        return scores
