"""Hierarchical Bradley-Terry with empirical Bayes shrinkage.

Trains per-subset BT models (meat, nonsweet_dairy, other_dairy), then
shrinks each subset's coefficients toward the global mean. Small subsets
(few categories, few products) get pulled more toward the global mean,
borrowing strength from larger subsets.

Shrinkage formula:
    w_shrunk_s = (1 - lambda_s) * w_s + lambda_s * w_global
    lambda_s = alpha / (alpha + n_pairs_s)

where alpha controls shrinkage strength (default: median n_pairs across
subsets, so a median-sized subset gets 50% shrinkage).
"""

from typing import Dict, List, Optional

import numpy as np
from scipy import stats
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from .base import PairwiseModel


# Merge cheese + sweet_dairy into other_dairy (2 categories each is too
# few for stable group-level estimation per the Deep Research analysis)
DEFAULT_CATEGORY_SUBSETS = {
    "meat": [
        "Bacon", "Bratwurst", "Breaded_Chicken_Filet", "Breakfast_Sausages",
        "Burgers", "Chicken_Strips", "Deli_Ham", "Deli_Turkey", "Hot_Dogs",
        "Meatballs", "Nuggets", "Pulled_Pork", "Steak", "Unbreaded_Chicken_Breast",
    ],
    "nonsweet_dairy": [
        "Barista_Milk", "Butter", "Cream_Cheese", "Creamer", "Milk", "Sour_Cream",
    ],
    "other_dairy": [
        "Cheddar_Cheese", "Mozzarella", "Ice_Cream_Hard_Serve", "Yogurt",
    ],
}


def _build_cat_to_subset(category_subsets: Dict[str, List[str]]) -> Dict[str, str]:
    """Reverse mapping: category -> subset name."""
    mapping = {}
    for subset, cats in category_subsets.items():
        for cat in cats:
            mapping[cat] = subset
    return mapping


class HierarchicalBT(PairwiseModel):
    """Hierarchical Bradley-Terry with empirical Bayes shrinkage.

    Trains per-subset BT models, then shrinks coefficients toward the
    global mean proportionally to subset size.
    """

    def __init__(
        self,
        C: float = 1.0,
        shrinkage_strength: Optional[float] = None,
        category_subsets: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Args:
            C: Inverse regularization strength for per-subset BT.
            shrinkage_strength: alpha parameter. If None, auto-set to
                median n_pairs across subsets.
            category_subsets: dict mapping subset_name -> list of categories.
                Default: meat, nonsweet_dairy, other_dairy (cheese+sweet merged).
        """
        super().__init__()
        self.C = C
        self.shrinkage_strength = shrinkage_strength
        self.category_subsets = category_subsets or DEFAULT_CATEGORY_SUBSETS
        self._cat_to_subset = _build_cat_to_subset(self.category_subsets)
        self._subset_weights: Dict[str, np.ndarray] = {}
        self._global_weights: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            categories: Optional[List[str]] = None,
            pair_weighting: Optional[str] = None,
            product_stds: Optional[np.ndarray] = None,
            product_ns: Optional[np.ndarray] = None,
            **kwargs) -> None:
        """Fit hierarchical BT via per-subset training + shrinkage.

        Args:
            X_train: (n_train, n_features)
            y_train: (n_train,) mean similarity scores
            categories: (n_train,) category labels (required)
            pair_weighting: "score_diff", "t_stat", "p_value", or None
            product_stds: per-product similarity std (for t_stat/p_value)
            product_ns: per-product panelist count (for t_stat/p_value)
        """
        if categories is None:
            raise ValueError("HierarchicalBT requires categories")

        self._X_train = X_train.copy()
        self._y_train = y_train.copy()
        self._train_categories = list(categories)

        cats = np.array(categories)
        n_features = X_train.shape[1]

        # Phase 1: Fit per-subset BT models
        subset_coefs = {}  # subset -> coefficient vector
        subset_n_pairs = {}  # subset -> number of training pairs

        for subset_name, subset_cats in self.category_subsets.items():
            subset_cat_set = set(subset_cats)
            mask = np.array([c in subset_cat_set for c in cats])

            if mask.sum() < 2:
                continue

            X_sub = X_train[mask]
            y_sub = y_train[mask]
            cats_sub = cats[mask]
            # Subset the per-product stats for pair weighting
            stds_sub = product_stds[mask] if product_stds is not None else None
            ns_sub = product_ns[mask] if product_ns is not None else None

            # Generate within-category pairs for this subset
            X_pairs, y_pairs, sw = [], [], []
            n = len(X_sub)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    if cats_sub[i] != cats_sub[j]:
                        continue
                    if y_sub[i] > y_sub[j]:
                        X_pairs.append(X_sub[i] - X_sub[j])
                        y_pairs.append(1)
                        X_pairs.append(X_sub[j] - X_sub[i])
                        y_pairs.append(0)

                        # Pair weight
                        w = 1.0
                        if pair_weighting == "score_diff":
                            w = abs(float(y_sub[i] - y_sub[j]))
                        elif pair_weighting in ("t_stat", "p_value") and stds_sub is not None and ns_sub is not None:
                            se_diff = np.sqrt(
                                stds_sub[i] ** 2 / max(ns_sub[i], 1)
                                + stds_sub[j] ** 2 / max(ns_sub[j], 1)
                            )
                            if se_diff > 0:
                                t_val = abs(float(y_sub[i] - y_sub[j])) / se_diff
                                if pair_weighting == "t_stat":
                                    w = min(t_val, 5.0)
                                else:  # p_value
                                    s1_sq_n1 = stds_sub[i] ** 2 / max(ns_sub[i], 1)
                                    s2_sq_n2 = stds_sub[j] ** 2 / max(ns_sub[j], 1)
                                    numer = (s1_sq_n1 + s2_sq_n2) ** 2
                                    denom = (s1_sq_n1 ** 2 / max(ns_sub[i] - 1, 1)
                                             + s2_sq_n2 ** 2 / max(ns_sub[j] - 1, 1))
                                    df = numer / denom if denom > 0 else 1.0
                                    p = 2.0 * stats.t.sf(t_val, df)
                                    w = 1.0 - p
                        sw.extend([w, w])

            if len(X_pairs) < 2:
                continue

            X_pairs = np.array(X_pairs)
            y_pairs = np.array(y_pairs)
            sample_weights = np.array(sw) if pair_weighting else None

            lr = LogisticRegression(
                C=self.C, fit_intercept=False, max_iter=1000, random_state=42,
            )
            lr.fit(X_pairs, y_pairs, sample_weight=sample_weights)
            subset_coefs[subset_name] = lr.coef_[0].copy()
            subset_n_pairs[subset_name] = len(X_pairs)

        if not subset_coefs:
            # Fallback: fit a single global model
            self._fit_global_fallback(X_train, y_train, cats,
                                      pair_weighting=pair_weighting,
                                      product_stds=product_stds,
                                      product_ns=product_ns)
            return

        # Phase 2: Compute global mean (weighted by n_pairs)
        total_pairs = sum(subset_n_pairs.values())
        self._global_weights = np.zeros(n_features)
        for s, coef in subset_coefs.items():
            self._global_weights += coef * (subset_n_pairs[s] / total_pairs)

        # Phase 3: Shrink toward global mean
        alpha = self.shrinkage_strength
        if alpha is None:
            # Auto: median n_pairs
            alpha = float(np.median(list(subset_n_pairs.values())))

        self._subset_weights = {}
        for s, coef in subset_coefs.items():
            n_s = subset_n_pairs[s]
            lambda_s = alpha / (alpha + n_s)
            self._subset_weights[s] = (1 - lambda_s) * coef + lambda_s * self._global_weights

        # For subsets with no data, use global weights
        for s in self.category_subsets:
            if s not in self._subset_weights:
                self._subset_weights[s] = self._global_weights.copy()

    def _fit_global_fallback(self, X_train, y_train, cats,
                             pair_weighting=None, product_stds=None, product_ns=None):
        """Fallback to single global BT if per-subset fitting fails."""
        from .base import PairwiseModel
        # Delegate to base class pair generation logic by creating a temp BT
        from .bradley_terry import FeatureBradleyTerry
        bt = FeatureBradleyTerry(C=self.C)
        bt.fit(X_train, y_train, categories=list(cats),
               pair_weighting=pair_weighting, product_stds=product_stds,
               product_ns=product_ns)
        self._global_weights = bt._lr.coef_[0].copy()
        for s in self.category_subsets:
            self._subset_weights[s] = self._global_weights.copy()

    def _fit_pairs(self, X_pairs: np.ndarray, y_pairs: np.ndarray,
                   sample_weights=None) -> None:
        # Not used — fit() is overridden to handle subset-based training
        pass

    def _predict_pair_proba(self, X_diff: np.ndarray) -> np.ndarray:
        # Not used directly — predict_score routes per-subset
        if self._global_weights is not None:
            return expit(X_diff @ self._global_weights)
        return np.full(X_diff.shape[0], 0.5)

    def predict_score(self, X: np.ndarray,
                      categories: Optional[List[str]] = None) -> np.ndarray:
        """Predict via within-category tournament using subset-specific weights."""
        n_test = X.shape[0]
        scores = np.zeros(n_test)

        for i in range(n_test):
            # Determine test product's subset
            test_cat = categories[i] if categories else None
            test_subset = self._cat_to_subset.get(test_cat, "meat")
            w = self._subset_weights.get(test_subset, self._global_weights)

            if w is None:
                scores[i] = 0.0
                continue

            # Tournament scoring against training products in same category
            if self._train_categories is not None:
                mask = np.array([c == test_cat for c in self._train_categories])
                X_same_cat = self._X_train[mask]
            else:
                X_same_cat = self._X_train

            if len(X_same_cat) == 0:
                scores[i] = 0.0
                continue

            diffs = X[i:i+1] - X_same_cat  # (n_same_cat, n_features)
            probs = expit(diffs @ w)
            scores[i] = probs.sum()

        return scores

    def get_params(self) -> dict:
        return {
            "C": self.C,
            "shrinkage_strength": self.shrinkage_strength,
        }
