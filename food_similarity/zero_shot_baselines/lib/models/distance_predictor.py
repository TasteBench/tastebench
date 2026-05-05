"""Distance-to-animal-centroid pairwise ranking model.

For each product category, computes the mean feature vector of the animal
reference product(s). Each plant-based product is scored by its distance
to this centroid — closer products are predicted as more similar to the
animal counterpart.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean

from ..features.base import BaseFeature
from ..preprocessor import FeaturePreprocessor
from .base import BaseModel

logger = logging.getLogger(__name__)


class DistancePredictor(BaseModel):
    """Distance-based pairwise ranking model.

    Algorithm:
    1. Extract feature vectors for all products per feature type
    2. Fit preprocessor (scaler + PCA) on all products
    3. Compute animal centroid per category per feature type
    4. Score each product by weighted distance to its category's animal centroid
    5. For pairwise prediction, pick the product with higher similarity (lower distance)
    """

    DISTANCE_FUNCTIONS = {
        "cosine": cosine,
        "euclidean": euclidean,
        "l2": euclidean,
    }

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.distance_metric: str = config.get("distance_metric", "cosine")
        self.normalization_method: str = config.get("normalization_method", "minmax")
        self.missing_feature_strategy: str = config.get(
            "missing_feature_strategy", "skip"
        )
        self.random_seed: int = config.get("random_seed", 42)

        if self.distance_metric not in self.DISTANCE_FUNCTIONS:
            raise ValueError(
                f"Unknown distance_metric: {self.distance_metric}. "
                f"Use one of: {list(self.DISTANCE_FUNCTIONS.keys())}"
            )

        # Populated during fit
        self.preprocessors: Dict[str, FeaturePreprocessor] = {}
        self.category_centroids: Dict[str, Dict[str, np.ndarray]] = {}
        self.distance_stats: Dict[str, Dict[str, float]] = {}
        self.category_train_distances: Dict[str, Dict[str, np.ndarray]] = {}
        self.product_scores: Dict[int, float] = {}
        self.feature_configs: Dict[str, dict] = {}
        self.enabled_features: List[str] = []

    def fit(
        self,
        products_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        features: Dict[str, BaseFeature],
    ) -> "DistancePredictor":
        """Fit the model: extract features, compute centroids, score all products.

        This is a transductive method: the scaler, PCA, and NaN imputation are
        fit on all products (including those in test pairs). This is valid because
        the model never observes pair labels — it only uses the unlabeled product
        features. Scoring a new product not in products_df requires re-fitting.
        """
        self.feature_configs = {name: features[name].config for name in features}
        self.enabled_features = list(features.keys())

        all_codes = list(products_df["Product code"].astype(int))
        animal_by_cat = self._identify_animal_products(labels_df)
        code_to_cat = dict(
            zip(labels_df["product_code"].astype(int), labels_df["category"])
        )

        # Process each feature type: extract → preprocess → centroid → distances
        raw_distances: Dict[str, Dict[int, float]] = {}
        for feat_name, feature in features.items():
            distances = self._process_feature(
                feat_name, feature, all_codes, animal_by_cat, code_to_cat
            )
            if distances is not None:
                raw_distances[feat_name] = distances

        # Normalize and combine distances into per-product scores
        self._compute_scores(raw_distances, code_to_cat, all_codes)

        logger.info(
            f"Model fitted: {len(self.product_scores)} products scored, "
            f"{len(self.enabled_features)} feature types"
        )
        return self

    def _identify_animal_products(
        self, labels_df: pd.DataFrame
    ) -> Dict[str, List[int]]:
        """Build {category: [animal product codes]} and log centroid stability."""
        animal_by_cat: Dict[str, List[int]] = {}
        for _, row in labels_df.iterrows():
            if str(row["product_type"]).startswith("animal"):
                cat = row["category"]
                code = int(row["product_code"])
                animal_by_cat.setdefault(cat, []).append(code)

        for cat in sorted(animal_by_cat):
            n = len(animal_by_cat[cat])
            level = logging.WARNING if n <= 2 else logging.INFO
            logger.log(
                level,
                f"Category '{cat}': {n} animal reference product(s) for centroid"
            )
        return animal_by_cat

    def _process_feature(
        self,
        feat_name: str,
        feature: BaseFeature,
        all_codes: List[int],
        animal_by_cat: Dict[str, List[int]],
        code_to_cat: Dict[int, str],
    ) -> Optional[Dict[int, float]]:
        """Extract, preprocess, compute centroids and distances for one feature type."""
        feat_config = feature.config
        logger.info(f"Processing feature: {feat_name}")

        # Extract
        feat_vectors = feature.extract_all(all_codes)
        if not feat_vectors:
            logger.warning(f"No vectors extracted for {feat_name}, skipping.")
            return None

        # Stack and impute NaN with column means. This is the single authoritative
        # imputation point — features may return vectors with NaN, and imputation
        # is deferred here so column means are computed across the full dataset.
        codes_with_feat = sorted(feat_vectors.keys())
        matrix = np.array([feat_vectors[c] for c in codes_with_feat])
        col_means = np.nanmean(matrix, axis=0)
        nan_mask = np.isnan(matrix)
        if nan_mask.any():
            for j in range(matrix.shape[1]):
                matrix[nan_mask[:, j], j] = col_means[j]

        # Preprocess (scale + optional PCA).
        # When distance_metric="cosine", StandardScaler produces whitened cosine
        # similarity (Mahalanobis cosine): each dimension contributes equally
        # regardless of variance.
        preprocessor = FeaturePreprocessor(
            pca_dim=feat_config.get("pca_dim"),
            scaler_type=feat_config.get("scaler_type", "standard"),
            random_seed=self.random_seed,
        )
        preprocessor.fit(matrix)
        self.preprocessors[feat_name] = preprocessor
        transformed = preprocessor.transform_batch(matrix)
        transformed_dict = {
            code: transformed[i] for i, code in enumerate(codes_with_feat)
        }

        # Compute animal centroid per category
        centroids: Dict[str, np.ndarray] = {}
        for cat, animal_codes in animal_by_cat.items():
            animal_vecs = [
                transformed_dict[c] for c in animal_codes if c in transformed_dict
            ]
            if animal_vecs:
                centroids[cat] = np.mean(animal_vecs, axis=0)
        self.category_centroids[feat_name] = centroids

        # Compute distances to centroid
        dist_fn = self.DISTANCE_FUNCTIONS[self.distance_metric]
        distances: Dict[int, float] = {}
        for code in codes_with_feat:
            cat = code_to_cat.get(code)
            centroid = centroids.get(cat)
            if centroid is None:
                distances[code] = np.nan
                continue
            try:
                distances[code] = float(dist_fn(transformed_dict[code], centroid))
            except ValueError:
                logger.warning(
                    f"Distance computation failed for product {code} "
                    f"({feat_name}), likely zero-norm vector"
                )
                distances[code] = np.nan

        # Store normalization stats
        valid_dists = [d for d in distances.values() if not np.isnan(d)]
        if valid_dists:
            self.distance_stats[feat_name] = {
                "min": min(valid_dists), "max": max(valid_dists),
            }
            cat_dists: Dict[str, list] = {}
            for code, d in distances.items():
                if not np.isnan(d):
                    cat = code_to_cat.get(code)
                    if cat:
                        cat_dists.setdefault(cat, []).append(d)
            self.category_train_distances[feat_name] = {
                cat: np.sort(ds) for cat, ds in cat_dists.items()
            }

        logger.info(
            f"  {feat_name}: {len(feat_vectors)} products, "
            f"{len(centroids)} category centroids"
        )
        return distances

    def _compute_scores(
        self,
        raw_distances: Dict[str, Dict[int, float]],
        code_to_cat: Dict[int, str],
        all_codes: List[int],
    ) -> None:
        """Normalize distances and combine into per-product similarity scores."""
        # Normalize distances per feature type
        normalized: Dict[str, Dict[int, float]] = {}
        for feat_name in raw_distances:
            if self.normalization_method == "minmax":
                normalized[feat_name] = self._normalize_minmax(
                    raw_distances[feat_name], feat_name
                )
            elif self.normalization_method == "rank":
                normalized[feat_name] = self._normalize_rank(
                    raw_distances[feat_name], feat_name, code_to_cat
                )
            else:
                normalized[feat_name] = raw_distances[feat_name]

        # Combine across feature types (weighted average), negate to get similarity
        for code in all_codes:
            valid_dists = []
            valid_weights = []

            for feat_name in self.enabled_features:
                if feat_name not in normalized:
                    continue
                dist = normalized[feat_name].get(code, np.nan)
                # Relative weight: 2.0 means twice as important as 1.0
                weight = self.feature_configs[feat_name].get("weight", 1.0)

                if np.isnan(dist):
                    if self.missing_feature_strategy == "skip":
                        continue
                    elif self.missing_feature_strategy == "zero":
                        dist = 0.0
                    else:
                        self.product_scores[code] = np.nan
                        break

                valid_dists.append(dist)
                valid_weights.append(weight)

            if code in self.product_scores:
                continue

            if not valid_dists:
                self.product_scores[code] = np.nan
                continue

            dists = np.array(valid_dists)
            weights = np.array(valid_weights)
            combined = np.sum(dists * weights) / np.sum(weights)

            # Negate: lower distance = higher similarity
            self.product_scores[code] = -combined

    def _normalize_minmax(
        self, distances: Dict[int, float], feat_name: str
    ) -> Dict[int, float]:
        """Normalize distances to [0, 1] using global min/max."""
        stats = self.distance_stats.get(feat_name, {"min": 0.0, "max": 1.0})
        min_val, max_val = stats["min"], stats["max"]
        denom = max_val - min_val if max_val > min_val else 1.0

        return {
            code: np.clip((d - min_val) / denom, 0.0, 1.0) if not np.isnan(d) else np.nan
            for code, d in distances.items()
        }

    def _normalize_rank(
        self,
        distances: Dict[int, float],
        feat_name: str,
        code_to_cat: Dict[int, str],
    ) -> Dict[int, float]:
        """Normalize distances using per-category percentile rank."""
        cat_dists = self.category_train_distances.get(feat_name, {})
        result = {}

        for code, dist in distances.items():
            if np.isnan(dist):
                result[code] = np.nan
                continue

            cat = code_to_cat.get(code)
            train_dists = cat_dists.get(cat)

            if train_dists is None or len(train_dists) == 0:
                # Fallback to minmax
                stats = self.distance_stats.get(feat_name, {"min": 0.0, "max": 1.0})
                denom = stats["max"] - stats["min"]
                result[code] = (
                    np.clip((dist - stats["min"]) / denom, 0.0, 1.0)
                    if denom > 0
                    else 0.5
                )
            else:
                rank_pos = np.searchsorted(train_dists, dist, side="right")
                result[code] = rank_pos / (len(train_dists) + 1)

        return result

    def predict_pairs(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """Predict which product in each pair has higher similarity to animal."""
        scores = pd.Series(self.product_scores)
        code1 = pairs_df["product_code_1"].astype(int)
        code2 = pairs_df["product_code_2"].astype(int)
        score1 = code1.map(scores)
        score2 = code2.map(scores)

        # Higher score = more similar to animal.
        # Tiebreak: pick code1 when both NaN or scores equal.
        pick_code2 = (score1.isna() & score2.notna()) | (score2 > score1)
        winner = np.where(pick_code2, code2, code1)

        return pd.DataFrame({
            "test_id": pairs_df["test_id"].astype(int),
            "higher_rated_product": winner,
        })
