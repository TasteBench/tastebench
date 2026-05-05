"""Distance-to-animal-centroid scoring (faithful port of unsupervised pipeline).

Ports the scoring logic from food_similarity/zero_shot_baselines/lib/models/
distance_predictor.py and food_similarity/zero_shot_baselines/lib/preprocessor.py
into the supervised evaluation framework.

Per-modality pipeline:
  1. Collect feature vectors for ALL products
  2. NaN impute with column means
  3. StandardScaler (transductive — fit on all products)
  4. Compute animal centroid per category (mean of animal product vectors)
  5. Cosine distance from every product to its category centroid
  6. Per-category rank normalization (percentile)
  7. Weighted average across modalities, negated for similarity

The only expected difference vs the unsupervised pipeline is the product
population (247 NECTAR vs 457 NECTAR+taste_like), which affects StandardScaler
parameters and rank normalization denominators. The logic is identical.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .base import BaseModel

logger = logging.getLogger(__name__)

ProductKey = Tuple[str, int]

# Category group definitions for category-specific nutrition.
# Each group becomes a separate sub-modality with its own scaler,
# centroid, and rank normalization — matching the unsupervised NCI config.
CATEGORY_SUBSETS = {
    "meat": [
        "Bacon", "Bratwurst", "Breakfast_Sausages", "Burgers",
        "Chicken_Strips", "Breaded_Chicken_Filet",
        "Unbreaded_Chicken_Breast", "Deli_Ham", "Deli_Turkey",
        "Hot_Dogs", "Meatballs", "Nuggets", "Pulled_Pork", "Steak",
    ],
    "nonsweet_dairy": [
        "Butter", "Cream_Cheese", "Sour_Cream", "Creamer",
        "Milk", "Barista_Milk",
    ],
    "cheese": ["Cheddar_Cheese", "Mozzarella"],
    "sweet_dairy": ["Ice_Cream_Hard_Serve", "Yogurt"],
}

DISTANCE_FUNCTIONS = {
    "cosine": cosine,
    "euclidean": euclidean,
    "l2": euclidean,
}

SCALER_TYPES = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
    "none": None,
}


class DistancePredictor(BaseModel):
    """Cosine distance to animal centroid with rank normalization.

    Faithful port of the unsupervised pipeline's DistancePredictor.
    Operates on pre-extracted features from product_features.pkl.

    Args:
        feature_types: Feature type names to use. The special name
            'category_nutrition' is expanded into 4 sub-modalities
            (meat, nonsweet_dairy, cheese, sweet_dairy), each with
            its own scaler/centroid/rank normalization.
        product_features: Full product_features.pkl dict.
        normalization: 'rank' (per-category percentile), 'minmax', or 'none'.
        distance_metric: 'cosine', 'euclidean', or 'l2'.
        missing_feature_strategy: 'skip', 'zero', or 'nan'.
        feature_weights: Per-feature weight overrides {name: float}.
            Defaults to 1.0 for all features.
        scaler_type: 'standard', 'minmax', 'robust', or 'none'.
    """

    def __init__(
        self,
        feature_types: Optional[List[str]] = None,
        product_features: Optional[dict] = None,
        normalization: str = "rank",
        distance_metric: str = "cosine",
        missing_feature_strategy: str = "skip",
        feature_weights: Optional[Dict[str, float]] = None,
        scaler_type: str = "standard",
    ):
        self.feature_types = feature_types or []
        self._product_features = product_features or {}
        self.normalization = normalization
        self.distance_metric = distance_metric
        self.missing_feature_strategy = missing_feature_strategy
        self.feature_weights = feature_weights or {}
        self.scaler_type = scaler_type

        if distance_metric not in DISTANCE_FUNCTIONS:
            raise ValueError(
                f"Unknown distance_metric: {distance_metric}. "
                f"Use one of: {list(DISTANCE_FUNCTIONS.keys())}"
            )
        if scaler_type not in SCALER_TYPES:
            raise ValueError(
                f"Unknown scaler_type: {scaler_type}. "
                f"Use one of: {list(SCALER_TYPES.keys())}"
            )

        # Populated during fit
        self._product_scores: Dict[ProductKey, float] = {}
        self._distance_stats: Dict[str, Dict[str, float]] = {}
        self._category_train_distances: Dict[str, Dict[str, np.ndarray]] = {}

    def fit(self, X_train: np.ndarray = None, y_train: np.ndarray = None) -> None:
        """Compute scores for all products (transductive, no labels used).

        X_train and y_train are ignored — scores come from distance to
        animal centroid, not from supervised training data.
        """
        pf = self._product_features
        all_keys = sorted(pf.keys())

        # Identify animal products per category
        animal_by_cat = self._identify_animal_products()

        # key -> category mapping
        key_to_cat = {k: pf[k]["category"] for k in all_keys}

        # Expand feature_types: 'category_nutrition' becomes 4 sub-modalities
        expanded_features = self._expand_feature_types()

        # Process each feature: extract → scale → centroid → distance
        raw_distances: Dict[str, Dict[ProductKey, float]] = {}
        for feat_name, feat_config in expanded_features.items():
            distances = self._process_feature(
                feat_name, feat_config, all_keys, animal_by_cat, key_to_cat
            )
            if distances is not None:
                raw_distances[feat_name] = distances

        # Normalize and combine into per-product scores
        self._compute_scores(raw_distances, expanded_features, key_to_cat, all_keys)

    def _identify_animal_products(self) -> Dict[str, List[ProductKey]]:
        """Build {category: [animal product keys]} and log centroid stability."""
        pf = self._product_features
        animal_by_cat: Dict[str, List[ProductKey]] = {}
        for key, p in pf.items():
            if str(p.get("product_type", "")).startswith("animal"):
                cat = p["category"]
                animal_by_cat.setdefault(cat, []).append(key)

        for cat in sorted(animal_by_cat):
            n = len(animal_by_cat[cat])
            level = logging.WARNING if n <= 2 else logging.INFO
            logger.log(
                level,
                f"Category '{cat}': {n} animal reference product(s) for centroid",
            )
        return animal_by_cat

    def _expand_feature_types(self) -> Dict[str, dict]:
        """Expand feature_types into a dict of {name: config}.

        'category_nutrition' is expanded into 4 sub-modalities, each
        filtering to its own category set. Other features pass through
        with no category filter.
        """
        expanded = {}
        for ft in self.feature_types:
            if ft == "category_nutrition":
                # Split into 4 sub-modalities matching unsupervised NCI config
                for group, cats in CATEGORY_SUBSETS.items():
                    sub_name = f"nutrition_{group}"
                    expanded[sub_name] = {
                        "source_key": "category_nutrition",
                        "categories": set(cats),
                        "weight": self.feature_weights.get(sub_name,
                                  self.feature_weights.get(ft, 1.0)),
                    }
            else:
                expanded[ft] = {
                    "source_key": ft,
                    "categories": None,  # no filter
                    "weight": self.feature_weights.get(ft, 1.0),
                }
        return expanded

    def _process_feature(
        self,
        feat_name: str,
        feat_config: dict,
        all_keys: List[ProductKey],
        animal_by_cat: Dict[str, List[ProductKey]],
        key_to_cat: Dict[ProductKey, str],
    ) -> Optional[Dict[ProductKey, float]]:
        """Extract, scale, compute centroid, and distance for one feature."""
        pf = self._product_features
        source_key = feat_config["source_key"]
        cat_filter = feat_config["categories"]
        logger.info(f"Processing feature: {feat_name}")

        # 1. Collect vectors (filter by category if applicable)
        vecs: Dict[ProductKey, np.ndarray] = {}
        for k in all_keys:
            if cat_filter is not None and pf[k]["category"] not in cat_filter:
                continue
            v = pf[k].get(source_key)
            if v is not None:
                vecs[k] = v

        if not vecs:
            logger.warning(f"No vectors extracted for {feat_name}, skipping.")
            return None

        # 2. NaN impute with column means
        keys_ordered = sorted(vecs.keys())
        matrix = np.array([vecs[k] for k in keys_ordered])
        col_means = np.nanmean(matrix, axis=0)
        nan_mask = np.isnan(matrix)
        if nan_mask.any():
            for j in range(matrix.shape[1]):
                matrix[nan_mask[:, j], j] = col_means[j]

        # 3. Scale (StandardScaler by default)
        scaler_cls = SCALER_TYPES[self.scaler_type]
        if scaler_cls is not None:
            scaler = scaler_cls()
            scaler.fit(matrix)
            matrix = scaler.transform(matrix)

        transformed = {k: matrix[i] for i, k in enumerate(keys_ordered)}

        # 4. Compute animal centroid per category
        centroids: Dict[str, np.ndarray] = {}
        for cat, animal_keys in animal_by_cat.items():
            animal_vecs = [
                transformed[k] for k in animal_keys if k in transformed
            ]
            if animal_vecs:
                centroids[cat] = np.mean(animal_vecs, axis=0)

        # 5. Cosine distance from every product to its category centroid
        dist_fn = DISTANCE_FUNCTIONS[self.distance_metric]
        distances: Dict[ProductKey, float] = {}
        for k in keys_ordered:
            cat = key_to_cat.get(k)
            centroid = centroids.get(cat)
            if centroid is None:
                distances[k] = np.nan
                continue
            try:
                distances[k] = float(dist_fn(transformed[k], centroid))
            except ValueError:
                logger.warning(
                    f"Distance computation failed for {k} ({feat_name}), "
                    f"likely zero-norm vector"
                )
                distances[k] = np.nan

        # 6. Store normalization stats (for rank and minmax fallback)
        valid_dists = [d for d in distances.values() if not np.isnan(d)]
        if valid_dists:
            self._distance_stats[feat_name] = {
                "min": min(valid_dists),
                "max": max(valid_dists),
            }
            cat_dists: Dict[str, list] = {}
            for k, d in distances.items():
                if not np.isnan(d):
                    cat = key_to_cat.get(k)
                    if cat:
                        cat_dists.setdefault(cat, []).append(d)
            self._category_train_distances[feat_name] = {
                cat: np.sort(ds) for cat, ds in cat_dists.items()
            }

        logger.info(
            f"  {feat_name}: {len(vecs)} products, "
            f"{len(centroids)} category centroids"
        )
        return distances

    def _compute_scores(
        self,
        raw_distances: Dict[str, Dict[ProductKey, float]],
        expanded_features: Dict[str, dict],
        key_to_cat: Dict[ProductKey, str],
        all_keys: List[ProductKey],
    ) -> None:
        """Normalize distances and combine into per-product similarity scores."""
        # Normalize distances per feature type
        normalized: Dict[str, Dict[ProductKey, float]] = {}
        for feat_name in raw_distances:
            if self.normalization == "minmax":
                normalized[feat_name] = self._normalize_minmax(
                    raw_distances[feat_name], feat_name
                )
            elif self.normalization == "rank":
                normalized[feat_name] = self._normalize_rank(
                    raw_distances[feat_name], feat_name, key_to_cat
                )
            else:
                normalized[feat_name] = raw_distances[feat_name]

        # Combine across feature types (weighted average), negate for similarity
        enabled = list(expanded_features.keys())
        for k in all_keys:
            valid_dists = []
            valid_weights = []

            for feat_name in enabled:
                if feat_name not in normalized:
                    continue
                dist = normalized[feat_name].get(k, np.nan)
                weight = expanded_features[feat_name].get("weight", 1.0)

                if np.isnan(dist):
                    if self.missing_feature_strategy == "skip":
                        continue
                    elif self.missing_feature_strategy == "zero":
                        dist = 0.0
                    else:  # "nan"
                        self._product_scores[k] = np.nan
                        break

                valid_dists.append(dist)
                valid_weights.append(weight)

            if k in self._product_scores:
                continue

            if not valid_dists:
                self._product_scores[k] = np.nan
                continue

            dists_arr = np.array(valid_dists)
            weights_arr = np.array(valid_weights)
            combined = np.sum(dists_arr * weights_arr) / np.sum(weights_arr)

            # Negate: lower distance = higher similarity
            self._product_scores[k] = -combined

    def _normalize_minmax(
        self, distances: Dict[ProductKey, float], feat_name: str
    ) -> Dict[ProductKey, float]:
        """Normalize distances to [0, 1] using global min/max."""
        stats = self._distance_stats.get(feat_name, {"min": 0.0, "max": 1.0})
        min_val, max_val = stats["min"], stats["max"]
        denom = max_val - min_val if max_val > min_val else 1.0

        return {
            k: np.clip((d - min_val) / denom, 0.0, 1.0)
            if not np.isnan(d) else np.nan
            for k, d in distances.items()
        }

    def _normalize_rank(
        self,
        distances: Dict[ProductKey, float],
        feat_name: str,
        key_to_cat: Dict[ProductKey, str],
    ) -> Dict[ProductKey, float]:
        """Normalize distances using per-category percentile rank."""
        cat_dists = self._category_train_distances.get(feat_name, {})
        result: Dict[ProductKey, float] = {}

        for k, dist in distances.items():
            if np.isnan(dist):
                result[k] = np.nan
                continue

            cat = key_to_cat.get(k)
            train_dists = cat_dists.get(cat)

            if train_dists is None or len(train_dists) == 0:
                # Fallback to minmax
                stats = self._distance_stats.get(
                    feat_name, {"min": 0.0, "max": 1.0}
                )
                denom = stats["max"] - stats["min"]
                result[k] = (
                    np.clip((dist - stats["min"]) / denom, 0.0, 1.0)
                    if denom > 0
                    else 0.5
                )
            else:
                rank_pos = np.searchsorted(train_dists, dist, side="right")
                result[k] = rank_pos / (len(train_dists) + 1)

        return result

    # --- Public API (compatible with supervised evaluation framework) ---

    def predict_score(
        self, X: np.ndarray = None, categories: Optional[List[str]] = None
    ) -> np.ndarray:
        """Return pre-computed scores (transductive model).

        X is ignored — scores are computed during fit() for all products.
        Provided for API compatibility. Use get_product_score() instead.
        """
        if X is not None:
            return np.full(X.shape[0], np.nan)
        return np.array([])

    def get_product_score(self, key: ProductKey) -> float:
        """Look up pre-computed score for a product."""
        return self._product_scores.get(key, np.nan)

    def get_all_scores(self) -> Dict[ProductKey, float]:
        """Return all product scores."""
        return dict(self._product_scores)

    def get_params(self) -> dict:
        return {
            "feature_types": self.feature_types,
            "normalization": self.normalization,
            "distance_metric": self.distance_metric,
            "missing_feature_strategy": self.missing_feature_strategy,
            "scaler_type": self.scaler_type,
        }
