"""Leave-One-Product-Out Cross-Validation iterator.

Yields (train_keys, held_out_key) for each of 247 NECTAR products.
Product keys are (Category, Product_Code) tuples using original NECTAR codes.

Also provides utilities for assembling feature matrices and pair sets
from the cached product_features.pkl, with optional per-feature-type
PCA (fit on training data, 95% explained variance).
"""

import pickle
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent

# Product key type: (Category, Product_Code)
ProductKey = Tuple[str, int]


def load_product_features(path: Optional[Path] = None) -> dict:
    """Load the cached product features dict.

    Returns:
        dict mapping (Category, Product_Code) -> {category, product_code,
        mean_similarity, nutrition, compound, text, image}
    """
    path = path or DATA_DIR / "product_features.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def get_analog_keys(product_features: dict) -> List[ProductKey]:
    """Get sorted list of analog (plant-based) product keys."""
    return sorted(k for k, pf in product_features.items() if pf.get("is_analog", True))


def get_reference_keys(product_features: dict) -> List[ProductKey]:
    """Get sorted list of animal reference product keys."""
    return sorted(k for k, pf in product_features.items() if pf.get("is_reference", False))


def loocv_iterator(product_features: dict) -> Iterator[Tuple[List[ProductKey], ProductKey]]:
    """Yield (train_keys, held_out_key) for each analog product.

    Only iterates over analog (plant-based) products. Animal reference
    and hybrid products are excluded from the LOOCV loop but remain
    accessible in product_features for models that need them.

    Args:
        product_features: dict from load_product_features()

    Yields:
        (train_keys, held_out_key) where train_keys is a sorted list of
        all analog product keys except held_out_key.
    """
    analog_keys = get_analog_keys(product_features)
    for held_out in analog_keys:
        train_keys = [k for k in analog_keys if k != held_out]
        yield train_keys, held_out


def build_feature_matrix(
    product_features: dict,
    product_keys: List[ProductKey],
    feature_types: List[str],
) -> Tuple[np.ndarray, List[ProductKey]]:
    """Build a feature matrix for the given products and feature types.

    Concatenates the requested feature types into a single matrix.
    Products missing any requested feature are excluded.

    Args:
        product_features: dict from load_product_features()
        product_keys: list of (Category, Product_Code) keys to include
        feature_types: list of feature names, e.g. ["nutrition", "compound", "text", "image"]

    Returns:
        X: (n_products, n_features) feature matrix
        valid_keys: list of product keys that had all requested features
    """
    rows = []
    valid_keys = []
    for key in product_keys:
        pf = product_features[key]
        vecs = []
        skip = False
        for ft in feature_types:
            vec = pf.get(ft)
            if vec is None:
                skip = True
                break
            vecs.append(vec)
        if skip:
            continue
        rows.append(np.concatenate(vecs))
        valid_keys.append(key)

    if not rows:
        return np.empty((0, 0)), []
    return np.array(rows), valid_keys


def build_feature_matrix_imputed(
    product_features: dict,
    product_keys: List[ProductKey],
    feature_types: List[str],
    n_neighbors: int = 5,
) -> Tuple[np.ndarray, List[ProductKey]]:
    """Build a feature matrix with KNN imputation for missing feature types.

    Products missing a feature type (e.g., image) get imputed from the
    K nearest neighbors in the same category, using the features that
    ARE available. This allows all products to be included even when
    some feature types are partially missing.

    Args:
        product_features: dict from load_product_features()
        product_keys: list of (Category, Product_Code) keys to include
        feature_types: list of feature names
        n_neighbors: K for KNN imputation (default 5)

    Returns:
        X: (n_products, n_features) feature matrix (no missing values)
        valid_keys: list of product keys included (all that have at least
            one non-missing feature type)
    """
    from sklearn.metrics.pairwise import cosine_distances

    # Identify which features each product has
    feature_dims = {}
    for pf in product_features.values():
        for ft in feature_types:
            if ft not in feature_dims and pf.get(ft) is not None:
                feature_dims[ft] = pf[ft].shape[0]

    # Separate products into complete and incomplete
    complete_keys = []
    incomplete_keys = []  # (key, set of missing feature types)
    for key in product_keys:
        pf = product_features[key]
        missing = [ft for ft in feature_types if pf.get(ft) is None]
        if not missing:
            complete_keys.append(key)
        elif len(missing) < len(feature_types):
            # Has at least some features — can be imputed
            incomplete_keys.append((key, set(missing)))

    if not complete_keys and not incomplete_keys:
        return np.empty((0, 0)), []

    # Build rows for complete products
    def _make_row(key):
        pf = product_features[key]
        return np.concatenate([pf[ft] for ft in feature_types])

    all_keys = complete_keys[:]
    rows = [_make_row(k) for k in complete_keys]

    # Impute incomplete products via KNN
    if incomplete_keys and complete_keys:
        # Find shared feature types for distance computation
        for key, missing_fts in incomplete_keys:
            pf = product_features[key]
            cat = pf.get("category", key[0])

            # Available features for distance computation
            avail_fts = [ft for ft in feature_types if ft not in missing_fts]
            if not avail_fts:
                continue

            # Build query vector from available features
            query_parts = [pf[ft] for ft in avail_fts]
            query = np.concatenate(query_parts).reshape(1, -1)

            # Find same-category products with complete features
            same_cat_keys = [k for k in complete_keys if product_features[k].get("category", k[0]) == cat]
            if not same_cat_keys:
                same_cat_keys = complete_keys  # fallback to all products

            # Build neighbor matrix from available features only
            neighbor_rows = []
            for nk in same_cat_keys:
                npf = product_features[nk]
                parts = [npf[ft] for ft in avail_fts]
                neighbor_rows.append(np.concatenate(parts))
            neighbor_mat = np.array(neighbor_rows)

            # KNN by cosine distance
            dists = cosine_distances(query, neighbor_mat)[0]
            k = min(n_neighbors, len(same_cat_keys))
            nn_idx = np.argsort(dists)[:k]

            # Impute missing features as mean of K neighbors
            row_parts = []
            for ft in feature_types:
                if ft in missing_fts:
                    # Impute from neighbors
                    nn_vecs = np.array([product_features[same_cat_keys[i]][ft] for i in nn_idx])
                    row_parts.append(nn_vecs.mean(axis=0))
                else:
                    row_parts.append(pf[ft])

            all_keys.append(key)
            rows.append(np.concatenate(row_parts))

    if not rows:
        return np.empty((0, 0)), []
    return np.array(rows), all_keys


def build_score_vector(
    product_features: dict,
    product_keys: List[ProductKey],
) -> np.ndarray:
    """Get mean_similarity scores for the given product keys.

    Args:
        product_features: dict from load_product_features()
        product_keys: list of (Category, Product_Code) keys

    Returns:
        1-D array of mean_similarity scores, aligned with product_keys.
    """
    return np.array([product_features[k]["mean_similarity"] for k in product_keys])


class FeatureProcessor:
    """Per-feature-type StandardScaler + optional PCA (95% variance).

    Fit on training data, apply to train and test. PCA is applied
    per feature type independently (nutrition, compound, text, image),
    then the reduced features are concatenated.

    Usage:
        proc = FeatureProcessor(feature_types, feature_dims, use_pca=True)
        X_train_proc = proc.fit_transform(X_train)
        X_test_proc = proc.transform(X_test)
    """

    def __init__(
        self,
        feature_types: List[str],
        feature_dims: Dict[str, int],
        use_pca: bool = False,
        pca_variance: float = 0.95,
        pca_n_components: Optional[Union[int, Dict[str, int]]] = None,
    ):
        """
        Args:
            feature_types: ordered list of feature type names
            feature_dims: dict mapping feature type -> dimensionality
            use_pca: whether to apply PCA after scaling
            pca_variance: cumulative variance threshold for PCA (default 0.95)
            pca_n_components: fixed PCA components per modality. Accepts:
                - int: same number for all modalities
                - dict: per-modality budgets, e.g. {"text": 30, "compound": 20, "image": 15}
                - None: use variance threshold (default)
        """
        self.feature_types = feature_types
        self.feature_dims = feature_dims
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.pca_n_components = pca_n_components
        self._scalers: Dict[str, StandardScaler] = {}
        self._pcas: Dict[str, PCA] = {}

    def _split(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Split concatenated matrix into per-feature-type matrices."""
        parts = {}
        offset = 0
        for ft in self.feature_types:
            dim = self.feature_dims[ft]
            parts[ft] = X[:, offset:offset + dim]
            offset += dim
        return parts

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit scalers (and optionally PCA) on training data, return transformed."""
        parts = self._split(X)
        transformed = []

        for ft in self.feature_types:
            # StandardScaler
            scaler = StandardScaler()
            scaled = scaler.fit_transform(parts[ft])
            self._scalers[ft] = scaler

            # Only apply PCA to high-dimensional feature types (>10 dims).
            # Small features (category_subset=4, nutrition=6) are kept as-is.
            if self.use_pca and scaled.shape[1] > 10:
                max_comp = min(scaled.shape[0], scaled.shape[1])
                # Resolve per-modality component budget
                ft_n_comp = None
                if isinstance(self.pca_n_components, dict):
                    ft_n_comp = self.pca_n_components.get(ft)
                elif isinstance(self.pca_n_components, int):
                    ft_n_comp = self.pca_n_components

                if ft_n_comp is not None:
                    n_comp = min(ft_n_comp, max_comp)
                    pca = PCA(n_components=n_comp)
                else:
                    # Variance-based threshold (default behavior)
                    pca = PCA(n_components=min(self.pca_variance, max_comp))
                reduced = pca.fit_transform(scaled)
                self._pcas[ft] = pca
                transformed.append(reduced)
            else:
                transformed.append(scaled)

        return np.hstack(transformed)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted scalers and PCA."""
        parts = self._split(X)
        transformed = []

        for ft in self.feature_types:
            scaled = self._scalers[ft].transform(parts[ft])
            if ft in self._pcas:
                scaled = self._pcas[ft].transform(scaled)
            transformed.append(scaled)

        return np.hstack(transformed)

    def get_output_dims(self) -> Dict[str, int]:
        """Return output dimensionality per feature type after processing."""
        dims = {}
        for ft in self.feature_types:
            if ft in self._pcas:
                dims[ft] = self._pcas[ft].n_components_
            else:
                dims[ft] = self.feature_dims[ft]
        return dims


class PLSProcessor:
    """Supervised dimensionality reduction via Partial Least Squares.

    Unlike PCA which finds directions of maximum variance (unsupervised),
    PLS finds directions that maximize covariance with the target variable.
    Must be fit within each LOOCV fold to avoid leakage.

    Usage:
        proc = PLSProcessor(feature_types, feature_dims, n_components=20)
        X_train_proc = proc.fit_transform(X_train, y_train)
        X_test_proc = proc.transform(X_test)
    """

    def __init__(
        self,
        feature_types: List[str],
        feature_dims: Dict[str, int],
        n_components: int = 20,
    ):
        self.feature_types = feature_types
        self.feature_dims = feature_dims
        self.n_components = n_components
        self._scaler: Optional[StandardScaler] = None
        self._pls = None

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit StandardScaler + PLS on training data, return transformed."""
        from sklearn.cross_decomposition import PLSRegression

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Cap components at min(n_samples, n_features)
        max_comp = min(X_scaled.shape[0], X_scaled.shape[1])
        n_comp = min(self.n_components, max_comp)

        self._pls = PLSRegression(n_components=n_comp, scale=False)
        if y is not None:
            X_out = self._pls.fit_transform(X_scaled, y)[0]
        else:
            X_out = self._pls.fit_transform(X_scaled)[0]
        return X_out

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted scaler and PLS."""
        X_scaled = self._scaler.transform(X)
        return self._pls.transform(X_scaled)

    def get_output_dims(self) -> Dict[str, int]:
        """Return output dimensionality (single PLS block)."""
        return {"pls": self._pls.x_loadings_.shape[1] if self._pls else self.n_components}


def get_category(product_features: dict, key: ProductKey) -> str:
    """Get the category of a product."""
    return product_features[key]["category"]


def get_products_by_category(product_features: dict) -> Dict[str, List[ProductKey]]:
    """Group product keys by category.

    Returns:
        dict mapping category_name -> sorted list of product keys
    """
    by_cat: Dict[str, List[ProductKey]] = {}
    for key, pf in product_features.items():
        cat = pf["category"]
        by_cat.setdefault(cat, []).append(key)
    for cat in by_cat:
        by_cat[cat].sort()
    return by_cat
