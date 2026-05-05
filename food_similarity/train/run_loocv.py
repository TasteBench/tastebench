"""LOOCV training driver for supervised ranking models.

Runs Leave-One-Product-Out CV for a given model and feature subset,
saves out-of-fold predictions, and computes metrics with BCa bootstrap CIs.

Usage:
    cd food_similarity
    python -m train.run_loocv --model lightgbm_reg --features NCTI
    python -m train.run_loocv --model bradley_terry --features SN
    python -m train.run_loocv --all  # run all models × all feature subsets
"""

import argparse
import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from data.loocv import (
    FeatureProcessor,
    build_feature_matrix,
    build_feature_matrix_imputed,
    build_score_vector,
    get_products_by_category,
    load_product_features,
    loocv_iterator,
)
from evaluation.metrics import compute_all_metrics, format_metrics
from evaluation.bootstrap import compute_bca_cis

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = SUPERVISED_DIR / "results"

# Feature subset codes
# S = category Subset (4-dim one-hot: meat, nonsweet_dairy, cheese, sweet_dairy)
# N = Nutrition (6-dim)
# C = Compound FART embeddings (768-dim)
# T = Text ingredient embeddings (1024-dim)
# I = Image DINOv3 embeddings (1024-dim)
FEATURE_MAP = {
    "S": ["category_subset"],
    "N": ["nutrition"],
    "C": ["compound"],
    "W": ["compound_weighted"],  # inverse-rank weighted avg over all ingredients (vs top-3 for C)
    "T": ["text"],
    "D": ["sensory"],
    "I": ["image"],
    "R": ["nutrition_ratios"],
    "G": ["ingredient_count"],
    "X": ["ref_cos_sim"],  # 4-dim cosine similarity to animal centroid per modality
}

# All 15 non-empty subsets of {N, C, T, I}, each prefixed with S (category subset
# is always included as context). The ablation tests which content features matter,
# not whether knowing the food category helps.
ALL_FEATURE_SUBSETS = []
for r in range(1, 5):
    for combo in combinations("NCTI", r):
        ALL_FEATURE_SUBSETS.append("S" + "".join(combo))

# Model registry
MODEL_REGISTRY = {
    "ridge": ("models.ridge", "RidgeRegressor", {"alpha": 1.0}),
    "bradley_terry": ("models.bradley_terry", "FeatureBradleyTerry", {}),
    "hierarchical_bt": ("models.hierarchical_bt", "HierarchicalBT", {}),
    "lightgbm_reg": ("models.lightgbm_reg", "LightGBMRegressor", {}),
    "kernel_ranksvm": ("models.kernel_ranksvm", "KernelRankSVM", {"C": 1.0, "gamma": "scale"}),
}


def resolve_feature_types(feature_code: str):
    """Convert feature code like 'NCTI' to list of feature type names."""
    types = []
    for char in feature_code:
        if char in FEATURE_MAP:
            types.extend(FEATURE_MAP[char])
        else:
            raise ValueError(f"Unknown feature code '{char}'. Valid: {list(FEATURE_MAP.keys())}")
    return types


def create_model(model_name: str):
    """Instantiate a model by name from the registry."""
    module_path, class_name, default_params = MODEL_REGISTRY[model_name]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**default_params)


def _make_processor(feature_types, feature_dims, use_pca=True, pca_variance=0.95,
                     pca_n_components=None):
    """Create a FeatureProcessor with scaling + PCA."""
    return FeatureProcessor(
        feature_types=feature_types,
        feature_dims=feature_dims,
        use_pca=use_pca,
        pca_variance=pca_variance,
        pca_n_components=pca_n_components,
    )


# Module-level config set by main() before any run
_PCA_VARIANCE = 0.95
_PCA_N_COMPONENTS = None  # Fixed PCA components per modality (overrides variance)
_SKIP_BOOTSTRAP = False
_DIM_REDUCTION = "pca"  # "pca" or "pls"
_PLS_COMPONENTS = 20
_PAIR_WEIGHTING = None  # "score_diff" or "t_stat"
_KNN_IMPUTE = 0  # 0 = off, >0 = K neighbors for imputation


def _make_dim_reducer(feature_types, feature_dims, y_train=None):
    """Create the appropriate dimensionality reducer based on _DIM_REDUCTION."""
    if _DIM_REDUCTION == "pls":
        from data.loocv import PLSProcessor
        return PLSProcessor(
            feature_types=feature_types,
            feature_dims=feature_dims,
            n_components=_PLS_COMPONENTS,
        )
    return _make_processor(feature_types, feature_dims, pca_variance=_PCA_VARIANCE,
                           pca_n_components=_PCA_N_COMPONENTS)


def run_loocv_pointwise(model_name, feature_types, product_features):
    """Run LOOCV for a pointwise model.

    Applies StandardScaler + PCA (90% variance) per feature type,
    fit on training data each fold.
    """
    oof_predictions = {}
    all_keys = sorted(k for k, pf in product_features.items() if pf.get("is_analog", True))
    feature_dims = _get_feature_dims(product_features, feature_types)

    for i, (train_keys, held_out) in enumerate(loocv_iterator(product_features)):
        X_train_raw, valid_train = _build_features(product_features, train_keys, feature_types)
        y_train = build_score_vector(product_features, valid_train)

        X_test_raw, valid_test = _build_features(product_features, [held_out], feature_types)

        if len(valid_test) == 0:
            oof_predictions[held_out] = np.nan
            continue

        # Scale + dimensionality reduction (fit on training data only)
        proc = _make_dim_reducer(feature_types, feature_dims, y_train)
        X_train = proc.fit_transform(X_train_raw, y_train)
        X_test = proc.transform(X_test_raw)

        model = create_model(model_name)
        model.fit(X_train, y_train)
        score = model.predict_score(X_test)[0]
        oof_predictions[held_out] = float(score)

        if (i + 1) % 50 == 0:
            logger.info(f"  LOOCV progress: {i + 1}/{len(all_keys)}")

    # Log output dims on first fold for visibility
    if feature_dims:
        proc = _make_dim_reducer(feature_types, feature_dims)
        X_all, valid_all = build_feature_matrix(
            product_features,
            sorted(k for k, pf in product_features.items() if pf.get("is_analog", True)),
            feature_types,
        )
        y_all = build_score_vector(product_features, valid_all)
        proc.fit_transform(X_all, y_all)
        logger.info(f"  Output dims: {proc.get_output_dims()}")

    return oof_predictions


def _build_features(product_features, keys, feature_types):
    """Build feature matrix, using KNN imputation if configured."""
    if _KNN_IMPUTE > 0:
        return build_feature_matrix_imputed(product_features, keys, feature_types, n_neighbors=_KNN_IMPUTE)
    return build_feature_matrix(product_features, keys, feature_types)


def _get_feature_dims(product_features, feature_types):
    """Get the dimensionality of each feature type from the data."""
    # Find a product that has all requested features
    for pf in product_features.values():
        if all(pf.get(ft) is not None for ft in feature_types):
            return {ft: pf[ft].shape[0] for ft in feature_types}
    return {}


def run_loocv_pairwise(model_name, feature_types, product_features):
    """Run LOOCV for a pairwise model (Bradley-Terry).

    Applies StandardScaler + PCA before generating within-category pairs.
    """
    oof_predictions = {}
    all_keys = sorted(k for k, pf in product_features.items() if pf.get("is_analog", True))
    feature_dims = _get_feature_dims(product_features, feature_types)

    for i, (train_keys, held_out) in enumerate(loocv_iterator(product_features)):
        X_train_raw, valid_train = _build_features(product_features, train_keys, feature_types)
        y_train = build_score_vector(product_features, valid_train)
        train_cats = [product_features[k]["category"] for k in valid_train]

        X_test_raw, valid_test = _build_features(product_features, [held_out], feature_types)
        if len(valid_test) == 0:
            oof_predictions[held_out] = np.nan
            continue

        test_cats = [product_features[held_out]["category"]]

        # Scale + dimensionality reduction
        proc = _make_dim_reducer(feature_types, feature_dims, y_train)
        X_train = proc.fit_transform(X_train_raw, y_train)
        X_test = proc.transform(X_test_raw)

        # Build pair weighting args
        pw_kwargs = {}
        if _PAIR_WEIGHTING:
            pw_kwargs["pair_weighting"] = _PAIR_WEIGHTING
            if _PAIR_WEIGHTING in ("t_stat", "p_value"):
                pw_kwargs["product_stds"] = np.array([
                    product_features[k].get("similarity_std", 0.0) for k in valid_train
                ])
                pw_kwargs["product_ns"] = np.array([
                    product_features[k].get("n_panelists", 1) for k in valid_train
                ])

        model = create_model(model_name)
        model.fit(X_train, y_train, categories=train_cats, **pw_kwargs)
        score = model.predict_score(X_test, categories=test_cats)[0]
        oof_predictions[held_out] = float(score)

        if (i + 1) % 50 == 0:
            logger.info(f"  LOOCV progress: {i + 1}/{len(all_keys)}")

    return oof_predictions


def run_single(model_name: str, feature_code: str, product_features: dict,
               suffix: str = ""):
    """Run a single model × feature subset combination.

    Args:
        suffix: optional suffix for output filenames (e.g., "_pca90", "_pls20")
    """
    feature_types = resolve_feature_types(feature_code)
    logger.info(f"Running {model_name} with features {feature_code} ({feature_types})"
                f" [dim_reduction={_DIM_REDUCTION}, pca_var={_PCA_VARIANCE}]")

    start = time.time()

    if model_name in ("bradley_terry", "hierarchical_bt", "kernel_ranksvm"):
        oof = run_loocv_pairwise(model_name, feature_types, product_features)
    else:
        oof = run_loocv_pointwise(model_name, feature_types, product_features)

    elapsed = time.time() - start
    logger.info(f"LOOCV completed in {elapsed:.1f}s")

    # Build results DataFrame
    rows = []
    for key, score in sorted(oof.items()):
        pf = product_features[key]
        rows.append({
            "category": pf["category"],
            "product_code": pf["product_code"],
            "true_score": pf["mean_similarity"],
            "predicted_score": score,
        })
    results_df = pd.DataFrame(rows)

    # Compute metrics
    metrics = compute_all_metrics(results_df)
    logger.info(format_metrics(metrics))

    cis = {}
    if not _SKIP_BOOTSTRAP:
        cis = compute_bca_cis(results_df, n_bootstrap=10000)
        logger.info("95% BCa Bootstrap CIs:")
        for metric_name, (lo, hi) in cis.items():
            point = metrics.get(metric_name, np.nan)
            logger.info(f"  {metric_name}: {point:.4f} [{lo:.4f}, {hi:.4f}]")

    # Save OOF predictions
    oof_dir = RESULTS_DIR / "oof_predictions"
    oof_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_name}_{feature_code}{suffix}"
    oof_path = oof_dir / f"{tag}.csv"
    results_df.to_csv(oof_path, index=False)
    logger.info(f"Saved OOF predictions to {oof_path}")

    # Save metrics
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_row = {"model": model_name, "features": feature_code, "n": len(results_df), **metrics}
    for metric_name, (lo, hi) in cis.items():
        metrics_row[f"{metric_name}_ci_lo"] = lo
        metrics_row[f"{metric_name}_ci_hi"] = hi
    metrics_row["elapsed_seconds"] = elapsed

    metrics_path = metrics_dir / f"{tag}.csv"
    pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False)

    return metrics, cis


def main():
    global _PCA_VARIANCE, _PCA_N_COMPONENTS, _SKIP_BOOTSTRAP, _DIM_REDUCTION, _PLS_COMPONENTS, _PAIR_WEIGHTING, _KNN_IMPUTE

    parser = argparse.ArgumentParser(description="Run LOOCV for supervised ranking models")
    parser.add_argument("--model", type=str, help="Model name (e.g., lightgbm_reg)")
    parser.add_argument("--features", type=str, help="Feature code (e.g., NCTI, N, CT)")
    parser.add_argument("--all", action="store_true", help="Run all models × all feature subsets")
    parser.add_argument("--pca-variance", type=float, default=0.95,
                        help="PCA variance threshold (default: 0.95)")
    parser.add_argument("--pca-n-components", type=int, default=None,
                        help="Fixed PCA components per modality (overrides --pca-variance)")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip BCa bootstrap CIs for faster iteration")
    parser.add_argument("--dim-reduction", type=str, default="pca",
                        choices=["pca", "pls"],
                        help="Dimensionality reduction method (default: pca)")
    parser.add_argument("--pls-components", type=int, default=20,
                        help="Number of PLS components (default: 20)")
    parser.add_argument("--pair-weighting", type=str, default=None,
                        choices=["score_diff", "t_stat", "p_value"],
                        help="Pair weighting strategy for pairwise models")
    parser.add_argument("--pca-text-dims", type=int, default=None,
                        help="Fixed PCA dims for text features (overrides --pca-n-components)")
    parser.add_argument("--pca-compound-dims", type=int, default=None,
                        help="Fixed PCA dims for compound features")
    parser.add_argument("--pca-image-dims", type=int, default=None,
                        help="Fixed PCA dims for image features")
    parser.add_argument("--knn-impute", type=int, default=0,
                        help="K for KNN imputation of missing features (0=off, 5=recommended)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix for output filenames (e.g., _pca90)")
    parser.add_argument(
        "--product_features_path",
        type=str,
        default=None,
        help="Override path to product_features.pkl (e.g., product_features_taste_gnn.pkl).",
    )
    args = parser.parse_args()

    _PCA_VARIANCE = args.pca_variance
    # Build per-modality PCA dict if any modality-specific args are set
    modality_dims = {}
    if args.pca_text_dims is not None:
        modality_dims["text"] = args.pca_text_dims
    if args.pca_compound_dims is not None:
        modality_dims["compound"] = args.pca_compound_dims
        modality_dims["compound_weighted"] = args.pca_compound_dims
    if args.pca_image_dims is not None:
        modality_dims["image"] = args.pca_image_dims
    _PCA_N_COMPONENTS = modality_dims if modality_dims else args.pca_n_components
    _SKIP_BOOTSTRAP = args.skip_bootstrap
    _PAIR_WEIGHTING = args.pair_weighting
    _DIM_REDUCTION = args.dim_reduction
    _PLS_COMPONENTS = args.pls_components
    _KNN_IMPUTE = args.knn_impute

    pf_path = Path(args.product_features_path) if args.product_features_path else None
    product_features = load_product_features(pf_path)
    logger.info(f"Loaded {len(product_features)} products")

    suffix = args.suffix

    if args.all:
        for model_name in MODEL_REGISTRY:
            for feature_code in ALL_FEATURE_SUBSETS:
                try:
                    run_single(model_name, feature_code, product_features, suffix=suffix)
                except Exception as e:
                    logger.error(f"FAILED: {model_name}/{feature_code}: {e}")
    elif args.model and args.features:
        run_single(args.model, args.features, product_features, suffix=suffix)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
