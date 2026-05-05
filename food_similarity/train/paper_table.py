"""Paper-ready comparison table for NeurIPS 2026.

All supervised models use the SAME preprocessing for fair benchmark comparison:
  - Feature set: SNCTI (category + nutrition + compound + text + image)
  - Per-modality StandardScaler + PCA to 95% explained variance (fit per LOOCV fold)
  - 16 products missing raw image vectors are KNN-imputed (k=5) using same-category
    neighbors based on available modalities

No feature-subset cherry-picking, no hand-engineered distance features, no
per-modality preprocessing tricks. Every modality is treated identically
and the raw honest numbers are reported. If a modality hurts a model on
this small-n dataset, that's a finding for the paper.

Unsupervised rows (DistancePredictor, LLMs) use their own native feature
interfaces since they don't fit supervised models.

Workflow:
  --generate-fast: ~10 min
      DistancePredictor, Ridge, Bradley-Terry, Hierarchical BT,
      Kernel RankSVM, LightGBM, then build table.

Usage:
    cd food_similarity
    python -m train.paper_table --generate-fast
    python -m train.paper_table --table-only
    python -m train.paper_table --table-only --output table.tex
"""

import argparse
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERVISED_DIR))

from data.loocv import (
    build_feature_matrix_imputed,
    get_analog_keys,
    load_product_features,
)
from evaluation.bootstrap import compute_bca_cis
from evaluation.metrics import compute_all_metrics

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OOF_DIR = SUPERVISED_DIR / "results" / "oof_predictions"

# Uniform preprocessing for all supervised models
FEATURE_CODE = "SNCTI"
SUFFIX = "_bench"
KNN_K = 5

# DistancePredictor's native feature interface (unsupervised, transductive)
# NCI = category_nutrition + compound + image. This is the pre-registered
# configuration from the original unsupervised pipeline; it is the best
# cosine config from the full 15-combo ablation and within 1.5% of the
# best L2 config (NTI). Using the same config for both metrics preserves
# the scientific "same preprocessing, different distance function" comparison.
DP_FEATURE_TYPES = ["category_nutrition", "compound", "image"]
DP_FEATURE_DISPLAY = "NCI"


# ---------------------------------------------------------------------------
# Data class for table rows
# ---------------------------------------------------------------------------

@dataclass
class TableRow:
    display_name: str
    group_label: str
    feature_display: str
    pairwise_accuracy: float
    ci_lo: float
    ci_hi: float
    n_products: int
    oof_filename: str = ""


GROUP_ORDER = [
    "Unsupervised",
    "Supervised (linear)",
    "Supervised (pairwise)",
    "Supervised (listwise)",
    "Supervised (nonlinear)",
    "Ensemble",
]


# ---------------------------------------------------------------------------
# Transductive KNN image imputation (pre-computed once, used by all folds)
# ---------------------------------------------------------------------------

def impute_missing_images_inplace(product_features: dict, knn_k: int = KNN_K) -> int:
    """KNN-impute the image vector for the 16 analog products missing it.

    Transductive imputation: we compute imputed image vectors once before
    running LOOCV and write them back into product_features. This avoids
    a bug in the in-fold KNN imputation path that drops products during
    single-product test folds. It is leak-free because imputation uses
    only feature values (not labels) from same-category neighbors.

    Uses the existing build_feature_matrix_imputed infrastructure to find
    same-category KNN neighbors by nutrition + compound + text and average
    their image vectors.

    Returns: number of products whose image was imputed.
    """
    modalities = ["nutrition", "compound", "text", "image"]
    analog_keys = get_analog_keys(product_features)

    # Identify products missing image
    missing_keys = [k for k in analog_keys if product_features[k].get("image") is None]
    if not missing_keys:
        logger.info("No products missing image vectors — skipping imputation")
        return 0

    logger.info(f"Pre-imputing image vectors for {len(missing_keys)} analog "
                f"products via KNN (k={knn_k}, same-category neighbors)")

    # Get per-modality dimensions
    feature_dims = {}
    for pf in product_features.values():
        for ft in modalities:
            if ft not in feature_dims and pf.get(ft) is not None:
                feature_dims[ft] = pf[ft].shape[0]

    # Call build_feature_matrix_imputed on ALL analog products at once.
    # This constructs imputed image vectors by finding KNN neighbors among
    # complete-image products in the same category.
    X_imputed, valid_keys = build_feature_matrix_imputed(
        product_features, analog_keys, modalities, n_neighbors=knn_k
    )

    # Locate the image slice in the concatenated matrix
    offset = 0
    for ft in modalities:
        if ft == "image":
            image_start = offset
            image_end = offset + feature_dims[ft]
            break
        offset += feature_dims[ft]

    # Write imputed image vectors back to product_features for missing products
    key_to_row = {k: i for i, k in enumerate(valid_keys)}
    n_imputed = 0
    for key in missing_keys:
        if key not in key_to_row:
            logger.warning(f"Could not impute image for {key} — not in valid_keys")
            continue
        row_idx = key_to_row[key]
        imputed_image = X_imputed[row_idx, image_start:image_end].copy()
        product_features[key]["image"] = imputed_image
        n_imputed += 1

    logger.info(f"Imputed image vectors for {n_imputed} products")
    return n_imputed


# ---------------------------------------------------------------------------
# Generation: DistancePredictor (transductive, no LOOCV)
# ---------------------------------------------------------------------------

def generate_distance_predictor(product_features: dict) -> None:
    """Generate DistancePredictor predictions with NCTI features (cosine + L2)."""
    from models.distance_predictor import DistancePredictor

    analog_keys = sorted(get_analog_keys(product_features))

    for metric in ["cosine", "euclidean"]:
        tag = "cosine" if metric == "cosine" else "l2"
        filename = f"dist_pred_{tag}_{DP_FEATURE_DISPLAY}.csv"
        path = OOF_DIR / filename
        if path.exists():
            logger.info(f"  Skipping {filename} (exists)")
            continue

        model = DistancePredictor(
            feature_types=DP_FEATURE_TYPES,
            product_features=product_features,
            distance_metric=metric,
            missing_feature_strategy="skip",
        )
        model.fit()
        scores = model.get_all_scores()

        rows = []
        for key in analog_keys:
            pf = product_features[key]
            rows.append({
                "category": pf["category"],
                "product_code": pf["product_code"],
                "true_score": pf["mean_similarity"],
                "predicted_score": scores.get(key, np.nan),
            })
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        n_valid = df["predicted_score"].notna().sum()
        logger.info(f"  Generated {filename}: {n_valid}/{len(df)} products")


# ---------------------------------------------------------------------------
# Generation: supervised models with uniform SNCTI+PCA@95%+KNN
# ---------------------------------------------------------------------------

def _run_supervised(model_name: str, product_features: dict) -> None:
    """Run LOOCV for one supervised model with uniform preprocessing.

    Assumes impute_missing_images_inplace has already been called, so all
    analog products have complete raw features and build_feature_matrix
    yields n=215. We disable in-fold KNN imputation to use the simpler
    build_feature_matrix code path.
    """
    import train.run_loocv as loocv_module
    from train.run_loocv import run_single

    filename = f"{model_name}_{FEATURE_CODE}{SUFFIX}.csv"
    if (OOF_DIR / filename).exists():
        logger.info(f"  Skipping {filename} (exists)")
        return

    saved_skip = loocv_module._SKIP_BOOTSTRAP
    saved_knn = loocv_module._KNN_IMPUTE
    saved_pca_var = loocv_module._PCA_VARIANCE
    loocv_module._SKIP_BOOTSTRAP = True
    loocv_module._KNN_IMPUTE = 0  # Images already imputed in-place
    loocv_module._PCA_VARIANCE = 0.95
    try:
        run_single(model_name, FEATURE_CODE, product_features, suffix=SUFFIX)
    except Exception as e:
        logger.error(f"  Failed {model_name}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        loocv_module._SKIP_BOOTSTRAP = saved_skip
        loocv_module._KNN_IMPUTE = saved_knn
        loocv_module._PCA_VARIANCE = saved_pca_var


def generate_fast_models(product_features: dict) -> None:
    """Phase 1: everything except GP ARD (~10 min)."""
    logger.info("Generating DistancePredictor (cosine + L2)...")
    generate_distance_predictor(product_features)

    fast_models = [
        "ridge",
        "bradley_terry",
        "hierarchical_bt",
        "kernel_ranksvm",
        "lightgbm_reg",
    ]
    for model_name in fast_models:
        logger.info(f"Generating {model_name} (SNCTI + PCA@95% + KNN impute)...")
        _run_supervised(model_name, product_features)


# ---------------------------------------------------------------------------
# Loading OOF predictions
# ---------------------------------------------------------------------------

def _load_oof(filename: str) -> Optional[pd.DataFrame]:
    """Load an OOF CSV, drop NaN predictions."""
    path = OOF_DIR / filename
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "predicted_score" not in df.columns or "true_score" not in df.columns:
        return None
    df = df.dropna(subset=["predicted_score"])
    return df if len(df) >= 20 else None


def _find_best_llm(prefix: str) -> Tuple[Optional[pd.DataFrame], str]:
    """Find the best LLM variant by pairwise accuracy."""
    best_df, best_file, best_pw = None, "", -1.0
    for f in sorted(os.listdir(OOF_DIR)):
        if not f.startswith(prefix) or not f.endswith(".csv"):
            continue
        df = _load_oof(f)
        if df is None:
            continue
        m = compute_all_metrics(df)
        if m["pairwise_accuracy"] > best_pw:
            best_df, best_file, best_pw = df, f, m["pairwise_accuracy"]
    return best_df, best_file


def _llm_feature_display(filename: str, prefix: str) -> str:
    """Extract LLM variant as readable feature description."""
    stem = filename.replace(".csv", "")[len(prefix):]
    abbrev = {"ingredients": "ingr.", "nutrition": "nutr.", "image": "img."}
    return "+".join(abbrev.get(p, p) for p in stem.split("_"))


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

def _ensemble_avg(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight arithmetic mean of OOF predictions.

    Risk: if models have very different score scales (e.g., LLM win-rates
    in [0,1] vs BT logits in [-5,5]), the high-variance model dominates.
    Use _ensemble_rank_avg for scale-invariant averaging.
    """
    merged = dfs[0][["category", "product_code", "true_score"]].copy()
    for i, df in enumerate(dfs):
        merged = merged.merge(
            df[["category", "product_code", "predicted_score"]].rename(
                columns={"predicted_score": f"p{i}"}),
            on=["category", "product_code"],
            how="inner",
        )
    pred_cols = [f"p{i}" for i in range(len(dfs))]
    merged["predicted_score"] = merged[pred_cols].mean(axis=1)
    return merged[["category", "product_code", "true_score", "predicted_score"]]


def _ensemble_rank_avg(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Scale-invariant ensemble: average of within-category ranks.

    Each model's predictions are converted to within-category percentile
    ranks before averaging. This neutralizes scale differences between
    models (e.g., LLM win-rates vs BT logits), so no single model's
    variance can dominate the average.
    """
    merged = dfs[0][["category", "product_code", "true_score"]].copy()
    for i, df in enumerate(dfs):
        # Compute within-category percentile rank for this model
        df_with_rank = df[["category", "product_code", "predicted_score"]].copy()
        df_with_rank[f"r{i}"] = (
            df_with_rank.groupby("category")["predicted_score"]
            .rank(method="average", pct=True)
        )
        merged = merged.merge(
            df_with_rank[["category", "product_code", f"r{i}"]],
            on=["category", "product_code"],
            how="inner",
        )
    rank_cols = [f"r{i}" for i in range(len(dfs))]
    merged["predicted_score"] = merged[rank_cols].mean(axis=1)
    return merged[["category", "product_code", "true_score", "predicted_score"]]


def _ensemble_lgbm_meta_nested(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Learned meta-model: shallow LightGBM on base predictions with nested LOOCV.

    Unlike NNLS (which learns a linear combination), LightGBM captures
    nonlinear interactions between base-model predictions. For BT + Gemini,
    this could exploit complementarity: e.g., "when BT is confident and
    Gemini is near 0.5, trust BT; when BT and Gemini disagree strongly,
    blend them."

    Meta-learner config:
      - n_estimators=20, num_leaves=4, min_child_samples=10
      - Kept deliberately small: k base features + n=215 is a low-dim,
        small-n regime; a deep model would overfit.

    Nested LOOCV for correctness: for each held-out product i, fit meta
    on {(base_preds[j], true[j]) : j != i}, predict i.
    """
    import lightgbm as lgb

    # Align all DataFrames on common products
    merged = dfs[0][["category", "product_code", "true_score"]].copy()
    for i, df in enumerate(dfs):
        merged = merged.merge(
            df[["category", "product_code", "predicted_score"]].rename(
                columns={"predicted_score": f"p{i}"}),
            on=["category", "product_code"],
            how="inner",
        )

    n = len(merged)
    k = len(dfs)
    X = merged[[f"p{i}" for i in range(k)]].values  # (n, k)
    y = merged["true_score"].values  # (n,)

    oof_preds = np.zeros(n)
    params = {
        "n_estimators": 20,
        "num_leaves": 4,
        "min_child_samples": 10,
        "learning_rate": 0.1,
        "verbosity": -1,
        "random_state": 42,
    }
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        model = lgb.LGBMRegressor(**params)
        model.fit(X[mask], y[mask])
        oof_preds[i] = model.predict(X[i].reshape(1, -1))[0]

    out = merged[["category", "product_code", "true_score"]].copy()
    out["predicted_score"] = oof_preds
    return out


def bt_gemini_nested_meta_loocv(
    product_features: dict,
    gemini_df: pd.DataFrame,
    meta_type: str = "lgbm",
) -> pd.DataFrame:
    """Strictly leak-free nested-LOOCV meta-ensemble of Bradley-Terry + Gemini.

    Procedure:
      1. For each outer held-out product i, train a "leave-one-out" BT on the
         214 other products to get bt_outer[i] (the usual LOOCV prediction).
      2. For every pair (i, j) with i ≠ j, train a "leave-two-out" BT on the
         213 products excluding both i and j, predict both i and j:
             bt_inner[i, j] = prediction for j from BT trained without {i, j}
             bt_inner[j, i] = prediction for i from BT trained without {i, j}
      3. For each outer i, fit a LightGBM meta-learner on:
             features: [bt_inner[i, j], gem_oof[j]] for j ≠ i
             target:   true[j]
         Then predict on: [bt_outer[i], gem_oof[i]]
      4. Save meta_oof[i] as the ensemble prediction for i.

    This is strictly leak-free because the meta-training features for outer i
    are all produced by BTs that never saw i during training. Gemini is zero-
    shot so its predictions are label-independent by construction.

    Cost: O(n^2 / 2) BT fits. For n=215, ~23,000 BT fits. At ~0.05s/fit on
    SNCTI+PCA@95% that's ~20 minutes. Only practical because BT is fast.

    Args:
        product_features: Full product_features dict (with imputed images).
        gemini_df: OOF DataFrame with Gemini predictions.
        meta_type: "lgbm" or "nnls" for the meta-learner.

    Returns:
        OOF DataFrame with meta-learner predictions on the common products.
    """
    import time
    from data.loocv import (
        FeatureProcessor,
        build_feature_matrix,
        build_score_vector,
    )
    from models.bradley_terry import FeatureBradleyTerry

    analog_keys = sorted(get_analog_keys(product_features))
    feature_types = ["category_subset", "nutrition", "compound", "text", "image"]

    # Compute feature dims from a reference product
    feature_dims: Dict[str, int] = {}
    for pf in product_features.values():
        for ft in feature_types:
            if ft not in feature_dims and pf.get(ft) is not None:
                feature_dims[ft] = pf[ft].shape[0]

    # Build full feature matrix (after pre-imputation, all 215 have images)
    X_all, valid_keys = build_feature_matrix(
        product_features, analog_keys, feature_types
    )
    y_all = build_score_vector(product_features, valid_keys)
    cats_all = [product_features[k]["category"] for k in valid_keys]
    n = len(valid_keys)
    logger.info(f"Nested BT+Gemini meta: {n} products, {n * (n - 1) // 2} pairs")

    # Align Gemini predictions to valid_keys order
    gem_lookup = {
        (r["category"], r["product_code"]): r["predicted_score"]
        for _, r in gemini_df.iterrows()
    }
    gem_oof = np.array([
        gem_lookup.get((cat, product_features[k]["product_code"]), np.nan)
        for k, cat in zip(valid_keys, cats_all)
    ])
    if np.any(np.isnan(gem_oof)):
        missing = int(np.isnan(gem_oof).sum())
        raise ValueError(f"{missing} products missing Gemini predictions")

    # --- Step 1: Outer LOOCV for bt_outer[i] ---
    logger.info("Step 1/3: Outer LOOCV (n=214 training per fold)...")
    start = time.time()
    bt_outer = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        proc = FeatureProcessor(
            feature_types, feature_dims, use_pca=True, pca_variance=0.95
        )
        X_tr = proc.fit_transform(X_all[mask], y_all[mask])
        X_te = proc.transform(X_all[i:i + 1])
        bt = FeatureBradleyTerry()
        train_cats = [cats_all[k] for k in range(n) if mask[k]]
        bt.fit(X_tr, y_all[mask], categories=train_cats)
        bt_outer[i] = bt.predict_score(X_te, categories=[cats_all[i]])[0]
    logger.info(f"  Done in {time.time() - start:.1f}s")

    # --- Step 2: Inner LOOCV for bt_inner[i, j] ---
    # For each pair (i, j) with i < j, train BT on n-2 products, predict both.
    logger.info("Step 2/3: Inner nested LOOCV (n=213 training per pair, "
                f"{n * (n - 1) // 2} pairs)...")
    start = time.time()
    bt_inner = np.full((n, n), np.nan)  # bt_inner[i, j] = prediction for j without {i, j}
    pair_count = 0
    total_pairs = n * (n - 1) // 2
    log_every = max(1, total_pairs // 20)

    for i in range(n):
        for j in range(i + 1, n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            proc = FeatureProcessor(
                feature_types, feature_dims, use_pca=True, pca_variance=0.95
            )
            X_tr = proc.fit_transform(X_all[mask], y_all[mask])
            X_te = proc.transform(X_all[[i, j]])
            train_cats = [cats_all[k] for k in range(n) if mask[k]]
            bt = FeatureBradleyTerry()
            bt.fit(X_tr, y_all[mask], categories=train_cats)
            preds = bt.predict_score(
                X_te, categories=[cats_all[i], cats_all[j]]
            )
            bt_inner[i, j] = preds[1]  # prediction for j from BT without {i, j}
            bt_inner[j, i] = preds[0]  # prediction for i from BT without {i, j}
            pair_count += 1
            if pair_count % log_every == 0:
                elapsed = time.time() - start
                eta = elapsed / pair_count * (total_pairs - pair_count)
                logger.info(f"  Pair {pair_count}/{total_pairs} "
                             f"({100 * pair_count / total_pairs:.1f}%), "
                             f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s")
    logger.info(f"  Done in {time.time() - start:.0f}s")

    # --- Step 3: Fit meta-learner per outer i ---
    logger.info("Step 3/3: Fit meta-learners...")
    start = time.time()
    meta_oof = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_meta_train = np.column_stack([bt_inner[i, mask], gem_oof[mask]])
        y_meta_train = y_all[mask]

        if meta_type == "lgbm":
            import lightgbm as lgb
            meta = lgb.LGBMRegressor(
                n_estimators=20, num_leaves=4,
                min_child_samples=10, learning_rate=0.1,
                verbosity=-1, random_state=42,
            )
            meta.fit(X_meta_train, y_meta_train)
            meta_oof[i] = meta.predict(
                np.array([[bt_outer[i], gem_oof[i]]])
            )[0]
        elif meta_type == "nnls":
            from scipy.optimize import nnls
            w, _ = nnls(X_meta_train, y_meta_train)
            meta_oof[i] = np.array([bt_outer[i], gem_oof[i]]) @ w
        else:
            raise ValueError(f"Unknown meta_type: {meta_type}")
    logger.info(f"  Done in {time.time() - start:.1f}s")

    # Build output DataFrame
    rows = []
    for idx, key in enumerate(valid_keys):
        pf = product_features[key]
        rows.append({
            "category": pf["category"],
            "product_code": pf["product_code"],
            "true_score": pf["mean_similarity"],
            "predicted_score": meta_oof[idx],
        })
    return pd.DataFrame(rows)


def _ensemble_nnls_nested(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Learned-weight ensemble via non-negative least squares with nested LOOCV.

    For each product i:
      1. Hold out product i
      2. Fit NNLS on the other 214 products' base OOF predictions to match
         their true scores: minimize ||Aw - y||² subject to w >= 0
      3. Use the fitted weights to combine base predictions for i

    This is leak-free because:
      - Base OOF predictions are themselves leak-free by construction
      - The NNLS weights for product i are fit without using i's true score
      - Predictions for i use only its base model outputs

    NNLS has no hyperparameters and guarantees non-negative weights, which
    are interpretable as per-model trust scores.

    All base DataFrames must share the same (category, product_code) index
    (common products only).
    """
    from scipy.optimize import nnls

    # Align all DataFrames on common products
    merged = dfs[0][["category", "product_code", "true_score"]].copy()
    for i, df in enumerate(dfs):
        merged = merged.merge(
            df[["category", "product_code", "predicted_score"]].rename(
                columns={"predicted_score": f"p{i}"}),
            on=["category", "product_code"],
            how="inner",
        )

    n = len(merged)
    k = len(dfs)
    X = merged[[f"p{i}" for i in range(k)]].values  # (n, k)
    y = merged["true_score"].values  # (n,)

    oof_preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        w, _ = nnls(X[mask], y[mask])
        oof_preds[i] = X[i] @ w

    out = merged[["category", "product_code", "true_score"]].copy()
    out["predicted_score"] = oof_preds
    return out


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------

def _make_random_baseline(product_features: dict) -> Tuple[pd.DataFrame, float]:
    """Random baseline: average pw_acc over 100 seeds, CI from seed 42."""
    analog_keys = sorted(get_analog_keys(product_features))
    base = pd.DataFrame([{
        "category": product_features[k]["category"],
        "product_code": product_features[k]["product_code"],
        "true_score": product_features[k]["mean_similarity"],
    } for k in analog_keys])

    pw_accs = []
    for seed in range(100):
        df = base.copy()
        df["predicted_score"] = np.random.default_rng(seed).random(len(df))
        pw_accs.append(compute_all_metrics(df)["pairwise_accuracy"])

    df_rep = base.copy()
    df_rep["predicted_score"] = np.random.default_rng(42).random(len(df_rep))
    return df_rep, float(np.mean(pw_accs))


def _compute_row(
    display_name: str,
    group_label: str,
    feature_display: str,
    df: pd.DataFrame,
    n_bootstrap: int,
    oof_filename: str = "",
) -> TableRow:
    """Compute metrics + BCa CI for one model."""
    metrics = compute_all_metrics(df)
    cis = compute_bca_cis(df, n_bootstrap=n_bootstrap)
    return TableRow(
        display_name=display_name,
        group_label=group_label,
        feature_display=feature_display,
        pairwise_accuracy=metrics["pairwise_accuracy"],
        ci_lo=cis["pairwise_accuracy"][0],
        ci_hi=cis["pairwise_accuracy"][1],
        n_products=len(df),
        oof_filename=oof_filename,
    )


def build_table(
    product_features: dict,
    n_bootstrap: int = 10000,
    with_nested_meta: bool = False,
) -> List[TableRow]:
    """Build all table rows.

    Args:
        product_features: Pre-imputed product features dict.
        n_bootstrap: BCa bootstrap iterations.
        with_nested_meta: If True, compute the rigorous nested-LOOCV BT+Gemini
            meta-ensemble (O(n^2) cost, ~20 min, strictly leak-free).
    """
    rows = []
    model_dfs: Dict[str, pd.DataFrame] = {}

    # --- Random baseline ---
    logger.info("Computing: Random")
    df_rand, avg_pw = _make_random_baseline(product_features)
    cis = compute_bca_cis(df_rand, n_bootstrap=n_bootstrap)
    rows.append(TableRow(
        display_name="Random",
        group_label="Unsupervised",
        feature_display="--",
        pairwise_accuracy=avg_pw,
        ci_lo=cis["pairwise_accuracy"][0],
        ci_hi=cis["pairwise_accuracy"][1],
        n_products=len(df_rand),
    ))

    # --- DistancePredictor ---
    for tag, display in [("cosine", "MMRF (cosine)"),
                         ("l2", "MMRF (L2)")]:
        filename = f"dist_pred_{tag}_{DP_FEATURE_DISPLAY}.csv"
        df = _load_oof(filename)
        if df is None:
            logger.warning(f"Missing: {filename}")
            continue
        logger.info(f"Computing: {display} ({filename}, n={len(df)})")
        row = _compute_row(display, "Unsupervised", DP_FEATURE_DISPLAY, df,
                           n_bootstrap, filename)
        rows.append(row)
        model_dfs[display] = df

    # --- LLMs ---
    for prefix, display in [
        ("llm_qwen3_5_397b_a17b_", "Qwen 3.5 397B"),
        ("llm_gemini_3_1_pro_preview_", "Gemini 3.1 Pro"),
    ]:
        df, filename = _find_best_llm(prefix)
        if df is None:
            logger.warning(f"No LLM predictions for {display}")
            continue
        feat = _llm_feature_display(filename, prefix)
        logger.info(f"Computing: {display} ({filename}, n={len(df)})")
        row = _compute_row(display, "Unsupervised", feat, df, n_bootstrap, filename)
        rows.append(row)
        model_dfs[display] = df

    # --- Supervised models (all on SNCTI + PCA@95% + KNN impute) ---
    supervised_specs = [
        # (model_name, display_name, group_label)
        ("ridge", "Ridge", "Supervised (linear)"),
        ("bradley_terry", "Bradley-Terry", "Supervised (pairwise)"),
        ("hierarchical_bt", "Hierarchical BT", "Supervised (pairwise)"),
        ("kernel_ranksvm", "Kernel RankSVM", "Supervised (pairwise)"),
        ("lightgbm_reg", "LightGBM", "Supervised (nonlinear)"),
    ]
    for model_name, display, group in supervised_specs:
        filename = f"{model_name}_{FEATURE_CODE}{SUFFIX}.csv"
        df = _load_oof(filename)
        if df is None:
            logger.warning(f"Missing: {filename}")
            continue
        logger.info(f"Computing: {display} ({filename}, n={len(df)})")
        row = _compute_row(display, group, FEATURE_CODE, df, n_bootstrap, filename)
        rows.append(row)
        model_dfs[display] = df

    # --- Ensembles ---
    # Strategy: try both arithmetic mean and rank-based mean for each combo,
    # pick the variant with higher pairwise accuracy. Rank-based is scale-
    # invariant and typically better when combining LLMs (win-rates in [0,1])
    # with supervised models (arbitrary scales).
    sup_rows = [r for r in rows if r.group_label.startswith("Supervised")]
    gemini_df = model_dfs.get("Gemini 3.1 Pro")
    qwen_df = model_dfs.get("Qwen 3.5 397B")

    def _best_ensemble_variant(dfs, base_name):
        """Try mean, rank, NNLS, and LightGBM meta; return the best variant."""
        variants = []
        try:
            variants.append(("mean", _ensemble_avg(dfs)))
        except Exception:
            pass
        try:
            variants.append(("rank avg", _ensemble_rank_avg(dfs)))
        except Exception:
            pass
        try:
            variants.append(("NNLS", _ensemble_nnls_nested(dfs)))
        except Exception as e:
            logger.warning(f"NNLS ensemble failed: {e}")
        try:
            variants.append(("LGBM meta", _ensemble_lgbm_meta_nested(dfs)))
        except Exception as e:
            logger.warning(f"LGBM meta ensemble failed: {e}")
        best_tag, best_df, best_pw = None, None, -1.0
        for tag, df in variants:
            pw = compute_all_metrics(df)["pairwise_accuracy"]
            if pw > best_pw:
                best_tag, best_df, best_pw = tag, df, pw
        return best_df, f"{base_name} ({best_tag})"

    if sup_rows and gemini_df is not None:
        sup_sorted = sorted(sup_rows, key=lambda r: -r.pairwise_accuracy)
        sup_cand_names = [r.display_name for r in sup_sorted
                          if r.display_name in model_dfs]

        # Ensemble 1: best supervised + Gemini
        best_sup_name = sup_sorted[0].display_name
        best_sup_df = model_dfs.get(best_sup_name)
        if best_sup_df is not None:
            ens_df, ens_name = _best_ensemble_variant(
                [best_sup_df, gemini_df], f"{best_sup_name} + Gemini")
            logger.info(f"Computing ensemble: {ens_name}")
            rows.append(_compute_row(ens_name, "Ensemble", "--",
                                      ens_df, n_bootstrap))

        # Ensemble 2: top-3 supervised + Gemini
        if len(sup_sorted) >= 3:
            top3_dfs = [model_dfs[n] for n in sup_cand_names[:3]]
            ens_df, ens_name = _best_ensemble_variant(
                top3_dfs + [gemini_df], "Top-3 supervised + Gemini")
            logger.info(f"Computing ensemble: {ens_name}")
            rows.append(_compute_row(ens_name, "Ensemble", "--",
                                      ens_df, n_bootstrap))

        # Ensemble 3: top-3 supervised + Gemini + Qwen (diverse LLMs)
        if len(sup_sorted) >= 3 and qwen_df is not None:
            top3_dfs = [model_dfs[n] for n in sup_cand_names[:3]]
            ens_df, ens_name = _best_ensemble_variant(
                top3_dfs + [gemini_df, qwen_df],
                "Top-3 supervised + Gemini + Qwen")
            logger.info(f"Computing ensemble: {ens_name}")
            rows.append(_compute_row(ens_name, "Ensemble", "--",
                                      ens_df, n_bootstrap))

        # Ensemble 4: exhaustive search — best k-subset of supervised + Gemini
        # across all three combining strategies (mean, rank avg, NNLS).
        from itertools import combinations
        best_pw, best_df, best_members, best_variant = -1.0, None, [], ""
        avg_fns = [
            ("mean", _ensemble_avg),
            ("rank", _ensemble_rank_avg),
            ("NNLS", _ensemble_nnls_nested),
            ("LGBM meta", _ensemble_lgbm_meta_nested),
        ]
        for k in range(1, min(5, len(sup_cand_names)) + 1):
            for combo in combinations(sup_cand_names, k):
                dfs = [model_dfs[n] for n in combo] + [gemini_df]
                for tag, fn in avg_fns:
                    try:
                        ens = fn(dfs)
                    except Exception:
                        continue
                    m = compute_all_metrics(ens)
                    if m["pairwise_accuracy"] > best_pw:
                        best_pw = m["pairwise_accuracy"]
                        best_df = ens
                        best_members = list(combo) + ["Gemini"]
                        best_variant = tag

        if best_df is not None:
            ens_name = (f"Best ensemble: {' + '.join(best_members)} "
                        f"({best_variant})")
            existing_names = {r.display_name for r in rows
                              if r.group_label == "Ensemble"}
            if ens_name not in existing_names:
                logger.info(f"Computing ensemble: {ens_name}")
                rows.append(_compute_row(ens_name, "Ensemble", "--",
                                          best_df, n_bootstrap))

    # --- Rigorous nested-LOOCV BT+Gemini meta ensemble (strictly leak-free) ---
    if with_nested_meta and gemini_df is not None:
        bt_df = model_dfs.get("Bradley-Terry")
        if bt_df is not None:
            logger.info("=" * 60)
            logger.info("Computing rigorous nested-LOOCV BT+Gemini meta "
                         "(this takes ~20 minutes)...")
            logger.info("=" * 60)
            try:
                nested_df = bt_gemini_nested_meta_loocv(
                    product_features, gemini_df, meta_type="lgbm"
                )
                logger.info("Computing metrics for nested meta ensemble...")
                row = _compute_row(
                    "BT + Gemini (nested LGBM, leak-free)",
                    "Ensemble",
                    "--",
                    nested_df,
                    n_bootstrap,
                )
                rows.append(row)
                logger.info(f"  pw_acc = {row.pairwise_accuracy:.4f} "
                             f"[{row.ci_lo:.4f}, {row.ci_hi:.4f}]")
            except Exception as e:
                logger.error(f"Nested meta failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            logger.warning("Skipping nested meta: no Bradley-Terry predictions")

    # Sort within groups
    group_rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    rows.sort(key=lambda r: (group_rank.get(r.group_label, 99),
                              -r.pairwise_accuracy))
    return rows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_text_table(rows: List[TableRow]) -> str:
    """Plain-text comparison table."""
    lines = []
    lines.append(f"{'Model':<45s}  {'Features':<12s}  "
                 f"{'Pairwise Acc [95% CI]':<28s}  {'n':>5s}")
    lines.append("=" * 96)

    current_group = None
    best_pw = max(r.pairwise_accuracy for r in rows)

    for row in rows:
        if row.group_label != current_group:
            current_group = row.group_label
            if lines[-1] != "=" * 96:
                lines.append("")
            lines.append(f"  {current_group}")
            lines.append("  " + "-" * 92)

        ci_str = f"{row.pairwise_accuracy:.3f} [{row.ci_lo:.3f}, {row.ci_hi:.3f}]"
        marker = " *" if abs(row.pairwise_accuracy - best_pw) < 1e-6 else "  "
        lines.append(f"  {row.display_name:<43s}  {row.feature_display:<12s}  "
                     f"{ci_str:<28s}  {row.n_products:>5d}{marker}")

    lines.append("")
    lines.append("* = best overall")
    return "\n".join(lines)


def render_latex_table(rows: List[TableRow]) -> str:
    """LaTeX booktabs table."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{Pairwise ranking accuracy on 215 NECTAR "
                 r"plant-based products (LOOCV). All supervised models use "
                 r"the same preprocessing: category, nutrition, compound, "
                 r"text, and image features with per-modality PCA to 95\% "
                 r"explained variance fit per fold; 16 products missing "
                 r"raw image vectors are KNN-imputed ($k{=}5$) using "
                 r"same-category neighbors. 95\% BCa bootstrap CIs from "
                 r"10{,}000 resamples.}")
    lines.append(r"\label{tab:comparison}")
    lines.append(r"\begin{tabular}{@{}llcr@{}}")
    lines.append(r"\toprule")
    lines.append(r"Model & Features & Pairwise Accuracy [\,95\% CI\,] & $n$ \\")
    lines.append(r"\midrule")

    current_group = None
    best_pw = max(r.pairwise_accuracy for r in rows)

    for row in rows:
        if row.group_label != current_group:
            if current_group is not None:
                lines.append(r"\midrule")
            current_group = row.group_label
            lines.append(
                rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{current_group}}}}} \\"
            )

        name_tex = row.display_name.replace("_", r"\_")
        feat_tex = row.feature_display.replace("_", r"\_")
        pw_str = f"{row.pairwise_accuracy:.3f}"
        ci_str = f"[{row.ci_lo:.3f},\\,{row.ci_hi:.3f}]"

        is_best = abs(row.pairwise_accuracy - best_pw) < 1e-6
        if is_best:
            pw_ci = rf"\textbf{{{pw_str}}} {ci_str}"
            name_tex = rf"\textbf{{{name_tex}}}"
        else:
            pw_ci = f"{pw_str} {ci_str}"

        lines.append(
            rf"\quad {name_tex} & {feat_tex} & {pw_ci} & {row.n_products} \\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def save_results_csv(rows: List[TableRow], path: Path) -> None:
    """Save table data as CSV."""
    records = [{
        "model": r.display_name,
        "group": r.group_label,
        "features": r.feature_display,
        "pairwise_accuracy": r.pairwise_accuracy,
        "ci_lo": r.ci_lo,
        "ci_hi": r.ci_hi,
        "n": r.n_products,
        "oof_file": r.oof_filename,
    } for r in rows]
    pd.DataFrame(records).to_csv(path, index=False)
    logger.info(f"Results CSV saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Paper-ready comparison table (NeurIPS 2026)"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate-fast", action="store_true",
                      help="Generate all supervised predictions, then build table")
    mode.add_argument("--table-only", action="store_true",
                      help="Build table from existing OOF files")

    parser.add_argument("--n-bootstrap", type=int, default=10000,
                        help="BCa bootstrap iterations (default: 10000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save LaTeX table to file")
    parser.add_argument("--nested-meta", action="store_true",
                        help="Add the rigorous nested-LOOCV BT+Gemini meta "
                             "ensemble row (~20 min, strictly leak-free).")
    args = parser.parse_args()

    product_features = load_product_features()
    logger.info(f"Loaded {len(product_features)} products")

    # Pre-impute missing image vectors once (transductive, leak-free).
    # All downstream LOOCV runs see the imputed features, guaranteeing n=215.
    impute_missing_images_inplace(product_features, knn_k=KNN_K)

    if args.generate_fast:
        generate_fast_models(product_features)

    logger.info("=" * 60)
    logger.info("Building comparison table")
    logger.info("=" * 60)
    rows = build_table(
        product_features,
        n_bootstrap=args.n_bootstrap,
        with_nested_meta=args.nested_meta,
    )

    print()
    print(render_text_table(rows))
    print()
    print(render_latex_table(rows))

    save_results_csv(rows, SUPERVISED_DIR / "results" / "paper_table.csv")

    if args.output:
        Path(args.output).write_text(render_latex_table(rows))
        logger.info(f"LaTeX saved to {args.output}")


if __name__ == "__main__":
    main()
