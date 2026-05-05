"""Compute the {model} + Gemini NNLS ensemble top-line for one supervised model.

Driver for the canonical NNLS configuration (BT + Gemini, .6829 pw-acc)
and the per-model variants in `table_per_model_nnls.tex`. Default
SUPERVISED_MODEL = bradley_terry; set to one of {hierarchical_bt,
ridge, lightgbm_reg, kernel_ranksvm} to compute the same ensemble with
that model in the supervised role.

Steps:
  1. Load `data/product_features.pkl`, KNN-impute missing image
     embeddings.
  2. Run the bench-config supervised LOOCV (SNCTI, PCA@95%,
     skip-bootstrap). Output: `{model}_SNCTI_v4.csv`.
  3. Build the nested base for ensembling: n outer LOOCV folds +
     n(n−1)/2 inner LOOCV folds. Cached to
     `results/cache/nested_{tag}_gemini_base_v4.npz`.
  4. Run scipy.optimize.nnls per fold. Output:
     `nested_{tag}_gemini_nnls_v4.csv`. For BT, also write rank-avg
     and equal-mean variants.
  5. Print pairwise accuracy + 95% BCa CI; compare to the canonical
     BT+Gemini NNLS row in Table~1 if present on disk.

Usage:
    cd food_similarity
    python -m train.compute_per_model_nnls                           # BT
    SUPERVISED_MODEL=hierarchical_bt python -m train.compute_per_model_nnls
"""
from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERVISED_DIR))

from data.loocv import (  # noqa: E402
    FeatureProcessor,
    build_feature_matrix,
    build_score_vector,
    get_analog_keys,
    load_product_features,
)
from evaluation.bootstrap import compute_bca_cis  # noqa: E402
from evaluation.metrics import compute_all_metrics  # noqa: E402
from models.bradley_terry import FeatureBradleyTerry  # noqa: E402
from models.hierarchical_bt import HierarchicalBT  # noqa: E402
from models.ridge import RidgeRegressor  # noqa: E402
from models.lightgbm_reg import LightGBMRegressor  # noqa: E402

# Allow swapping the supervised model via env var. Default is BT (paper
# choice); other options support comparison tables (pointwise models like
# ridge / lightgbm don't take `categories` in fit/predict_score).
_MODEL_CLASS = {
    "bradley_terry": (FeatureBradleyTerry, True),    # (cls, is_pairwise)
    "hierarchical_bt": (HierarchicalBT, True),
    "ridge": (RidgeRegressor, False),
    "lightgbm_reg": (LightGBMRegressor, False),
}
# kernel_ranksvm is imported lazily because it has heavy sklearn deps
if os.environ.get("SUPERVISED_MODEL") == "kernel_ranksvm":
    from models.kernel_ranksvm import KernelRankSVM  # noqa: E402
    _MODEL_CLASS["kernel_ranksvm"] = (KernelRankSVM, True)
_SUPERVISED_MODEL = os.environ.get("SUPERVISED_MODEL", "bradley_terry")
if _SUPERVISED_MODEL not in _MODEL_CLASS:
    raise ValueError(f"Unsupported SUPERVISED_MODEL: {_SUPERVISED_MODEL!r}")
SupervisedModel, _IS_PAIRWISE = _MODEL_CLASS[_SUPERVISED_MODEL]


def _fit_supervised(model, X, y, cats):
    if _IS_PAIRWISE:
        model.fit(X, y, categories=cats)
    else:
        model.fit(X, y)


def _predict_supervised(model, X, cats):
    if _IS_PAIRWISE:
        return model.predict_score(X, categories=cats)
    return model.predict_score(X)
from train.paper_table import impute_missing_images_inplace  # noqa: E402
from train.run_loocv import run_single  # noqa: E402
import train.run_loocv as loocv_module  # noqa: E402

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OOF_DIR = SUPERVISED_DIR / "results" / "oof_predictions"
CACHE_DIR = SUPERVISED_DIR / "results" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_FEATURE_TYPES_BY_CODE = {
    "SNCTI": ["category_subset", "nutrition", "compound", "text", "image"],
    "SNWTI": ["category_subset", "nutrition", "compound_weighted", "text", "image"],
    "SNCWTI": ["category_subset", "nutrition", "compound", "compound_weighted", "text", "image"],
    "SNDCTI": ["category_subset", "nutrition", "sensory", "compound", "text", "image"],
    "SNCTID": ["category_subset", "nutrition", "compound", "text", "image", "sensory"],
    "SNCT": ["category_subset", "nutrition", "compound", "text"],
    "SNTI": ["category_subset", "nutrition", "text", "image"],
}
# Env-overridable knobs for the cheap-experiment hyperparameter sweep:
# FEATURE_CODE  — bench feature subset (default SNCTI; SNWTI swaps the
#                 top-3 compound feature for the rank-weighted variant)
# PCA_VARIANCE  — per-modality PCA variance threshold (default 0.95)
FEATURE_CODE = os.environ.get("FEATURE_CODE", "SNCTI")
if FEATURE_CODE not in _FEATURE_TYPES_BY_CODE:
    raise ValueError(f"Unsupported FEATURE_CODE for swap: {FEATURE_CODE!r}")
FEATURE_TYPES = _FEATURE_TYPES_BY_CODE[FEATURE_CODE]
PCA_VARIANCE = float(os.environ.get("PCA_VARIANCE", "0.95"))
KNN_K = 5

_VARIANT = os.environ.get("V4_VARIANT", "v4")
_PF_OVERRIDE = os.environ.get("PRODUCT_FEATURES_PATH")

# Short tag per supervised model so the per-model NNLS comparison table
# does not have all 5 models clobbering nested_bt_gemini_nnls_v4.csv.
_MODEL_SHORT = {
    "bradley_terry":   "bt",
    "hierarchical_bt": "hbt",
    "ridge":           "ridge",
    "lightgbm_reg":    "lgbm",
    "kernel_ranksvm":  "ksvm",
}[_SUPERVISED_MODEL]

V4_BT_FILE   = f"{_SUPERVISED_MODEL}_{FEATURE_CODE}_{_VARIANT}.csv"
V4_NNLS_FILE = f"nested_{_MODEL_SHORT}_gemini_nnls_{_VARIANT}.csv"
V4_CACHE     = CACHE_DIR / f"nested_{_MODEL_SHORT}_gemini_base_{_VARIANT}.npz"
GEMINI_FILE  = "llm_gemini_3_1_pro_preview_ingredients_image.csv"

CANONICAL_BT_FILE = "bradley_terry_SNCTI_bench.csv"
CANONICAL_NNLS_FILE = "nested_bt_gemini_nnls.csv"


# ---------------------------------------------------------------------------
# Step 1: BT LOOCV with bench config, save as _v4
# ---------------------------------------------------------------------------

def run_bt_loocv_bench(product_features: dict) -> Path:
    out_path = OOF_DIR / V4_BT_FILE
    if out_path.exists():
        logger.info(f"Skipping BT LOOCV; {out_path.name} already exists")
        return out_path

    saved_skip = loocv_module._SKIP_BOOTSTRAP
    saved_knn = loocv_module._KNN_IMPUTE
    saved_pca = loocv_module._PCA_VARIANCE
    loocv_module._SKIP_BOOTSTRAP = True
    loocv_module._KNN_IMPUTE = 0  # images pre-imputed in-place
    loocv_module._PCA_VARIANCE = PCA_VARIANCE
    try:
        run_single(_SUPERVISED_MODEL, FEATURE_CODE, product_features, suffix=f"_{_VARIANT}")
    finally:
        loocv_module._SKIP_BOOTSTRAP = saved_skip
        loocv_module._KNN_IMPUTE = saved_knn
        loocv_module._PCA_VARIANCE = saved_pca

    if not out_path.exists():
        raise RuntimeError(f"Expected output {out_path} was not produced")
    return out_path


# ---------------------------------------------------------------------------
# Step 2: nested BT base features (n + n*(n-1)/2 BT fits) + NNLS meta
# ---------------------------------------------------------------------------

def compute_nested_base_v4(product_features: dict) -> dict:
    """Compute bt_outer, bt_inner, gem_oof, y_all and cache to v4 file."""
    if V4_CACHE.exists():
        logger.info(f"Loading cached v4 base features from {V4_CACHE}")
        d = np.load(V4_CACHE, allow_pickle=True)
        return {
            "bt_outer": d["bt_outer"],
            "bt_inner": d["bt_inner"],
            "gem_oof": d["gem_oof"],
            "y_all": d["y_all"],
            "categories": d["categories"],
            "product_keys": [tuple(k) for k in d["product_keys"]],
        }

    analog_keys = sorted(get_analog_keys(product_features))
    feature_dims = {}
    for pf in product_features.values():
        for ft in FEATURE_TYPES:
            if ft not in feature_dims and pf.get(ft) is not None:
                feature_dims[ft] = pf[ft].shape[0]

    X_all, valid_keys = build_feature_matrix(product_features, analog_keys, FEATURE_TYPES)
    y_all = build_score_vector(product_features, valid_keys)
    cats_all = np.array([product_features[k]["category"] for k in valid_keys], dtype=object)
    n = len(valid_keys)
    logger.info(f"v4 nested base: n={n}, X={X_all.shape}")

    # Align Gemini OOF
    gem_df = pd.read_csv(OOF_DIR / GEMINI_FILE).dropna(subset=["predicted_score"])
    gem_lookup = {
        (r["category"], int(r["product_code"])): r["predicted_score"]
        for _, r in gem_df.iterrows()
    }
    gem_oof = np.array([
        gem_lookup.get((cat, int(product_features[k]["product_code"])), np.nan)
        for k, cat in zip(valid_keys, cats_all)
    ])
    if np.any(np.isnan(gem_oof)):
        missing = int(np.isnan(gem_oof).sum())
        raise ValueError(f"{missing} products missing Gemini predictions")

    # Outer LOOCV
    logger.info("Step 1/2: Outer LOOCV (n=214 train per fold)...")
    t0 = time.time()
    bt_outer = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        proc = FeatureProcessor(FEATURE_TYPES, feature_dims, use_pca=True, pca_variance=PCA_VARIANCE)
        X_tr = proc.fit_transform(X_all[mask], y_all[mask])
        X_te = proc.transform(X_all[i:i + 1])
        bt = SupervisedModel()  # name kept for back-compat; can be any supervised model
        train_cats = [cats_all[k] for k in range(n) if mask[k]]
        _fit_supervised(bt, X_tr, y_all[mask], train_cats)
        bt_outer[i] = _predict_supervised(bt, X_te, [cats_all[i]])[0]
    logger.info(f"  Outer done in {time.time()-t0:.0f}s")

    # Inner nested LOOCV — parallelized over independent (i, j) pair fits.
    # JOBLIB_N_JOBS env var lets multiple variants share cores when run in
    # parallel (e.g. set 4 per variant when running 3 variants together).
    total_pairs = n * (n - 1) // 2
    if os.environ.get("JOBLIB_N_JOBS"):
        n_jobs = int(os.environ["JOBLIB_N_JOBS"])
    else:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    logger.info(f"Step 2/2: Inner nested LOOCV ({total_pairs} pairs, n_jobs={n_jobs})...")
    t0 = time.time()
    pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def _fit_pair(i: int, j: int):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        proc = FeatureProcessor(FEATURE_TYPES, feature_dims, use_pca=True, pca_variance=PCA_VARIANCE)
        X_tr = proc.fit_transform(X_all[mask], y_all[mask])
        X_te = proc.transform(X_all[[i, j]])
        train_cats = [cats_all[k] for k in range(n) if mask[k]]
        bt = SupervisedModel()  # name kept for back-compat; can be any supervised model
        _fit_supervised(bt, X_tr, y_all[mask], train_cats)
        preds = _predict_supervised(bt, X_te, [cats_all[i], cats_all[j]])
        return (i, j, float(preds[1]), float(preds[0]))

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(_fit_pair)(i, j) for (i, j) in pair_indices
    )

    bt_inner = np.full((n, n), np.nan)
    for i, j, pij, pji in results:
        bt_inner[i, j] = pij  # prediction for j from BT trained without {i, j}
        bt_inner[j, i] = pji  # prediction for i from BT trained without {i, j}
    logger.info(f"  Inner done in {time.time()-t0:.0f}s ({total_pairs} pairs)")

    np.savez(
        V4_CACHE,
        bt_outer=bt_outer,
        bt_inner=bt_inner,
        gem_oof=gem_oof,
        y_all=y_all,
        categories=cats_all,
        product_keys=np.array([list(k) for k in valid_keys], dtype=object),
    )
    logger.info(f"Saved v4 cache to {V4_CACHE}")

    return {
        "bt_outer": bt_outer,
        "bt_inner": bt_inner,
        "gem_oof": gem_oof,
        "y_all": y_all,
        "categories": cats_all,
        "product_keys": valid_keys,
    }


def run_nnls_meta(base: dict) -> np.ndarray:
    from scipy.optimize import nnls
    bt_outer = base["bt_outer"]
    bt_inner = base["bt_inner"]
    gem_oof = base["gem_oof"]
    y_all = base["y_all"]
    n = len(bt_outer)

    oof = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_meta = np.column_stack([bt_inner[i, mask], gem_oof[mask]])
        y_meta = y_all[mask]
        w, _ = nnls(X_meta, y_meta)
        oof[i] = np.array([bt_outer[i], gem_oof[i]]) @ w
    return oof


def run_equal_mean_meta(base: dict) -> np.ndarray:
    """Simple (bt_outer + gem) / 2 — leak-free."""
    return 0.5 * (base["bt_outer"] + base["gem_oof"])


def run_rank_avg_meta(base: dict) -> np.ndarray:
    """Within-category rank-percentile average — leak-free, scale-invariant."""
    bt_outer = base["bt_outer"]
    gem_oof = base["gem_oof"]
    cats = base["categories"]
    df = pd.DataFrame({"category": cats, "bt": bt_outer, "gem": gem_oof})
    df["r_bt"] = df.groupby("category")["bt"].rank(pct=True)
    df["r_gem"] = df.groupby("category")["gem"].rank(pct=True)
    return 0.5 * (df["r_bt"].values + df["r_gem"].values)


def save_oof(base: dict, predictions: np.ndarray, filename: str) -> pd.DataFrame:
    rows = []
    for i, key in enumerate(base["product_keys"]):
        cat, code = key
        rows.append({
            "category": cat,
            "product_code": int(code),
            "true_score": base["y_all"][i],
            "predicted_score": predictions[i],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OOF_DIR / filename, index=False)
    return df


# ---------------------------------------------------------------------------
# Step 3: pairwise accuracy + 95% BCa CI
# ---------------------------------------------------------------------------

_SKIP_BCA = os.environ.get("SKIP_BCA", "").lower() in ("1", "true", "yes")


def report_metrics(df: pd.DataFrame, label: str) -> tuple:
    m = compute_all_metrics(df)
    pw = m["pairwise_accuracy"]
    sp = m["spearman"]
    n_pairs = m.get("n_pairs", 0)
    n_prod = len(df)
    if _SKIP_BCA:
        logger.info(
            f"{label}: pairwise_accuracy = {pw:.4f} (spearman = {sp:.4f}, "
            f"n_pairs={n_pairs}, n_products={n_prod}) [BCa CI skipped]"
        )
        return pw, float("nan"), float("nan"), sp, n_pairs, n_prod
    cis = compute_bca_cis(df, n_bootstrap=10000)
    lo, hi = cis["pairwise_accuracy"]
    sp_lo, sp_hi = cis.get("spearman", (np.nan, np.nan))
    logger.info(
        f"{label}: pairwise_accuracy = {pw:.4f} [{lo:.4f}, {hi:.4f}] "
        f"(spearman = {sp:.4f} [{sp_lo:.4f}, {sp_hi:.4f}], n_pairs={n_pairs}, n_products={n_prod})"
    )
    return pw, lo, hi, sp, n_pairs, n_prod


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pf_path = Path(_PF_OVERRIDE) if _PF_OVERRIDE else None
    pf = load_product_features(pf_path)
    logger.info(f"Loaded v4 product_features ({_VARIANT}): {len(pf)} products"
                + (f" from {pf_path}" if pf_path else ""))

    impute_missing_images_inplace(pf, knn_k=KNN_K)

    logger.info("=== Step A: BT SNCTI bench LOOCV (v4 features) ===")
    bt_path = run_bt_loocv_bench(pf)
    bt_v4 = pd.read_csv(bt_path).dropna(subset=["predicted_score"])
    report_metrics(bt_v4, "BT SNCTI v4")

    logger.info("=== Step B: nested BT+Gemini base features (v4) ===")
    base = compute_nested_base_v4(pf)

    logger.info("=== Step C: NNLS meta-learner ===")
    nnls_oof = run_nnls_meta(base)
    nnls_df = save_oof(base, nnls_oof, V4_NNLS_FILE)

    # Also save rank-avg and equal-mean ensemble variants (BT only,
    # since the canonical results table reports all three for BT+Gemini).
    if _SUPERVISED_MODEL == "bradley_terry":
        rank_oof = run_rank_avg_meta(base)
        save_oof(base, rank_oof, f"nested_bt_gemini_rank_{_VARIANT}.csv")
        mean_oof = run_equal_mean_meta(base)
        save_oof(base, mean_oof, f"nested_bt_gemini_mean_{_VARIANT}.csv")

    logger.info("=== Step D: pairwise accuracy + 95% BCa CI ===")
    regenerated = report_metrics(nnls_df, f"{_MODEL_SHORT.upper()}+Gemini NNLS (regenerated)")

    # Compare against the shipped canonical OOFs.
    logger.info("=== Step E: comparison against canonical OOFs ===")
    for label, fn in [
        ("BT SNCTI (canonical)", CANONICAL_BT_FILE),
        ("BT+Gemini NNLS (canonical)", CANONICAL_NNLS_FILE),
    ]:
        path = OOF_DIR / fn
        if path.exists():
            df = pd.read_csv(path).dropna(subset=["predicted_score"])
            report_metrics(df, label)
        else:
            logger.warning(f"{fn} not found — skipping comparison")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    pw, lo, hi, sp, np_, n_prod = regenerated
    print(f"Regenerated BT + Gemini NNLS pairwise_accuracy = {pw:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"  spearman = {sp:.4f}, n_pairs = {np_}, n_products = {n_prod}")
    print()
    canonical_path = OOF_DIR / CANONICAL_NNLS_FILE
    if canonical_path.exists():
        df_c = pd.read_csv(canonical_path).dropna(subset=["predicted_score"])
        m_c = compute_all_metrics(df_c)
        pw_c = m_c["pairwise_accuracy"]
        if _SKIP_BCA:
            print(f"Canonical pairwise_accuracy = {pw_c:.4f}")
        else:
            c_c = compute_bca_cis(df_c, n_bootstrap=10000)
            lo_c, hi_c = c_c["pairwise_accuracy"]
            print(f"Canonical pairwise_accuracy = {pw_c:.4f} [{lo_c:.4f}, {hi_c:.4f}]")
        print(f"  Δ pairwise_accuracy = {pw - pw_c:+.4f}")
    print()


if __name__ == "__main__":
    main()
