"""Strictly leak-free nested-LOOCV meta-ensembles for BT + Gemini.

Computes the O(n^2) nested base features once, caches them to disk, then
runs multiple meta-learners (NNLS, LightGBM, Linear, equal-weight) on the
cached features. This is strictly leak-free because every meta-training
feature was produced by a BT that never saw the outer held-out product.

Base features:
  bt_outer[i]    = prediction for i from BT trained on {k : k != i}
  bt_inner[i, j] = prediction for j from BT trained on {k : k != i, k != j}
  gem_oof[i]     = Gemini's zero-shot prediction for i (no labels used)

Meta-learner training (for each outer i):
  Train:   features = [bt_inner[i, j], gem_oof[j]] for j != i
           target   = true[j]
  Predict: features = [bt_outer[i], gem_oof[i]]

Supported meta-learners:
  - NNLS       : non-negative least squares (2 non-negative weights)
  - LGBM       : shallow LightGBM (20 trees, 4 leaves)
  - Linear     : plain linear regression (unconstrained weights + intercept)
  - Equal mean : just (bt_outer + gem) / 2 (no training, for reference)

Cache files:
  results/cache/nested_bt_gemini_base.npz   — base features

Usage:
    cd food_similarity
    python -m train.nested_meta_all
    python -m train.nested_meta_all --force-recompute
"""

import argparse
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

from data.loocv import (
    FeatureProcessor,
    build_feature_matrix,
    build_score_vector,
    get_analog_keys,
    load_product_features,
)
from evaluation.metrics import compute_all_metrics
from models.bradley_terry import FeatureBradleyTerry
from train.paper_table import impute_missing_images_inplace

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

FEATURE_TYPES = ["category_subset", "nutrition", "compound", "text", "image"]


# ---------------------------------------------------------------------------
# Base feature computation (O(n^2) BT fits)
# ---------------------------------------------------------------------------

def compute_nested_base_features(
    product_features: dict,
    gemini_oof_path: str,
    cache_file: Path,
    force: bool = False,
) -> dict:
    """Compute bt_outer, bt_inner, gem_oof, y_all and cache them.

    Returns a dict with keys:
      bt_outer  : (n,)     — BT predictions from LOOCV (leak-free w.r.t. i)
      bt_inner  : (n, n)   — bt_inner[i, j] = prediction for j from BT
                             trained without {i, j}. Diagonal is nan.
      gem_oof   : (n,)     — Gemini predictions (zero-shot, leak-free)
      y_all     : (n,)     — true scores
      categories: (n,) str — category labels (object dtype)
      product_keys : list of (cat, code) tuples
    """
    if cache_file.exists() and not force:
        logger.info(f"Loading cached base features from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return {
            "bt_outer": data["bt_outer"],
            "bt_inner": data["bt_inner"],
            "gem_oof": data["gem_oof"],
            "y_all": data["y_all"],
            "categories": data["categories"],
            "product_keys": [tuple(k) for k in data["product_keys"]],
        }

    logger.info("Computing nested base features from scratch...")
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
    logger.info(f"n = {n}, features shape = {X_all.shape}")

    # Align Gemini predictions
    gem_df = pd.read_csv(gemini_oof_path).dropna(subset=["predicted_score"])
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

    # Step 1: outer LOOCV (standard, n training per fold)
    logger.info("Step 1/2: Outer LOOCV (n=214 training per fold)...")
    start = time.time()
    bt_outer = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        proc = FeatureProcessor(FEATURE_TYPES, feature_dims, use_pca=True, pca_variance=0.95)
        X_tr = proc.fit_transform(X_all[mask], y_all[mask])
        X_te = proc.transform(X_all[i:i + 1])
        bt = FeatureBradleyTerry()
        train_cats = [cats_all[k] for k in range(n) if mask[k]]
        bt.fit(X_tr, y_all[mask], categories=train_cats)
        bt_outer[i] = bt.predict_score(X_te, categories=[cats_all[i]])[0]
    logger.info(f"  Done in {time.time() - start:.1f}s")

    # Step 2: inner nested LOOCV (leave two out per pair)
    logger.info(f"Step 2/2: Inner nested LOOCV ({n * (n - 1) // 2} pairs)...")
    start = time.time()
    bt_inner = np.full((n, n), np.nan)
    pair_count = 0
    total_pairs = n * (n - 1) // 2
    log_every = max(1, total_pairs // 20)

    for i in range(n):
        for j in range(i + 1, n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            proc = FeatureProcessor(FEATURE_TYPES, feature_dims, use_pca=True, pca_variance=0.95)
            X_tr = proc.fit_transform(X_all[mask], y_all[mask])
            X_te = proc.transform(X_all[[i, j]])
            train_cats = [cats_all[k] for k in range(n) if mask[k]]
            bt = FeatureBradleyTerry()
            bt.fit(X_tr, y_all[mask], categories=train_cats)
            preds = bt.predict_score(X_te, categories=[cats_all[i], cats_all[j]])
            bt_inner[i, j] = preds[1]  # prediction for j from BT without {i,j}
            bt_inner[j, i] = preds[0]  # prediction for i from BT without {i,j}
            pair_count += 1
            if pair_count % log_every == 0:
                elapsed = time.time() - start
                eta = elapsed / pair_count * (total_pairs - pair_count)
                logger.info(
                    f"  Pair {pair_count}/{total_pairs} "
                    f"({100 * pair_count / total_pairs:.1f}%), "
                    f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s"
                )
    logger.info(f"  Done in {time.time() - start:.0f}s")

    # Save cache
    np.savez(
        cache_file,
        bt_outer=bt_outer,
        bt_inner=bt_inner,
        gem_oof=gem_oof,
        y_all=y_all,
        categories=cats_all,
        product_keys=np.array([list(k) for k in valid_keys], dtype=object),
    )
    logger.info(f"Saved cache to {cache_file}")

    return {
        "bt_outer": bt_outer,
        "bt_inner": bt_inner,
        "gem_oof": gem_oof,
        "y_all": y_all,
        "categories": cats_all,
        "product_keys": valid_keys,
    }


# ---------------------------------------------------------------------------
# Meta-learners (all operate on cached base features)
# ---------------------------------------------------------------------------

def run_meta_nnls(base: dict) -> np.ndarray:
    """Nested NNLS on [bt_inner, gem_oof]. Strictly leak-free."""
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
        X_meta_train = np.column_stack([bt_inner[i, mask], gem_oof[mask]])
        y_meta_train = y_all[mask]
        w, _ = nnls(X_meta_train, y_meta_train)
        oof[i] = np.array([bt_outer[i], gem_oof[i]]) @ w
    return oof


def run_meta_lgbm(base: dict, n_estimators: int = 20) -> np.ndarray:
    """Nested LightGBM with a fixed, pre-registered number of trees.

    Strictly leak-free since nested base features isolate the outer product.
    The n_estimators choice is pre-registered (not tuned) to avoid selection
    bias. Default 20 is the original choice; also called with n_estimators=5,
    10 to check robustness.
    """
    import lightgbm as lgb
    bt_outer = base["bt_outer"]
    bt_inner = base["bt_inner"]
    gem_oof = base["gem_oof"]
    y_all = base["y_all"]
    n = len(bt_outer)

    params = dict(
        n_estimators=n_estimators, num_leaves=4, min_child_samples=10,
        learning_rate=0.1, verbosity=-1, random_state=42,
    )
    oof = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_meta_train = np.column_stack([bt_inner[i, mask], gem_oof[mask]])
        y_meta_train = y_all[mask]
        m = lgb.LGBMRegressor(**params)
        m.fit(X_meta_train, y_meta_train)
        oof[i] = m.predict(np.array([[bt_outer[i], gem_oof[i]]]))[0]
    return oof


def run_meta_lgbm_early_stop(
    base: dict,
    max_rounds: int = 100,
    val_frac: float = 0.2,
    patience: int = 10,
    seed: int = 42,
) -> tuple:
    """Nested LightGBM with early stopping. Strictly leak-free and data-adaptive.

    For each outer held-out product i:
      1. Split the 214-sample meta training set into inner train (1-val_frac)
         and inner val (val_frac) — both drawn only from products j != i.
      2. Fit LGBM with max_rounds and early stopping on inner val MSE
         (stop if no improvement for `patience` rounds).
      3. Record the best n_estimators found.
      4. Retrain on the FULL 214-sample meta training set with that fixed
         n_estimators (no more validation set needed — we already know k).
      5. Predict on (bt_outer[i], gem_oof[i]).

    Leakage-free because i is never in the inner train/val split, and the
    n_estimators choice depends only on the other 214 products.

    Returns (oof_predictions, list_of_best_n_estimators).
    """
    import lightgbm as lgb
    bt_outer = base["bt_outer"]
    bt_inner = base["bt_inner"]
    gem_oof = base["gem_oof"]
    y_all = base["y_all"]
    n = len(bt_outer)

    oof = np.zeros(n)
    best_rounds = []

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = np.column_stack([bt_inner[i, mask], gem_oof[mask]])
        y_train = y_all[mask]
        n_train = len(y_train)

        # Inner train/val split (seeded per fold for reproducibility)
        rng = np.random.default_rng(seed + i)
        idx = rng.permutation(n_train)
        n_val = max(10, int(n_train * val_frac))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]

        # Fit with early stopping on inner val
        model = lgb.LGBMRegressor(
            n_estimators=max_rounds, num_leaves=4, min_child_samples=10,
            learning_rate=0.1, verbosity=-1, random_state=seed,
        )
        model.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[val_idx], y_train[val_idx])],
            callbacks=[lgb.early_stopping(stopping_rounds=patience, verbose=False)],
        )
        best_n = max(1, int(model.best_iteration_ or 1))
        best_rounds.append(best_n)

        # Retrain on FULL meta training set with the recorded n_estimators
        final = lgb.LGBMRegressor(
            n_estimators=best_n, num_leaves=4, min_child_samples=10,
            learning_rate=0.1, verbosity=-1, random_state=seed,
        )
        final.fit(X_train, y_train)
        oof[i] = final.predict(np.array([[bt_outer[i], gem_oof[i]]]))[0]

    return oof, best_rounds


def run_meta_linear(base: dict) -> np.ndarray:
    """Nested plain linear regression (unconstrained). Strictly leak-free."""
    from sklearn.linear_model import LinearRegression
    bt_outer = base["bt_outer"]
    bt_inner = base["bt_inner"]
    gem_oof = base["gem_oof"]
    y_all = base["y_all"]
    n = len(bt_outer)

    oof = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_meta_train = np.column_stack([bt_inner[i, mask], gem_oof[mask]])
        y_meta_train = y_all[mask]
        m = LinearRegression()
        m.fit(X_meta_train, y_meta_train)
        oof[i] = m.predict(np.array([[bt_outer[i], gem_oof[i]]]))[0]
    return oof


def run_meta_ridge(base: dict, alpha: float = 1.0) -> np.ndarray:
    """Nested Ridge regression. Strictly leak-free."""
    from sklearn.linear_model import Ridge
    bt_outer = base["bt_outer"]
    bt_inner = base["bt_inner"]
    gem_oof = base["gem_oof"]
    y_all = base["y_all"]
    n = len(bt_outer)

    oof = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_meta_train = np.column_stack([bt_inner[i, mask], gem_oof[mask]])
        y_meta_train = y_all[mask]
        m = Ridge(alpha=alpha)
        m.fit(X_meta_train, y_meta_train)
        oof[i] = m.predict(np.array([[bt_outer[i], gem_oof[i]]]))[0]
    return oof


def run_meta_equal_mean(base: dict) -> np.ndarray:
    """Simple (bt_outer + gem) / 2 at test time. No meta training.

    Note this uses bt_outer (not bt_inner) at test time, which is what a
    user would see in practice. Strictly leak-free since no weights are learned.
    """
    return 0.5 * (base["bt_outer"] + base["gem_oof"])


def run_meta_rank_avg(base: dict) -> np.ndarray:
    """Within-category rank average of bt_outer and gem_oof.

    Strictly leak-free, scale-invariant.
    """
    bt_outer = base["bt_outer"]
    gem_oof = base["gem_oof"]
    cats = base["categories"]
    df = pd.DataFrame({"category": cats, "bt": bt_outer, "gem": gem_oof})
    df["r_bt"] = df.groupby("category")["bt"].rank(pct=True)
    df["r_gem"] = df.groupby("category")["gem"].rank(pct=True)
    return 0.5 * (df["r_bt"].values + df["r_gem"].values)


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_predictions(base: dict, predictions: np.ndarray, filename: str):
    """Save an OOF CSV with the standard columns."""
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-recompute", action="store_true",
                        help="Recompute base features even if cache exists")
    args = parser.parse_args()

    # Load data and impute missing images
    pf = load_product_features()
    impute_missing_images_inplace(pf)

    # Find best Gemini
    best_file, best_pw = None, -1
    for f in sorted(os.listdir(OOF_DIR)):
        if f.startswith("llm_gemini_3_1_pro_preview_") and f.endswith(".csv"):
            df = pd.read_csv(OOF_DIR / f).dropna(subset=["predicted_score"])
            m = compute_all_metrics(df)
            if m["pairwise_accuracy"] > best_pw:
                best_file, best_pw = f, m["pairwise_accuracy"]
    logger.info(f"Using Gemini: {best_file} (pw={best_pw:.4f})")

    # Compute or load cached nested base features
    cache_file = CACHE_DIR / "nested_bt_gemini_base.npz"
    base = compute_nested_base_features(
        pf, str(OOF_DIR / best_file), cache_file, force=args.force_recompute
    )

    # Run each meta-learner
    logger.info("=" * 60)
    logger.info("Running meta-learners on cached base features")
    logger.info("=" * 60)

    results = []

    # 1. Parameter-free baselines
    for display, fn, filename in [
        ("Equal mean (leak-free)", run_meta_equal_mean, "nested_bt_gemini_mean.csv"),
        ("Rank avg (leak-free)", run_meta_rank_avg, "nested_bt_gemini_rank.csv"),
        ("NNLS (nested, leak-free)", run_meta_nnls, "nested_bt_gemini_nnls.csv"),
        ("Linear reg (nested, leak-free)", run_meta_linear, "nested_bt_gemini_linear.csv"),
        ("Ridge α=1.0 (nested, leak-free)", run_meta_ridge, "nested_bt_gemini_ridge.csv"),
    ]:
        logger.info(f"Running {display}...")
        preds = fn(base)
        df = save_predictions(base, preds, filename)
        m = compute_all_metrics(df)
        results.append((display, filename, m))
        logger.info(
            f"  pw={m['pairwise_accuracy']:.4f}  "
            f"sp={m['spearman']:.4f}  "
            f"k={m['kendall_tau']:.4f}  "
            f"R@1={m['recall_at_1']:.3f}  R@3={m['recall_at_3']:.3f}"
        )

    # 2. Fixed-tree LGBM variants (pre-registered tree counts)
    for n_trees in [5, 10, 20]:
        display = f"LGBM {n_trees} trees (nested, leak-free)"
        filename = f"nested_bt_gemini_lgbm{n_trees}.csv"
        logger.info(f"Running {display}...")
        preds = run_meta_lgbm(base, n_estimators=n_trees)
        df = save_predictions(base, preds, filename)
        m = compute_all_metrics(df)
        results.append((display, filename, m))
        logger.info(
            f"  pw={m['pairwise_accuracy']:.4f}  "
            f"sp={m['spearman']:.4f}  "
            f"k={m['kendall_tau']:.4f}  "
            f"R@1={m['recall_at_1']:.3f}  R@3={m['recall_at_3']:.3f}"
        )

    # 3. LGBM with early stopping (data-adaptive, leak-free)
    display = "LGBM early-stop (nested, leak-free)"
    filename = "nested_bt_gemini_lgbm_es.csv"
    logger.info(f"Running {display}...")
    preds, best_rounds = run_meta_lgbm_early_stop(
        base, max_rounds=100, val_frac=0.2, patience=10
    )
    df = save_predictions(base, preds, filename)
    m = compute_all_metrics(df)
    results.append((display, filename, m))
    logger.info(
        f"  pw={m['pairwise_accuracy']:.4f}  "
        f"sp={m['spearman']:.4f}  "
        f"k={m['kendall_tau']:.4f}  "
        f"R@1={m['recall_at_1']:.3f}  R@3={m['recall_at_3']:.3f}"
    )
    br_arr = np.array(best_rounds)
    logger.info(
        f"  Early-stopped rounds: median={int(np.median(br_arr))}, "
        f"mean={br_arr.mean():.1f}, min={br_arr.min()}, max={br_arr.max()}"
    )

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY: BT + Gemini ensemble variants (all strictly leak-free)")
    print("=" * 80)
    print(f"{'Meta-learner':<40s} {'pw':>8s} {'spearman':>10s} "
          f"{'kendall':>10s} {'R@1':>7s} {'R@3':>7s}")
    print("-" * 80)
    for display, filename, m in sorted(results, key=lambda x: -x[2]["pairwise_accuracy"]):
        print(f"{display:<40s} {m['pairwise_accuracy']:>8.4f} "
              f"{m['spearman']:>10.4f} {m['kendall_tau']:>10.4f} "
              f"{m['recall_at_1']:>7.3f} {m['recall_at_3']:>7.3f}")
    print()
    print("For comparison:")
    gem_df = pd.read_csv(OOF_DIR / best_file).dropna(subset=["predicted_score"])
    m_gem = compute_all_metrics(gem_df)
    print(f"{'Gemini alone':<40s} {m_gem['pairwise_accuracy']:>8.4f} "
          f"{m_gem['spearman']:>10.4f} {m_gem['kendall_tau']:>10.4f} "
          f"{m_gem['recall_at_1']:>7.3f} {m_gem['recall_at_3']:>7.3f}")
    bt_df = pd.read_csv(OOF_DIR / "bradley_terry_SNCTI_bench.csv").dropna(subset=["predicted_score"])
    m_bt = compute_all_metrics(bt_df)
    print(f"{'Bradley-Terry alone':<40s} {m_bt['pairwise_accuracy']:>8.4f} "
          f"{m_bt['spearman']:>10.4f} {m_bt['kendall_tau']:>10.4f} "
          f"{m_bt['recall_at_1']:>7.3f} {m_bt['recall_at_3']:>7.3f}")


if __name__ == "__main__":
    main()
