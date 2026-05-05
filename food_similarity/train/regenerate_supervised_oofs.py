"""Regenerate every supervised + distance OOF on `data/product_features.pkl`.

For each of the 5 supervised models (BT, hierarchical BT, ridge,
LightGBM, kernel RankSVM): run LOOCV across all 15 non-empty subsets
of {N, C, T, I} → `{model}_S{subset}.csv`, plus the full-SNCTI bench
config → `{model}_SNCTI_bench.csv`.

For each distance predictor (cosine, L2): run LOOCV across all 15
non-empty subsets → `dist_pred_{metric}_{subset}.csv`.

These OOFs feed the ablation table, the per-category tables, the
main results table, and the per-model NNLS ensemble. Per-model
nested NNLS ensembles themselves live in `regenerate_per_model_nnls.py`
(they take ~30 min per model and so are kept in a separate driver).

Usage:
    cd food_similarity
    python -m train.regenerate_supervised_oofs
"""
from __future__ import annotations

import logging
import sys
import time
from itertools import combinations
from pathlib import Path

import pandas as pd

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERVISED_DIR))

from data.loocv import load_product_features  # noqa: E402
from evaluation.metrics import compute_all_metrics  # noqa: E402
from train.paper_table import (  # noqa: E402
    impute_missing_images_inplace,
    KNN_K,
    generate_distance_predictor,
)
from train.run_loocv import run_single, MODEL_REGISTRY  # noqa: E402
import train.run_loocv as loocv_module  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PKL = SUPERVISED_DIR / "data" / "product_features.pkl"
OOF_DIR = SUPERVISED_DIR / "results" / "oof_predictions"

# Paper subset grammar. Map subset code (e.g., "NCTI") to feature_code
# string the loocv runner expects. The loocv runner uses the same
# convention with an 'S' prefix to indicate category_subset is included.
FEATURE_LETTERS = ["N", "C", "T", "I"]


def _all_subset_codes() -> list[str]:
    """All 15 non-empty subsets of {N, C, T, I}."""
    out = []
    for r in range(1, 5):
        for combo in combinations(FEATURE_LETTERS, r):
            out.append("S" + "".join(combo))
    return out


def main():
    t_start = time.time()
    logger.info(f"Loading product features from {PKL}")
    pf = load_product_features(PKL)
    impute_missing_images_inplace(pf, knn_k=KNN_K)
    logger.info(f"Loaded {len(pf)} products")

    saved = (
        loocv_module._SKIP_BOOTSTRAP,
        loocv_module._KNN_IMPUTE,
        loocv_module._PCA_VARIANCE,
    )
    # Match paper_table.py canonical settings.
    loocv_module._SKIP_BOOTSTRAP = True
    loocv_module._KNN_IMPUTE = 0  # already imputed above
    loocv_module._PCA_VARIANCE = 0.95

    subset_codes = _all_subset_codes()
    results = []

    try:
        # ---- Per-model × per-subset ablation OOFs ----
        for model_name in MODEL_REGISTRY:
            for code in subset_codes:
                tag = f"{model_name}_{code}"
                t0 = time.time()
                try:
                    run_single(model_name, code, pf, suffix="")
                    csv = OOF_DIR / f"{model_name}_{code}.csv"
                    df = pd.read_csv(csv).dropna(subset=["predicted_score"])
                    m = compute_all_metrics(df)
                    elapsed = time.time() - t0
                    logger.info(
                        f"  {tag:<35s} pw={m['pairwise_accuracy']:.4f} "
                        f"sp={m['spearman']:.4f} ({elapsed:.0f}s)"
                    )
                    results.append((tag, m["pairwise_accuracy"], elapsed))
                except Exception as e:
                    logger.error(f"  {tag} failed: {e}")

        # ---- Per-model × full SNCTI _bench ----
        for model_name in MODEL_REGISTRY:
            tag = f"{model_name}_SNCTI_bench"
            t0 = time.time()
            try:
                run_single(model_name, "SNCTI", pf, suffix="_bench")
                csv = OOF_DIR / f"{tag}.csv"
                df = pd.read_csv(csv).dropna(subset=["predicted_score"])
                m = compute_all_metrics(df)
                elapsed = time.time() - t0
                logger.info(
                    f"  {tag:<35s} pw={m['pairwise_accuracy']:.4f} "
                    f"sp={m['spearman']:.4f} ({elapsed:.0f}s)"
                )
                results.append((tag, m["pairwise_accuracy"], elapsed))
            except Exception as e:
                logger.error(f"  {tag} failed: {e}")

        # ---- Distance predictors (cosine, l2) × all subsets ----
        # generate_distance_predictor() iterates the paper's subset
        # convention internally. It writes dist_pred_{metric}_{subset}.csv.
        try:
            t0 = time.time()
            generate_distance_predictor(pf)
            elapsed = time.time() - t0
            logger.info(f"  distance predictors completed ({elapsed:.0f}s)")
        except Exception as e:
            logger.error(f"  distance predictors failed: {e}")

    finally:
        loocv_module._SKIP_BOOTSTRAP, loocv_module._KNN_IMPUTE, loocv_module._PCA_VARIANCE = saved

    total = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Phase G complete in {total/60:.1f} min")
    logger.info("=" * 70)
    logger.info(f"{'tag':<40s} {'pw':>8s} {'time':>8s}")
    for tag, pw, el in sorted(results, key=lambda r: -r[1])[:15]:
        logger.info(f"  {tag:<38s} {pw:>8.4f} {el:>7.0f}s")


if __name__ == "__main__":
    main()
