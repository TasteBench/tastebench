"""Regenerate `dist_pred_{cosine,l2}_{subset}.csv` for all N/C/T/I subsets.

Distance baselines (Multi-modal Rank Fusion) score each plant-based
product by its distance to the within-category animal-based centroid
across the chosen feature modalities, normalised to a percentile rank
within category. The canonical paper convention uses
`category_nutrition` (4-dim per-category-subset columns) rather than
the supervised `nutrition` (6-dim full panel).

Reads `data/product_features.pkl`; writes 30 OOF CSVs (15 subsets ×
2 metrics) into `results/oof_predictions/`.

Usage:
    cd food_similarity
    python -m train.regenerate_distance_oofs
"""
from __future__ import annotations

import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERVISED_DIR))

from data.loocv import get_analog_keys, load_product_features  # noqa: E402
from models.distance_predictor import DistancePredictor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

PKL = SUPERVISED_DIR / "data" / "product_features.pkl"
OOF_DIR = SUPERVISED_DIR / "results" / "oof_predictions"

# Canonical letter → distance-predictor feature name. N is mapped to
# `category_nutrition` (4-dim per-category subset), matching the
# unsupervised pipeline's original convention. Distance baselines do
# NOT include the `category_subset` indicator (that's a supervised-only
# feature for letting the model learn per-category coefficients).
FEATURE_MAP = {
    "N": "category_nutrition",
    "C": "compound",
    "T": "text",
    "I": "image",
}

LETTERS = ["N", "C", "T", "I"]
METRICS = [("cosine", "cosine"), ("euclidean", "l2")]


def _all_subsets() -> list[str]:
    out = []
    for r in range(1, 5):
        for combo in combinations(LETTERS, r):
            out.append("".join(combo))
    return out


def main():
    logger.info(f"Loading {PKL}")
    pf = load_product_features(PKL)
    analog_keys = sorted(get_analog_keys(pf))

    subsets = _all_subsets()
    logger.info(f"Regenerating {len(subsets) * 2} distance OOFs ({len(subsets)} subsets × 2 metrics)")

    written = 0
    for subset in subsets:
        feat_types = [FEATURE_MAP[c] for c in subset]
        for metric, tag in METRICS:
            filename = f"dist_pred_{tag}_{subset}.csv"
            path = OOF_DIR / filename
            try:
                model = DistancePredictor(
                    feature_types=feat_types,
                    product_features=pf,
                    distance_metric=metric,
                    missing_feature_strategy="skip",
                )
                model.fit()
                scores = model.get_all_scores()

                rows = []
                for key in analog_keys:
                    p = pf[key]
                    rows.append({
                        "category": p["category"],
                        "product_code": p["product_code"],
                        "true_score": p["mean_similarity"],
                        "predicted_score": scores.get(key, np.nan),
                    })
                df = pd.DataFrame(rows)
                df.to_csv(path, index=False)
                n_valid = df["predicted_score"].notna().sum()
                logger.info(f"  {filename}  {n_valid}/{len(df)} products")
                written += 1
            except Exception as e:
                logger.error(f"  {filename} failed: {e}")

    logger.info(f"Wrote {written}/{len(subsets) * 2} distance OOFs.")


if __name__ == "__main__":
    main()
