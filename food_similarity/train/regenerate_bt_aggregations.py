"""Compute the BT+Gemini rank-avg and equal-mean ensemble OOFs.

The expensive nested LOOCV is run once by `compute_per_model_nnls.py`
and cached at `results/cache/nested_bt_gemini_base_v4.npz`. This
script re-uses that cache to compute the cheap rank-avg and equal-
mean aggregations without re-doing the inner loop. Output:

    nested_bt_gemini_rank.csv   (within-category rank-percentile mean)
    nested_bt_gemini_mean.csv   (arithmetic mean of bt_outer + gemini)

Both feed the BT+Gemini ensemble rows in `table_results.tex`.

Usage:
    cd food_similarity
    python -m train.regenerate_bt_aggregations
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERVISED_DIR))

from evaluation.metrics import compute_all_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OOF_DIR = SUPERVISED_DIR / "results" / "oof_predictions"
CACHE = SUPERVISED_DIR / "results" / "cache" / "nested_bt_gemini_base_v4.npz"


def main():
    if not CACHE.exists():
        logger.error(f"Cache not found: {CACHE}")
        logger.error("Run `python -m train.compute_per_model_nnls` first to populate it.")
        return 1

    d = np.load(CACHE, allow_pickle=True)
    bt_outer = d["bt_outer"]
    gem_oof = d["gem_oof"]
    y_all = d["y_all"]
    cats = d["categories"]
    keys = d["product_keys"]

    # Equal mean
    mean_oof = 0.5 * (bt_outer + gem_oof)
    # Rank avg (within category)
    df = pd.DataFrame({"cat": cats, "bt": bt_outer, "gem": gem_oof})
    df["r_bt"] = df.groupby("cat")["bt"].rank(pct=True)
    df["r_gem"] = df.groupby("cat")["gem"].rank(pct=True)
    rank_oof = 0.5 * (df["r_bt"].values + df["r_gem"].values)

    for name, oof in [("nested_bt_gemini_mean_v4.csv", mean_oof),
                      ("nested_bt_gemini_rank_v4.csv", rank_oof)]:
        rows = []
        for i, k in enumerate(keys):
            cat, code = k
            rows.append({
                "category": cat,
                "product_code": int(code),
                "true_score": y_all[i],
                "predicted_score": float(oof[i]),
            })
        out = pd.DataFrame(rows)
        out.to_csv(OOF_DIR / name, index=False)
        m = compute_all_metrics(out)
        logger.info(f"  {name}: pw={m['pairwise_accuracy']:.4f} sp={m['spearman']:.4f}")
        logger.info(f"    saved {OOF_DIR / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
