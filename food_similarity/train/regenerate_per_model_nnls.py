"""Run `compute_per_model_nnls.py` for each of the 5 supervised models.

Produces `nested_{tag}_gemini_nnls.csv` for tag in {bt, hbt, ridge,
lgbm, ksvm}. These feed `table_per_model_nnls.tex` and the canonical
NNLS row in `table_results.tex`. The BT case is run by default; the
other 4 are spawned sequentially via subprocess with SUPERVISED_MODEL
set per call.

The nested inner LOOCV is the long pole (~30 min per model). Running
sequentially keeps n_jobs at the default per process; running 4 in
parallel would cause cache contention.

Usage:
    cd food_similarity
    python -m train.regenerate_per_model_nnls
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

SUPERVISED_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# bradley_terry already done in Phase F; rerun if SKIP_BT=0 is set
MODELS_DEFAULT = ["hierarchical_bt", "ridge", "lightgbm_reg", "kernel_ranksvm"]

OOF_DIR = SUPERVISED_DIR / "results" / "oof_predictions"

_MODEL_SHORT = {
    "bradley_terry":   "bt",
    "hierarchical_bt": "hbt",
    "ridge":           "ridge",
    "lightgbm_reg":    "lgbm",
    "kernel_ranksvm":  "ksvm",
}


def main():
    skip_bt = os.environ.get("SKIP_BT", "1") == "1"
    models = MODELS_DEFAULT.copy()
    if not skip_bt:
        models.insert(0, "bradley_terry")

    t_start = time.time()
    results = []

    for model in models:
        short = _MODEL_SHORT[model]
        nnls_csv = OOF_DIR / f"nested_{short}_gemini_nnls_v4.csv"
        if nnls_csv.exists() and os.environ.get("FORCE", "0") != "1":
            logger.info(f"Skipping {model}: {nnls_csv.name} already exists "
                        f"(set FORCE=1 to override)")
            continue
        logger.info("=" * 70)
        logger.info(f"Running compute_per_model_nnls with SUPERVISED_MODEL={model}")
        logger.info("=" * 70)
        env = os.environ.copy()
        env["SUPERVISED_MODEL"] = model

        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "train.compute_per_model_nnls"],
            cwd=SUPERVISED_DIR,
            env=env,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            logger.error(f"  {model} failed (returncode={proc.returncode})")
            continue

        results.append((model, elapsed))
        logger.info(f"  {model} completed in {elapsed/60:.1f} min")

    total = time.time() - t_start
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"Phase H complete in {total/60:.1f} min")
    logger.info("=" * 70)
    for model, el in results:
        logger.info(f"  {model:<25s} {el/60:>6.1f} min")


if __name__ == "__main__":
    main()
