"""Post-grid selection, FART-test evaluation, embedding-cache build, NECTAR transfer.

Pipeline
--------
1. Read grid_summary.csv from the results directory.
2. Pick the single run with the highest val_macro_f1. This is the ONLY
   selection decision and it is made on the FART val split; FART test and
   NECTAR LOO are not peeked at during selection.
3. Evaluate the val-best checkpoint on FART test.
4. Rebuild the taste_gnn compound-embedding cache at
   shared/data/caches/taste_gnn_best_compound_embeddings.pkl
   from the val-best checkpoint's penultimate-layer outputs.
5. Run the supervised encoder-transfer pipeline (MMRF + supervised LOOCV +
   nested BT+Gemini) so the *_tastegnn.csv OOFs feed Tables 2 / 3.

Usage
-----
    python -m molecular.src.train.select_best_and_evaluate \\
        --results_dir molecular/results/grid
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

TOP_DIR = Path(__file__).resolve().parents[3]
SHARED_CACHES = TOP_DIR / "shared/data/caches"
BEST_CACHE_PATH = SHARED_CACHES / "taste_gnn_best_compound_embeddings.pkl"
FART_SOURCE_CACHE = SHARED_CACHES / "fart_compound_embeddings.pkl"


def _run(cmd: list[str]) -> None:
    logger.info("$ %s", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, cwd=TOP_DIR)


def pick_best(results_dir: Path) -> tuple[Path, dict]:
    """Return (run_dir, config) for the highest-val_macro_f1 run."""
    summary = results_dir / "grid_summary.csv"
    if not summary.exists():
        raise SystemExit(f"Missing {summary} -- run grid_search.py first.")
    df = pd.read_csv(summary).sort_values("val_macro_f1", ascending=False)
    if df.empty:
        raise SystemExit("grid_summary.csv is empty.")
    best = df.iloc[0].to_dict()
    run_dir = results_dir / best["run_name"]
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"Missing {cfg_path}")
    cfg = yaml.safe_load(cfg_path.read_text())
    logger.info("Selected best run: %s (val_macro_f1=%.4f, "
                "class_weighting=%s, depth=%s, dropout=%s)",
                best["run_name"], best["val_macro_f1"],
                best["class_weighting"], best["depth"], best["dropout"])
    return run_dir, cfg


def evaluate_fart_test(ckpt_path: Path, out_dir: Path) -> dict:
    _run([sys.executable, "-m", "molecular.src.eval.evaluate",
          "--model_type", "dmpnn_ckpt",
          "--ckpt", str(ckpt_path),
          "--test_csv", "molecular/data/splits/fart_test.csv",
          "--output_dir", str(out_dir)])
    return json.loads((out_dir / "metrics.json").read_text())


def build_compound_cache(ckpt_path: Path) -> None:
    """Regenerate shared/data/caches/taste_gnn_best_compound_embeddings.pkl."""
    if BEST_CACHE_PATH.exists():
        BEST_CACHE_PATH.unlink()
    _run([sys.executable, "-m", "molecular.src.embed.generate_cache",
          "--checkpoint",   str(ckpt_path),
          "--source_cache", str(FART_SOURCE_CACHE),
          "--output_pkl",   str(BEST_CACHE_PATH)])


def rerun_nectar_encoder_transfer() -> None:
    """Run the supervised side: MMRF + 5 supervised models + nested BT+Gemini
    with C = the cache currently at BEST_CACHE_PATH. Writes *_tastegnn.csv
    OOFs that feed Tables 2 / 3."""
    _run([sys.executable, "-m",
          "food_similarity.train.run_taste_gnn_encoder_transfer"])


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True, type=Path)
    ap.add_argument("--skip_fart_test", action="store_true")
    ap.add_argument("--skip_nectar", action="store_true")
    args = ap.parse_args()

    best_run_dir, _ = pick_best(args.results_dir)
    best_ckpt = best_run_dir / "ckpt.pt"
    if not best_ckpt.exists():
        raise SystemExit(f"Missing best checkpoint: {best_ckpt}")

    if not args.skip_fart_test:
        m = evaluate_fart_test(best_ckpt, best_run_dir / "fart_test_eval")
        logger.info("FART test: acc=%.4f macro_f1=%.4f", m["accuracy"], m["macro_f1"])

    if args.skip_nectar:
        return 0

    build_compound_cache(best_ckpt)
    rerun_nectar_encoder_transfer()
    logger.info("Done. Re-render with:")
    logger.info("  python molecular/scripts/render_table_molecular_prediction.py")
    logger.info("  python food_similarity/scripts/render_table_gnn_per_model.py")
    logger.info("  python food_similarity/scripts/render_table_gnn_grid.py")
    logger.info("  python food_similarity/scripts/compile_check_tables.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
