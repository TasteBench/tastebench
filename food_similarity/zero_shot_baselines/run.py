"""Experiment runner for the pairwise ranking challenge.

Usage:
    python run.py --config configs/cosine_dist/N.yaml
    python run.py --config configs/llm/qwen3_5_397b_a17b/ingredients_nutrition.yaml
    python run.py --config configs/llm/qwen3_5_397b_a17b/ingredients_nutrition.yaml --resume
    python run.py --all  # runs all distance configs (excludes LLM)
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Add shared/ to path for shared packages (compound_mapping)
_shared_root = str(Path(__file__).resolve().parent.parent.parent / "shared")
if _shared_root not in sys.path:
    sys.path.insert(0, _shared_root)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import random

import numpy as np
import pandas as pd
import yaml

from lib.data import load_labels, load_pairs, load_products
from lib.features import get_feature
from lib.models import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_llm_paths(results_dir: Path, experiment_name: str):
    """Resolve submission, config, and log paths for LLM experiments."""
    exp_parent = str(Path(experiment_name).parent)
    exp_stem = Path(experiment_name).stem
    # e.g. "llm/qwen3_5_397b_a17b" → model_name = "qwen3_5_397b_a17b"
    model_name = Path(exp_parent).name
    filename = f"{model_name}_{exp_stem}"
    return (
        results_dir / exp_parent / "submissions" / f"{filename}.csv",
        results_dir / exp_parent / "configs" / f"{filename}.yaml",
        results_dir / exp_parent / "logs" / f"{filename}.csv",
    )


def run_experiment(config_path: str, name: str = None, resume: bool = False,
                   nectar_only: bool = False):
    """Run a single experiment from a config file."""
    config_path = Path(config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiment_name = name or config.get(
        "experiment_name", config_path.stem
    )
    logger.info(f"Experiment: {experiment_name}")

    # Resolve paths relative to this script's directory
    baselines_dir = Path(__file__).parent
    data_dir = baselines_dir / config.get("data", {}).get("data_dir", "data/competition")
    labels_path = baselines_dir / config.get("data", {}).get(
        "labels_path", "data/product_labels_manually_cleaned.csv"
    )

    # Load data
    logger.info("Loading data...")
    products_df = load_products(data_dir)
    labels_df = load_labels(labels_path)
    pairs_df = load_pairs(data_dir)
    logger.info(
        f"Loaded {len(products_df)} products, {len(pairs_df)} pairs, "
        f"{len(labels_df)} labels"
    )

    # Filter to nectar-vs-nectar pairs if requested; use deterministic
    # fallback for non-nectar pairs so the submission is still complete.
    fallback_results = []
    fallback_log = []
    if nectar_only:
        from lib.models.llm_predictor import _deterministic_fallback

        code_map_path = data_dir / "product_code_map.csv"
        if not code_map_path.exists():
            raise FileNotFoundError(
                f"product_code_map.csv not found at {code_map_path}. "
                "Required for --nectar-only filtering."
            )
        code_map = pd.read_csv(code_map_path)
        nectar_codes = set(
            code_map[code_map["Source"] == "nectar"]["New_Product_Code"].astype(int)
        )
        seed_val = config.get("random_seed", 42)

        is_nectar = (
            pairs_df["product_code_1"].astype(int).isin(nectar_codes)
            & pairs_df["product_code_2"].astype(int).isin(nectar_codes)
        )
        other_pairs = pairs_df[~is_nectar]
        pairs_df = pairs_df[is_nectar].copy()

        for _, row in other_pairs.iterrows():
            tid = int(row["test_id"])
            c1, c2 = int(row["product_code_1"]), int(row["product_code_2"])
            fb = _deterministic_fallback(c1, c2, seed_val)
            winner = c2 if fb == "2" else c1
            fallback_results.append({"test_id": tid, "higher_rated_product": winner})
            fallback_log.append({
                "test_id": tid, "category": row.get("product_category", ""),
                "product_code_1": c1, "product_code_2": c2,
                "prediction": fb, "raw_response": "", "reasoning": "",
                "status": "NON_NECTAR", "swapped": False,
                "elapsed_seconds": 0.0, "prompt_tokens": 0,
                "completion_tokens": 0,
            })

        logger.info(
            f"Nectar-only: {len(pairs_df)} nectar pairs to run, "
            f"{len(other_pairs)} non-nectar pairs (deterministic fallback)"
        )

    # Set global random seeds for reproducibility
    seed = config.get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # Initialize and fit model
    model_config = config.get("model", {})
    model_config["missing_feature_strategy"] = config.get(
        "missing_feature_strategy", "skip"
    )
    model_config["random_seed"] = config.get("random_seed", 42)
    model_type = model_config.get("type", "distance_predictor")

    # Initialize enabled features (LLM predictor reads raw data directly,
    # so it does not use the feature extraction pipeline)
    features = {}
    if model_type != "llm_predictor":
        features_config = config.get("features", {})
        for feat_name, feat_cfg in features_config.items():
            if not feat_cfg.get("enabled", True):
                continue
            logger.info(f"Initializing feature: {feat_name}")
            features[feat_name] = get_feature(feat_name, feat_cfg, products_df, labels_df)
        if not features:
            raise ValueError("No features enabled in config.")
    logger.info(f"Initializing model: {model_type}")
    model = get_model(model_type, model_config)

    logger.info("Fitting model...")
    model.fit(products_df, labels_df, features)

    # Resume: load existing log, filter to remaining pairs, merge after
    results_dir = baselines_dir / "results"
    is_llm = hasattr(model, "response_log")
    prev_log_df = None
    prev_submission_df = None

    if resume and is_llm:
        _, _, log_path = _resolve_llm_paths(results_dir, experiment_name)
        if log_path.exists():
            prev_log_df = pd.read_csv(log_path)
            # Keep successful and non-nectar pairs; only retry real failures
            keep_statuses = {"OK", "NON_NECTAR", "NO_DATA"}
            ok_log = prev_log_df[prev_log_df["status"].isin(keep_statuses)]
            failed_count = len(prev_log_df) - len(ok_log)
            ok_ids = set(ok_log["test_id"].astype(int))

            # Reconstruct submission from successful pairs only
            prev_results = []
            for _, row in ok_log.iterrows():
                tid = int(row["test_id"])
                pred = str(row["prediction"])
                c1 = int(row["product_code_1"])
                c2 = int(row["product_code_2"])
                winner = c2 if pred == "2" else c1
                prev_results.append({"test_id": tid, "higher_rated_product": winner})
            prev_submission_df = pd.DataFrame(prev_results)
            prev_log_df = ok_log

            remaining = pairs_df[~pairs_df["test_id"].astype(int).isin(ok_ids)]
            logger.info(
                f"Resuming: {len(ok_ids)} OK, {failed_count} failed (will retry), "
                f"{len(remaining)} pairs to run"
            )
            if remaining.empty:
                logger.info("All pairs already completed successfully.")
                return
            pairs_df = remaining

    # Resolve output paths
    if is_llm:
        submission_path, config_copy_path, log_path = _resolve_llm_paths(
            results_dir, experiment_name
        )
    else:
        submission_path = results_dir / f"{experiment_name}.csv"
        config_copy_path = results_dir / f"{experiment_name}.yaml"
        log_path = None

    # Periodic checkpoint callback for LLM runs
    def _save_checkpoint(results_list, log_list):
        """Flush current results to disk so --resume can recover from crashes."""
        # Merge with previous results if resuming
        sub = pd.DataFrame(results_list)
        log = pd.DataFrame(log_list)
        if prev_submission_df is not None:
            sub = pd.concat([prev_submission_df, sub], ignore_index=True)
        if prev_log_df is not None:
            log = pd.concat([prev_log_df, log], ignore_index=True)
        sub = sub.sort_values("test_id").reset_index(drop=True)
        log = log.sort_values("test_id").reset_index(drop=True)

        submission_path.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(submission_path, index=False)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(log_path, index=False)

    # Generate predictions
    logger.info("Generating predictions...")
    if is_llm:
        submission_df = model.predict_pairs(pairs_df, on_checkpoint=_save_checkpoint)
    else:
        submission_df = model.predict_pairs(pairs_df)

    # Merge with non-nectar fallback results
    if fallback_results:
        submission_df = pd.concat(
            [submission_df, pd.DataFrame(fallback_results)], ignore_index=True,
        )
        if hasattr(model, "response_log"):
            model.response_log = model.response_log + fallback_log

    # Merge with previous results if resuming
    if prev_submission_df is not None:
        submission_df = pd.concat([prev_submission_df, submission_df], ignore_index=True)
    if prev_log_df is not None and hasattr(model, "response_log"):
        new_log_df = pd.DataFrame(model.response_log)
        merged_log = pd.concat([prev_log_df, new_log_df], ignore_index=True)
        model.response_log = merged_log.to_dict("records")

    submission_df = submission_df.sort_values("test_id").reset_index(drop=True)
    if hasattr(model, "response_log"):
        model.response_log = sorted(model.response_log, key=lambda e: e["test_id"])

    # Save final results
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)

    config_copy_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, config_copy_path)

    if is_llm and model.response_log:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(model.response_log).to_csv(log_path, index=False)
        logger.info(f"Response log saved: {log_path}")

    logger.info(f"Submission saved: {submission_path}")
    logger.info(f"Config saved: {config_copy_path}")

    _print_diagnostics(experiment_name, submission_df, features, model,
                       model_config, labels_df, submission_path)


def _print_diagnostics(experiment_name, submission_df, features, model,
                       model_config, labels_df, submission_path):
    """Print score distribution and per-category breakdown to stdout."""
    model_type = model_config.get("type", "distance_predictor")

    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}")
    print(f"Predictions: {len(submission_df)}")
    print(f"Model: {model_type}")

    if model_type == "llm_predictor":
        print(f"LLM: {model_config.get('model_name', 'unknown')}")
        print(f"Prompt features: {model_config.get('prompt_features', [])}")
        if hasattr(model, "response_log") and model.response_log:
            statuses = [e["status"] for e in model.response_log]
            ok = statuses.count("OK")
            total = len(statuses)
            print(f"\nAPI results: {ok}/{total} OK "
                  f"({total - ok} failures)")
            elapsed = [e["elapsed_seconds"] for e in model.response_log if e["elapsed_seconds"] > 0]
            if elapsed:
                print(f"Latency: min={min(elapsed):.1f}s  max={max(elapsed):.1f}s  "
                      f"mean={np.mean(elapsed):.1f}s")
            total_prompt = sum(e["prompt_tokens"] for e in model.response_log)
            total_completion = sum(e["completion_tokens"] for e in model.response_log)
            print(f"Tokens: {total_prompt:,} prompt, {total_completion:,} completion")
    else:
        print(f"Features: {list(features.keys())}")
        print(f"Distance metric: {model_config.get('distance_metric', 'cosine')}")
        print(f"Normalization: {model_config.get('normalization_method', 'minmax')}")

        if hasattr(model, "product_scores"):
            scores = [s for s in model.product_scores.values() if not np.isnan(s)]
            if scores:
                print(f"\nScore distribution (n={len(scores)}):")
                print(f"  min={min(scores):.4f}  max={max(scores):.4f}  "
                      f"mean={np.mean(scores):.4f}  std={np.std(scores):.4f}")

            code_to_cat = dict(
                zip(labels_df["product_code"].astype(int), labels_df["category"])
            )
            print(f"\nPer-category score stats:")
            for cat in sorted(set(code_to_cat.values())):
                cat_scores = [
                    model.product_scores.get(c, np.nan)
                    for c in labels_df[labels_df["category"] == cat]["product_code"].astype(int)
                    if not np.isnan(model.product_scores.get(c, np.nan))
                ]
                if cat_scores:
                    print(f"  {cat:30s}  n={len(cat_scores):3d}  "
                          f"min={min(cat_scores):.4f}  max={max(cat_scores):.4f}")

    print(f"\nSubmission: {submission_path}")


def main():
    parser = argparse.ArgumentParser(description="Run pairwise ranking experiments.")
    parser.add_argument(
        "--config", type=str, help="Path to a single YAML config file."
    )
    parser.add_argument(
        "--name", type=str, default=None, help="Experiment name override."
    )
    parser.add_argument(
        "--all", action="store_true", help="Run all distance configs in configs/."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume an LLM experiment from an existing log, skipping completed pairs.",
    )
    parser.add_argument(
        "--nectar-only", action="store_true",
        help="Only run LLM on nectar-vs-nectar pairs; use deterministic fallback for the rest.",
    )
    args = parser.parse_args()

    if args.all:
        baselines_dir = Path(__file__).parent
        all_configs = sorted((baselines_dir / "configs").glob("**/*.yaml"))
        # Separate LLM configs (require API key + cost $$$) from distance configs
        dist_configs = [c for c in all_configs if "/llm/" not in str(c)]
        llm_configs = [c for c in all_configs if "/llm/" in str(c)]
        logger.info(
            f"Running {len(dist_configs)} distance configs "
            f"(skipping {len(llm_configs)} LLM configs — use --config to run individually)"
        )
        for cfg in dist_configs:
            run_experiment(str(cfg))
    elif args.config:
        run_experiment(args.config, args.name, resume=args.resume,
                       nectar_only=args.nectar_only)
    else:
        parser.error("Either --config or --all is required.")


if __name__ == "__main__":
    main()
