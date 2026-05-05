"""Convert LLM pairwise predictions to per-product scores.

Loads LLM logs from zero_shot_baselines/, maps competition product codes
to NECTAR codes, and computes per-product tournament scores (win rates).
Saves results as OOF-like prediction CSVs compatible with the ensemble framework.

Usage:
    cd food_similarity
    python -m train.integrate_llm
"""

import sys
from pathlib import Path

import pandas as pd

FOOD_SIM_DIR = Path(__file__).resolve().parent.parent
LLM_RESULTS_DIR = FOOD_SIM_DIR / "zero_shot_baselines" / "results" / "llm"
CODE_MAP_PATH = FOOD_SIM_DIR.parent / "kaggle_tastebench" / "generate_data" / "dataset" / "product_code_map.csv"

sys.path.insert(0, str(FOOD_SIM_DIR))

from data.loocv import load_product_features, get_analog_keys
from evaluation.metrics import compute_all_metrics

OOF_DIR = FOOD_SIM_DIR / "results" / "oof_predictions"


def load_code_map():
    """Load competition→NECTAR product code mapping."""
    df = pd.read_csv(CODE_MAP_PATH)
    nectar = df[df["Source"] == "nectar"]
    # Map: (Category, New_Product_Code) → Original_Product_Code
    comp_to_nectar = {}
    for _, r in nectar.iterrows():
        comp_to_nectar[(r["Category"], int(r["New_Product_Code"]))] = int(r["Original_Product_Code"])
    return comp_to_nectar


def load_llm_log(model_dir, variant):
    """Load an LLM log file."""
    path = model_dir / "logs" / f"{variant}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def llm_to_product_scores(log, comp_to_nectar, pf):
    """Convert LLM pairwise predictions to per-product scores.

    For each NECTAR product, computes:
    - Win count from all valid pairs (including cross-source)
    - Total pairs participated in
    - Win rate = wins / total

    Args:
        log: LLM log DataFrame
        comp_to_nectar: mapping from (Category, competition_code) → nectar_code
        pf: product_features dict

    Returns:
        DataFrame with columns: category, product_code, true_score, predicted_score
    """
    analog_keys = set(get_analog_keys(pf))

    # Count wins per NECTAR product
    wins = {}    # (category, nectar_code) → win count
    total = {}   # (category, nectar_code) → total pairs

    for _, r in log.iterrows():
        cat = r["category"]
        code1 = int(r["product_code_1"])
        code2 = int(r["product_code_2"])
        pred = r["prediction"]
        status = r.get("status", "OK")

        if status != "OK":
            continue

        # Map to NECTAR codes
        nectar1 = comp_to_nectar.get((cat, code1))
        nectar2 = comp_to_nectar.get((cat, code2))

        # Determine winner (competition code)
        if pred == 1 or pred == "1":
            winner_comp = code1
        elif pred == 2 or pred == "2":
            winner_comp = code2
        else:
            continue

        winner_nectar = comp_to_nectar.get((cat, winner_comp))

        # Update counts for each NECTAR product involved
        for nectar_code, is_winner in [
            (nectar1, winner_nectar == nectar1 if nectar1 and winner_nectar else False),
            (nectar2, winner_nectar == nectar2 if nectar2 and winner_nectar else False),
        ]:
            if nectar_code is None:
                continue
            key = (cat, nectar_code)
            if key not in analog_keys:
                continue
            total[key] = total.get(key, 0) + 1
            if is_winner:
                wins[key] = wins.get(key, 0) + 1

    # Build results DataFrame
    rows = []
    for key in sorted(analog_keys):
        cat, code = key
        w = wins.get(key, 0)
        t = total.get(key, 0)
        win_rate = w / t if t > 0 else 0.5  # default to 0.5 for products with no data
        rows.append({
            "category": cat,
            "product_code": code,
            "true_score": pf[key]["mean_similarity"],
            "predicted_score": win_rate,
        })

    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    pf = load_product_features()
    comp_to_nectar = load_code_map()
    print(f"Product features: {len(pf)} products")
    print(f"Code map: {len(comp_to_nectar)} NECTAR mappings")

    # Find all LLM model directories
    llm_models = []
    for model_dir in sorted(LLM_RESULTS_DIR.iterdir()):
        if model_dir.is_dir() and (model_dir / "logs").exists():
            llm_models.append(model_dir)

    print(f"Found {len(llm_models)} LLM model directories")

    all_results = []

    for model_dir in llm_models:
        model_name = model_dir.name
        print(f"\n--- {model_name} ---")

        log_dir = model_dir / "logs"
        for log_path in sorted(log_dir.glob("*.csv")):
            variant = log_path.stem
            log = pd.read_csv(log_path)

            # Convert to product scores
            scores_df = llm_to_product_scores(log, comp_to_nectar, pf)

            if len(scores_df) == 0:
                print(f"  {variant}: no valid predictions")
                continue

            # Evaluate
            metrics = compute_all_metrics(scores_df)
            n_with_data = sum(1 for _, r in scores_df.iterrows() if r["predicted_score"] != 0.5)

            print(f"  {variant:<55s} pw_acc={metrics['pairwise_accuracy']:.4f} "
                  f"spearman={metrics['spearman']:.4f} "
                  f"R@1={metrics['recall_at_1']:.3f} R@3={metrics['recall_at_3']:.3f} "
                  f"n={len(scores_df)} ({n_with_data} with data)")

            # Save OOF predictions
            tag = f"llm_{variant}"
            scores_df.to_csv(OOF_DIR / f"{tag}.csv", index=False)

            all_results.append({
                "model": model_name,
                "variant": variant,
                "n": len(scores_df),
                "n_with_data": n_with_data,
                **metrics,
            })

    # Summary table
    print("\n" + "=" * 100)
    print("LLM PREDICTION SUMMARY (sorted by pairwise accuracy)")
    print("=" * 100)
    results_df = pd.DataFrame(all_results).sort_values("pairwise_accuracy", ascending=False)
    for _, row in results_df.iterrows():
        print(f"  {row['variant']:<55s} pw_acc={row['pairwise_accuracy']:.4f} "
              f"spearman={row['spearman']:.4f} n_data={row['n_with_data']}")



if __name__ == "__main__":
    main()
