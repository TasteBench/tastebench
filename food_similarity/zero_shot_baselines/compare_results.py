"""Compare two result CSVs: agreement rate, Cohen's kappa, per-category breakdown.

Usage:
    python compare_results.py results/cosine_dist/cosine_N.csv results/l2_dist/l2_N.csv
    python compare_results.py results/llm/qwen3_5_397b_a17b/submissions/ingredients_nutrition.csv results/cosine_dist/cosine_NCTI.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def cohens_kappa(y1: np.ndarray, y2: np.ndarray) -> float:
    """Compute Cohen's kappa for two binary label arrays."""
    n = len(y1)
    if n == 0:
        return float("nan")
    observed_agreement = np.mean(y1 == y2)
    # Expected agreement under independence
    p1_a = np.mean(y1)
    p2_a = np.mean(y2)
    expected_agreement = p1_a * p2_a + (1 - p1_a) * (1 - p2_a)
    if expected_agreement == 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def compare(path_a: str, path_b: str, pairs_path: str | None = None) -> None:
    """Load two submission CSVs and print agreement statistics."""
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    merged = df_a.merge(df_b, on="test_id", suffixes=("_a", "_b"))
    if len(merged) == 0:
        print("No overlapping test_ids found.")
        return

    agree = merged["higher_rated_product_a"] == merged["higher_rated_product_b"]

    # Overall stats
    name_a = Path(path_a).stem
    name_b = Path(path_b).stem
    print(f"\nComparing: {name_a}  vs  {name_b}")
    print(f"{'='*60}")
    print(f"Pairs:     {len(merged)}")
    print(f"Agreement: {agree.sum()}/{len(merged)} ({agree.mean():.1%})")

    # Cohen's kappa — encode as binary (did model pick product_code_2?)
    if pairs_path:
        pairs_df = pd.read_csv(pairs_path)
        merged = merged.merge(pairs_df[["test_id", "product_category", "product_code_2"]], on="test_id")
        y_a = (merged["higher_rated_product_a"] == merged["product_code_2"]).astype(int).values
        y_b = (merged["higher_rated_product_b"] == merged["product_code_2"]).astype(int).values
        kappa = cohens_kappa(y_a, y_b)
        print(f"Cohen's k: {kappa:.3f}")

        # Per-category breakdown
        print(f"\n{'Category':<30s}  {'N':>5s}  {'Agree':>6s}  {'Rate':>6s}  {'Kappa':>6s}")
        print("-" * 60)
        for cat in sorted(merged["product_category"].unique()):
            mask = merged["product_category"] == cat
            cat_agree = agree[mask]
            cat_y_a = y_a[mask.values]
            cat_y_b = y_b[mask.values]
            cat_kappa = cohens_kappa(cat_y_a, cat_y_b)
            print(
                f"{cat:<30s}  {mask.sum():>5d}  {cat_agree.sum():>6d}  "
                f"{cat_agree.mean():>5.1%}  {cat_kappa:>6.3f}"
            )
    else:
        print("(Pass --pairs for per-category breakdown and Cohen's kappa)")


def main():
    parser = argparse.ArgumentParser(description="Compare two result CSVs.")
    parser.add_argument("result_a", help="First result CSV path")
    parser.add_argument("result_b", help="Second result CSV path")
    parser.add_argument(
        "--pairs", default=None,
        help="Path to ranking_pairs.csv for per-category breakdown (default: data/competition/ranking_pairs.csv)",
    )
    args = parser.parse_args()

    pairs_path = args.pairs
    if pairs_path is None:
        default = Path(__file__).parent / "data" / "competition" / "ranking_pairs.csv"
        if default.exists():
            pairs_path = str(default)

    compare(args.result_a, args.result_b, pairs_path)


if __name__ == "__main__":
    main()
