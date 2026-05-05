"""Convert an LLM OOF prediction to a Kaggle TasteBench submission.

The food-similarity LLM baselines (gemini, qwen) live in
``food_similarity/zero_shot_baselines/`` and write per-product
predicted scores to:

    food_similarity/results/oof_predictions/llm_<model>_<modality>.csv

Those rows are keyed on **NECTAR** product codes, but the public Kaggle
competition (``tastebench-challenge-2026``) renumbers product codes
and interleaves Taste Like distractor products. This script produces a
Kaggle-format submission CSV from an existing LLM OOF without
re-running the LLM:

  1. Read the LLM OOF and build a ``(category, NECTAR_code) → score`` map.
  2. Read ``product_code_map.csv`` to translate NECTAR codes into the
     Kaggle codes used in ``ranking_pairs.csv``.
  3. For each Kaggle pair, look up the LLM scores for both products
     and emit the higher-scoring one as ``higher_rated_product``.

Taste Like distractor products were never seen by the LLM (they are
not in NECTAR). Pairs where one or both products are Taste Like are
broken at random with a fixed seed so the submission is deterministic;
the script reports the proportion of randomized pairs to stderr so the
user can interpret the leaderboard score appropriately.

Reads:
  - food_similarity/results/oof_predictions/llm_<model>_<modality>.csv
  - kaggle_tastebench/generate_data/dataset/product_code_map.csv  (gated)
  - kaggle_tastebench/generate_data/dataset/ranking_pairs.csv

Writes:
  - <out>.csv  (test_id, higher_rated_product) — upload to Kaggle

Usage:
    python kaggle_tastebench/predict/llm_to_kaggle_submission.py \\
        --model gemini_3_1_pro_preview \\
        --modality ingredients_image \\
        --out submission_gemini.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = KAGGLE_DIR / "generate_data" / "dataset"
NEURIPS_DIR = KAGGLE_DIR.parent
OOF_DIR = NEURIPS_DIR / "food_similarity" / "results" / "oof_predictions"

CODE_MAP_CSV = DATASET_DIR / "product_code_map.csv"
PAIRS_CSV = DATASET_DIR / "ranking_pairs.csv"

SEED = 42


def load_oof(model: str, modality: str) -> pd.DataFrame:
    name = f"llm_{model}_{modality}.csv"
    path = OOF_DIR / name
    if not path.exists():
        raise SystemExit(
            f"Missing LLM OOF: {path.relative_to(NEURIPS_DIR.parent)}\n"
            f"Available models/modalities under "
            f"{OOF_DIR.relative_to(NEURIPS_DIR.parent)}: "
            + ", ".join(sorted(p.name.removeprefix('llm_').removesuffix('.csv')
                               for p in OOF_DIR.glob('llm_*.csv')))
        )
    df = pd.read_csv(path)
    expected = {"category", "product_code", "predicted_score"}
    if not expected.issubset(df.columns):
        raise SystemExit(
            f"OOF {path.name} is missing columns; expected {expected}, got {set(df.columns)}"
        )
    return df[["category", "product_code", "predicted_score"]]


def load_code_map() -> pd.DataFrame:
    if not CODE_MAP_CSV.exists():
        raise SystemExit(
            f"Missing {CODE_MAP_CSV.relative_to(NEURIPS_DIR.parent)}.\n"
            f"This file maps NECTAR product codes to the renumbered Kaggle "
            f"codes and is gated under NECTAR's NDA. Request access via the "
            f"Google Form linked in {NEURIPS_DIR / 'data' / 'GATED.md'}."
        )
    df = pd.read_csv(CODE_MAP_CSV)
    expected = {"Category", "Original_Product_Code", "New_Product_Code", "Source"}
    if not expected.issubset(df.columns):
        raise SystemExit(
            f"product_code_map.csv columns are {set(df.columns)}, "
            f"expected superset of {expected}"
        )
    return df


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True,
                    help="LLM model tag, e.g. gemini_3_1_pro_preview")
    ap.add_argument("--modality", required=True,
                    help="Modality combo, e.g. ingredients_image")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output Kaggle submission CSV path")
    args = ap.parse_args(argv)

    oof = load_oof(args.model, args.modality)
    code_map = load_code_map()
    pairs = pd.read_csv(PAIRS_CSV)

    # NECTAR-keyed lookup: (category, nectar_code) -> predicted_score
    nectar_lookup: dict[tuple[str, int], float] = {
        (r.category, int(r.product_code)): float(r.predicted_score)
        for r in oof.itertuples(index=False)
    }

    # Kaggle code -> (category, nectar_code) for nectar-sourced rows;
    # taste_like rows have no nectar_code, so map to None.
    kaggle_to_nectar: dict[tuple[str, int], int | None] = {}
    for r in code_map.itertuples(index=False):
        kaggle_to_nectar[(r.Category, int(r.New_Product_Code))] = (
            int(r.Original_Product_Code) if r.Source == "nectar" else None
        )

    def score(category: str, kaggle_code: int) -> float | None:
        nectar = kaggle_to_nectar.get((category, kaggle_code))
        if nectar is None:
            return None
        return nectar_lookup.get((category, nectar))

    rng = np.random.default_rng(SEED)
    submission_rows: list[dict] = []
    randomised = 0
    no_score = 0
    for r in pairs.itertuples(index=False):
        s1 = score(r.product_category, int(r.product_code_1))
        s2 = score(r.product_category, int(r.product_code_2))
        if s1 is None or s2 is None:
            # At least one product is Taste Like (or missing) — coin flip.
            randomised += 1
            higher = int(r.product_code_1 if rng.random() < 0.5 else r.product_code_2)
        elif s1 == s2:
            no_score += 1
            higher = int(r.product_code_1 if rng.random() < 0.5 else r.product_code_2)
        else:
            higher = int(r.product_code_1 if s1 > s2 else r.product_code_2)
        submission_rows.append({"test_id": int(r.test_id),
                                "higher_rated_product": higher})

    out_df = pd.DataFrame(submission_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(f"Wrote {args.out}  ({len(out_df)} pairs)")
    if randomised:
        print(f"  {randomised} pairs randomised (one or both products are Taste Like "
              f"distractors, never scored by the LLM)", file=sys.stderr)
    if no_score:
        print(f"  {no_score} pairs broken at random due to LLM score tie", file=sys.stderr)
    coverage = (len(out_df) - randomised) / len(out_df) if len(out_df) else 0
    print(f"  LLM coverage: {coverage:.1%} of pairs", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
