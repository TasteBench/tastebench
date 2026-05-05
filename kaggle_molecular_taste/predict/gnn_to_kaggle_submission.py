"""Convert GNN test predictions to a Kaggle Molecular Taste submission.

The molecular task's val-best D-MPNN checkpoint emits per-molecule
class probabilities to:

    molecular/results/grid/<best_run>/fart_test_eval/predictions.parquet

That parquet file is keyed by SMILES and includes y_pred (integer in
[0..4] over LABEL_ORDER = [sweet, bitter, sour, umami, undefined]).
The Kaggle test split (kaggle_molecular_taste/dataset/test.csv) is the
same 2,254 molecules as the FART test split but is keyed by an
integer ``id`` and a ``canonicalized_smiles`` column. This script:

  1. Auto-locates the val-best run via results/grid/best/ (or, failing
     that, the first run that has fart_test_eval/predictions.parquet).
  2. Reads predictions.parquet → dict[smiles → predicted taste label].
  3. Reads test.csv → emits a (id, taste) submission CSV.

Reads:
  - molecular/results/grid/<best>/fart_test_eval/predictions.parquet
  - kaggle_molecular_taste/dataset/test.csv

Writes:
  - <out>.csv  (id, taste) — upload to Kaggle Molecular Taste competition

Usage:
    python kaggle_molecular_taste/predict/gnn_to_kaggle_submission.py \\
        --out submission_gnn.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

KAGGLE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = KAGGLE_DIR / "dataset"
NEURIPS_DIR = KAGGLE_DIR.parent
MOLECULAR_RESULTS = NEURIPS_DIR / "molecular" / "results"

# Must match molecular.src.data.dataset.LABEL_ORDER
LABEL_ORDER = ["sweet", "bitter", "sour", "umami", "undefined"]


def find_best_predictions_parquet() -> Path:
    grid = MOLECULAR_RESULTS / "grid"
    best_link = grid / "best"
    if best_link.exists():
        target = best_link.resolve() if best_link.is_symlink() else best_link
        cand = target / "fart_test_eval" / "predictions.parquet"
        if cand.exists():
            return cand
    for cand in sorted(grid.glob("run_*/fart_test_eval/predictions.parquet")):
        return cand
    raise SystemExit(
        f"No fart_test_eval/predictions.parquet under "
        f"{grid.relative_to(NEURIPS_DIR.parent)}. "
        f"Run select_best_and_evaluate.py first."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path,
                    help="Output Kaggle submission CSV path")
    args = ap.parse_args(argv)

    pred_path = find_best_predictions_parquet()
    print(f"Reading {pred_path.relative_to(NEURIPS_DIR.parent)}", file=sys.stderr)
    pred = pd.read_parquet(pred_path)
    smi_to_label = {
        row.smiles: LABEL_ORDER[int(row.y_pred)]
        for row in pred.itertuples(index=False)
    }

    test = pd.read_csv(DATASET_DIR / "test.csv")
    rows: list[dict] = []
    missing = 0
    for r in test.itertuples(index=False):
        label = smi_to_label.get(r.canonicalized_smiles)
        if label is None:
            missing += 1
            label = "undefined"
        rows.append({"id": int(r.id), "taste": label})

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    print(f"Wrote {args.out}  ({len(out_df)} rows)")
    if missing:
        print(f"  WARNING: {missing} test SMILES had no GNN prediction "
              f"(filled with 'undefined')", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
