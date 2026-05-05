"""Compute BCa pairwise-accuracy CIs for many OOF files in parallel.

One BCa per (model_key, subset) cell is serial (compute_bca_cis loops 10k
resamples internally), but the 200+ cells of the 31-subset ablation run
independently → outer-parallelize across cells on 8 cores.

Usage: pass a newline-separated list of OOF file paths on stdin, or use
--discover to enumerate all S-prefixed subset OOFs for the paper columns.

Output: a DataFrame with columns (file, pw_acc_point, pw_acc_lo, pw_acc_hi)
written to --out.
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERVISED_DIR))

from evaluation.bootstrap_fast import compute_bca_pw_acc

OOF_DIR = SUPERVISED_DIR / "results" / "oof_predictions"


def compute_one(path: Path) -> dict:
    df = pd.read_csv(path).dropna(subset=["predicted_score", "true_score"])
    point, lo, hi = compute_bca_pw_acc(df, n_bootstrap=10_000, seed=42)
    try:
        name = str(path.relative_to(OOF_DIR))
    except ValueError:
        name = str(path)
    return {
        "file": name,
        "n_rows": len(df),
        "pw_acc_point": point,
        "pw_acc_lo": lo,
        "pw_acc_hi": hi,
    }


def discover_ablation_oofs() -> list:
    """All S-prefixed NCTI subset OOFs for the 7 ablation columns."""
    subs = ["S" + "".join(c) for r in range(1, 5) for c in combinations("NCTI", r)]
    paths = []
    for s in subs:
        for m in ("ridge", "bradley_terry", "hierarchical_bt",
                  "kernel_ranksvm", "lightgbm_reg"):
            for cand in (f"{m}_{s}.csv", f"{m}_{s}_bench.csv"):
                q = OOF_DIR / cand
                if q.exists():
                    paths.append(q); break
        # MMRF variants (non-S subset: strip leading S)
        pure = s[1:]
        for m in ("dist_pred_cosine", "dist_pred_l2"):
            q = OOF_DIR / f"{m}_{pure}.csv"
            if q.exists():
                paths.append(q)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    if args.discover:
        paths = discover_ablation_oofs()
    else:
        paths = []
        for line in sys.stdin:
            s = line.strip()
            if not s:
                continue
            p = Path(s) if s.startswith("/") else (OOF_DIR / s)
            paths.append(p)
    print(f"Computing BCa CIs for {len(paths)} OOF files on {args.n_jobs} workers...",
          flush=True)

    rows = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(compute_one)(p) for p in paths
    )
    df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.6f")
    print(f"Wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
