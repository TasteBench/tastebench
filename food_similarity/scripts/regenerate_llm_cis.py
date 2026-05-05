"""Phase B.4: regenerate BCa CIs for all 14 LLM OOFs.

Parallelized across OOFs (compute_bca_cis is internally serial).
Emits one row per OOF/metric plus a byte-match check of point estimates
vs. archived table_results.tex values (Gemini ing+img .654, Qwen ing+img .630).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERVISED_DIR))

import pandas as pd
from joblib import Parallel, delayed

from evaluation.bootstrap import compute_bca_cis
from evaluation.metrics import compute_all_metrics

OOF_DIR = SUPERVISED_DIR / "results" / "oof_predictions"
OUT_PATH = SUPERVISED_DIR / "results" / "llm_bootstrap_cis.csv"

LLM_OOFS = sorted(p.name for p in OOF_DIR.glob("llm_*.csv"))
N_BOOTSTRAP = 10_000
SEED = 42


def process_one(name: str) -> dict:
    df = pd.read_csv(OOF_DIR / name)
    point = compute_all_metrics(df)
    ci = compute_bca_cis(df, n_bootstrap=N_BOOTSTRAP, seed=SEED)
    row = {"oof": name.replace(".csv", "")}
    for metric in ("pairwise_accuracy", "spearman", "kendall_tau",
                   "recall_at_1", "recall_at_2", "recall_at_3"):
        row[f"{metric}_point"] = point[metric]
        row[f"{metric}_lo"], row[f"{metric}_hi"] = ci[metric]
    return row


def main() -> int:
    t0 = time.time()
    print(f"Regenerating BCa CIs for {len(LLM_OOFS)} LLM OOFs "
          f"(n_bootstrap={N_BOOTSTRAP}, seed={SEED}) — parallel outer loop.")
    rows = Parallel(n_jobs=-1, backend="loky")(
        delayed(process_one)(name) for name in LLM_OOFS
    )
    df = pd.DataFrame(rows).sort_values("oof").reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False, float_format="%.6f")
    print(f"Wrote {OUT_PATH.relative_to(SUPERVISED_DIR.parent)} in {time.time()-t0:.1f}s\n")

    # Byte-match check: archived points for the 2 rows that appear in table_results.tex
    anchors = {
        "llm_gemini_3_1_pro_preview_ingredients_image": 0.654,
        "llm_qwen3_5_397b_a17b_ingredients_image": 0.630,
    }
    print("Point-estimate byte-match check vs. archived table_results.tex:")
    all_ok = True
    for oof, archived in anchors.items():
        row = df[df["oof"] == oof].iloc[0]
        regen_rounded = round(row["pairwise_accuracy_point"], 3)
        match = regen_rounded == archived
        print(f"  {'OK ' if match else 'FAIL'}  {oof:<55} "
              f"regen={row['pairwise_accuracy_point']:.6f} → {regen_rounded:.3f}  "
              f"archived={archived:.3f}")
        all_ok = all_ok and match

    print("\nFull table:")
    print(df[["oof", "pairwise_accuracy_point",
              "pairwise_accuracy_lo", "pairwise_accuracy_hi"]].to_string(index=False))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
