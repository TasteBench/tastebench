"""Sanity-check the 4 canonical paper-table OOFs against expected CIs.

Catches drift between the committed OOFs and the rendered tables —
useful before re-rendering, or after a cache/pickle regeneration.
Tolerance: 0.001 on point estimate or either CI bound triggers a stop.

Targets are the rows reported in Table~1 (canonical FoodAtlas v4.0 run):
  Bradley-Terry (bradley_terry_SNCTI_bench):   .610 [.571, .661]
  NNLS (nested_bt_gemini_nnls):                .683 [.654, .740]
  Gemini 3.1 Pro (ingredients + image):        .654 [.619, .713]
  Qwen 3.5 397B (ingredients + image):         .630 [.591, .688]
"""
from __future__ import annotations

import sys
from pathlib import Path

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERVISED_DIR))

import pandas as pd
from evaluation.bootstrap import compute_bca_cis
from evaluation.metrics import compute_all_metrics

OOF_DIR = SUPERVISED_DIR / "results" / "oof_predictions"
TOLERANCE = 0.001

TARGETS = [
    ("bradley_terry_SNCTI_bench",                       0.6102, 0.5706, 0.6610),
    ("nested_bt_gemini_nnls",                           0.6829, 0.6540, 0.7401),
    ("llm_gemini_3_1_pro_preview_ingredients_image",    0.6540, 0.6187, 0.7132),
    ("llm_qwen3_5_397b_a17b_ingredients_image",         0.6299, 0.5909, 0.6875),
]


def check_one(name: str, pt_arch: float, lo_arch: float, hi_arch: float) -> bool:
    df = pd.read_csv(OOF_DIR / f"{name}.csv")
    point = compute_all_metrics(df)["pairwise_accuracy"]
    ci = compute_bca_cis(df, n_bootstrap=10_000, seed=42)["pairwise_accuracy"]
    lo, hi = ci
    dp, dl, dh = abs(point - pt_arch), abs(lo - lo_arch), abs(hi - hi_arch)
    ok = max(dp, dl, dh) <= TOLERANCE
    status = "PASS" if ok else "FAIL"
    print(
        f"  {status}  {name:<55} "
        f"point={point:.4f} (target {pt_arch:.4f}, Δ={dp:.4f})  "
        f"lo={lo:.4f} (target {lo_arch:.4f}, Δ={dl:.4f})  "
        f"hi={hi:.4f} (target {hi_arch:.4f}, Δ={dh:.4f})"
    )
    return ok


def main() -> int:
    print(f"Paper-table OOF sanity check — tolerance {TOLERANCE} on point/CI-lo/CI-hi\n")
    results = [check_one(*t) for t in TARGETS]
    ok = all(results)
    print(f"\n{'ALL PASS' if ok else 'DRIFT DETECTED — STOP'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
