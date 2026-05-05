"""Render table_ablation_llm.tex.

Zero-shot LLM modality ablation: pairwise accuracy with 95% BCa CIs
(10,000 resamples, seed 42) for Gemini 3.1 Pro and Qwen 3.5 397B-A17B
across 7 input subsets (Ingr., Nutr., Img., and pairwise/triple combos).

Reads precomputed CIs from results/llm_bootstrap_cis.csv (produced by
scripts/regenerate_llm_cis.py during reproduce.sh Phase 8). Bolds the
best row per model.

Usage:
    cd food_similarity
    python scripts/render_table_ablation_llm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
CIS_CSV = SUPERVISED_DIR / "results" / "llm_bootstrap_cis.csv"
TABLE_OUT = SUPERVISED_DIR.parent / "paper" / "model_results_tables" / "table_ablation_llm.tex"

# (display label, modality suffix used in the OOF filename)
MODALITIES = [
    ("Ingr.",        "ingredients"),
    ("Nutr.",        "nutrition"),
    ("Img.",         "image"),
    ("Ingr.+Nutr.",  "ingredients_nutrition"),
    ("Ingr.+Img.",   "ingredients_image"),
    ("Nutr.+Img.",   "nutrition_image"),
    ("All",          "ingredients_nutrition_image"),
]
MODELS = [
    ("Gemini", "llm_gemini_3_1_pro_preview"),
    ("Qwen",   "llm_qwen3_5_397b_a17b"),
]


def _overlaps(lo_a: float, hi_a: float,
              lo_b: float, hi_b: float) -> bool:
    """Two CIs overlap iff max(lo_a, lo_b) <= min(hi_a, hi_b)."""
    import math
    if any(math.isnan(v) for v in (lo_a, hi_a, lo_b, hi_b)):
        return False
    return max(lo_a, lo_b) <= min(hi_a, hi_b)


def _fmt(point: float, lo: float, hi: float,
         bold: bool, overlap: bool = False) -> str:
    """Inline ``.XXX [lo,hi]`` cell. Parent table is \\scriptsize, so no
    extra font wrap on the bracket.

    Strips leading "0" so .607 renders rather than 0.607 (matches the
    paper's table style for sub-1.0 accuracies). ``overlap=True`` appends
    ``$^\\dag$`` to mark cells whose CI overlaps the column leader's CI.
    """
    p = f"{point:.3f}".lstrip("0")
    l = f"{lo:.3f}".lstrip("0")
    h = f"{hi:.3f}".lstrip("0")
    if bold:
        return rf"\textbf{{{p}}} [{l},{h}]"
    if overlap:
        return rf"{p}$^\dag$ [{l},{h}]"
    return f"{p} [{l},{h}]"


def main() -> int:
    if not CIS_CSV.exists():
        print(f"Missing {CIS_CSV.relative_to(SUPERVISED_DIR.parent)}; "
              f"run `python scripts/regenerate_llm_cis.py` first.",
              file=sys.stderr)
        return 1
    cis = pd.read_csv(CIS_CSV).set_index("oof")

    # Per model, find the modality with the highest point estimate (used to
    # bold both the label and the cell). Cache the leader's CI for overlap
    # marker computation.
    best_modality = {}
    leader_ci = {}
    for model_label, prefix in MODELS:
        best_point, best_mod = -1.0, None
        for _, suffix in MODALITIES:
            row = cis.loc[f"{prefix}_{suffix}"]
            if row["pairwise_accuracy_point"] > best_point:
                best_point = row["pairwise_accuracy_point"]
                best_mod = suffix
        best_modality[prefix] = best_mod
        row = cis.loc[f"{prefix}_{best_mod}"]
        leader_ci[prefix] = (row["pairwise_accuracy_lo"],
                             row["pairwise_accuracy_hi"])

    # Row label is bolded if either model picks that modality as its best.
    label_is_best = {
        suffix: any(best_modality[prefix] == suffix for _, prefix in MODELS)
        for _, suffix in MODALITIES
    }

    body_lines = []
    for label, suffix in MODALITIES:
        label_str = rf"\textbf{{{label}}}" if label_is_best[suffix] else label
        cells = [label_str]
        for _, prefix in MODELS:
            row = cis.loc[f"{prefix}_{suffix}"]
            bold = best_modality[prefix] == suffix
            ovl = (not bold) and _overlaps(
                row["pairwise_accuracy_lo"],
                row["pairwise_accuracy_hi"],
                *leader_ci[prefix],
            )
            cells.append(_fmt(
                row["pairwise_accuracy_point"],
                row["pairwise_accuracy_lo"],
                row["pairwise_accuracy_hi"],
                bold,
                overlap=ovl,
            ))
        body_lines.append(" & ".join(cells) + r" \\")

    tex = r"""\begin{center}
\captionof{table}{Input ablation: pairwise accuracy with 95\% BCa CIs (10{,}000 resamples) for zero-shot LLMs across modality subsets. Ingr.\ = ingredient list text; Nutr.\ = nutrition facts; Img.\ = product image. \textbf{Bold} = best per model; $^\dag$ = CI overlaps the column leader (no significant difference at 95\%).}
\label{tab:ablation-llm}
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{0.95}
\begin{tabular}{@{}lcc@{}}
\toprule
Input & Gemini & Qwen \\
\midrule
""" + "\n".join(body_lines) + r"""
\bottomrule
\end{tabular}
\end{center}
"""
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.write_text(tex)
    print(f"Wrote {TABLE_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
