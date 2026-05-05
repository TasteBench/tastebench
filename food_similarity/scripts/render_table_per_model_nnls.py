"""Render table_per_model_nnls.tex.

Per-supervised-model NNLS ensemble: each row is "{model} + Gemini NNLS"
with the canonical SNCTI feature configuration. Adds a new view onto
the paper's main results that asks: "does swapping the supervised
component of the ensemble change the top-line?"

Reads nested_{short}_gemini_nnls.csv for short in {bt, hbt, ridge,
lgbm, ksvm}. Each file is produced by compute_per_model_nnls.py with
SUPERVISED_MODEL set to the appropriate value (see regenerate_per_model_nnls.py).

For comparison context, also includes the standalone supervised model
({model}_SNCTI_bench.csv) and Gemini-only (the LLM OOF) at the top of
the table.

95% BCa CIs (10,000 resamples, seed 42) on every cell.

Usage:
    cd food_similarity
    python scripts/render_table_per_model_nnls.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERVISED_DIR))

from evaluation.bootstrap_fast import compute_bca_pw_acc  # noqa: E402
from evaluation.metrics import compute_all_metrics  # noqa: E402

OOF = SUPERVISED_DIR / "results" / "oof_predictions"
TABLE_OUT = SUPERVISED_DIR.parent / "paper" / "model_results_tables" / "table_per_model_nnls.tex"

# Map: model short tag → (latex name, standalone OOF, ensemble OOF).
MODELS = [
    ("bt",    r"Bradley--Terry",     "bradley_terry_SNCTI_bench.csv",     "nested_bt_gemini_nnls.csv"),
    ("hbt",   r"Hierarchical BT",    "hierarchical_bt_SNCTI_bench.csv",   "nested_hbt_gemini_nnls.csv"),
    ("ridge", r"Ridge",              "ridge_SNCTI_bench.csv",             "nested_ridge_gemini_nnls.csv"),
    ("ksvm",  r"Kernel RankSVM",     "kernel_ranksvm_SNCTI_bench.csv",    "nested_ksvm_gemini_nnls.csv"),
    ("lgbm",  r"LightGBM",           "lightgbm_reg_SNCTI_bench.csv",      "nested_lgbm_gemini_nnls.csv"),
]
GEMINI_OOF = "llm_gemini_3_1_pro_preview_ingredients_image.csv"

N_BOOTSTRAP = 10_000
SEED = 42


def _load(name: str) -> pd.DataFrame | None:
    path = OOF / name
    if not path.exists():
        return None
    return pd.read_csv(path).dropna(subset=["predicted_score", "true_score"])


def _bca_cell(df: pd.DataFrame) -> tuple[float, float, float]:
    return compute_bca_pw_acc(df, n_bootstrap=N_BOOTSTRAP, seed=SEED)


def _short(v: float) -> str:
    """Format X.XXX as .XXX (project-wide convention; matches other tables)."""
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0") else s


def _overlaps(lo_a: float, hi_a: float,
              lo_b: float, hi_b: float) -> bool:
    """Two CIs overlap iff max(lo_a, lo_b) <= min(hi_a, hi_b)."""
    import math
    if any(math.isnan(v) for v in (lo_a, hi_a, lo_b, hi_b)):
        return False
    return max(lo_a, lo_b) <= min(hi_a, hi_b)


def _fmt_cell(point: float, lo: float, hi: float,
              bold: bool = False, overlap: bool = False) -> str:
    """Stacked cell: point on top, CI in \\scriptsize below. Lets the
    6-column table fit a single NeurIPS column at \\footnotesize without
    \\resizebox (which would scale text inconsistently across appendix
    tables). Uses built-in \\shortstack — no extra package.

    ``overlap=True`` appends ``$^\\dag$`` to mark cells whose CI overlaps
    the column leader's CI."""
    pt = _short(point)
    if bold:
        pt = f"\\textbf{{{pt}}}"
    elif overlap:
        pt = f"{pt}$^\\dag$"
    return (f"\\shortstack{{{pt} \\\\ "
            f"{{\\scriptsize [{_short(lo)},{_short(hi)}]}}}}")


def main() -> int:
    # First pass: collect raw values (None when a row lacks that column).
    # Each entry: (label, std_pt, std_lo, std_hi, ens_pt, ens_lo, ens_hi)
    raw: list[tuple] = []
    gem = _load(GEMINI_OOF)
    if gem is not None:
        p, lo, hi = _bca_cell(gem)
        raw.append(("Gemini 3.1 Pro (alone)", p, lo, hi, None, None, None))

    for short, name, std_oof, ens_oof in MODELS:
        std = _load(std_oof)
        ens = _load(ens_oof)
        std_pt = std_lo = std_hi = None
        ens_pt = ens_lo = ens_hi = None
        if std is not None:
            std_pt, std_lo, std_hi = _bca_cell(std)
        if ens is not None:
            ens_pt, ens_lo, ens_hi = _bca_cell(ens)
        raw.append((name, std_pt, std_lo, std_hi, ens_pt, ens_lo, ens_hi))

    # Per-column max + leader CI, comparing at 3-decimal precision.
    def _col_max_and_leader(col_idx: int):
        pts = [r[col_idx] for r in raw if r[col_idx] is not None]
        if not pts:
            return None, (None, None)
        mx = round(max(pts), 3)
        leader = (None, None)
        for r in raw:
            if r[col_idx] is not None and round(r[col_idx], 3) == mx:
                leader = (r[col_idx + 1], r[col_idx + 2])
                break
        return mx, leader

    std_max, std_leader = _col_max_and_leader(1)
    ens_max, ens_leader = _col_max_and_leader(4)

    def _ci_str(pt, lo, hi, mx, leader):
        if pt is None:
            return "—"
        is_best = mx is not None and round(pt, 3) == mx
        ovl = (not is_best) and leader[0] is not None \
            and _overlaps(lo, hi, leader[0], leader[1])
        return _fmt_cell(pt, lo, hi, bold=is_best, overlap=ovl)

    body_lines = []
    for (label, std_pt, std_lo, std_hi, ens_pt, ens_lo, ens_hi) in raw:
        cells = [
            label,
            _ci_str(std_pt, std_lo, std_hi, std_max, std_leader),
            _ci_str(ens_pt, ens_lo, ens_hi, ens_max, ens_leader),
        ]
        body_lines.append(" & ".join(cells) + r" \\")

    tex = r"""\begin{center}
\captionof{table}{Effect of swapping the supervised base in the
BT+Gemini NNLS ensemble. Each row reports the pairwise accuracy of
a supervised model alone (Standalone) and combined with
Gemini~3.1~Pro via nested LOOCV NNLS (+~Gemini NNLS), on the SNCTI
feature set ($n=215$ NECTAR plant-based products, 935 within-category
pairs). Point estimate above, 95\% BCa CI ($10{,}000$ resamples) below.
\textbf{Bold} = best in column; $^\dag$ = CI overlaps the leader
(no significant difference at 95\%).}
\label{tab:per-model-nnls}
\footnotesize
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.05}
\begin{tabular}{lcc}
\toprule
Supervised model & Standalone & + Gemini NNLS \\
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
