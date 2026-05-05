"""Render table_ablation_features.tex.

Feature ablation: pairwise accuracy with 95% BCa CIs (10,000 resamples)
for 5 supervised models and 2 MMRF distance predictors across 15 feature
subsets (Cartesian product of N=Nutrition, C=Compound, T=Text, I=Image,
minus the empty set).

Reads from results/oof_predictions/{model}_{subset}.csv. For full SNCTI,
falls back to the `_bench` suffix used by the main results table.

Single-column rendering: wraps the tabular in
\\resizebox{\\columnwidth}{!}{...} so it scales to one column of a
two-column layout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERVISED_DIR))

from evaluation.bootstrap import compute_bca_cis
from evaluation.metrics import compute_all_metrics

OOF = SUPERVISED_DIR / "results" / "oof_predictions"
OUT_TEX = SUPERVISED_DIR.parent / "paper" / "model_results_tables" / "table_ablation_features.tex"
CACHE = SUPERVISED_DIR / "results" / "cis_ablation_features.csv"

N_BOOT = 10_000
SEED = 42

# 15 feature subsets (powerset of {N, C, T, I} minus empty), grouped by size
# for visual midrules (singles | pairs | triples | full).
SUBSETS = [
    ["N", "C", "T", "I"],
    ["NC", "NT", "NI", "CT", "CI", "TI"],
    ["NCT", "NCI", "NTI", "CTI"],
    ["NCTI"],
]

# (column_label, model_key, oof_filename_pattern)
# Supervised models prefix the subset with S; MMRF distance predictors do not.
COLS = [
    ("Ridge",    "ridge",           "ridge_S{s}.csv"),
    ("BT",       "bradley_terry",   "bradley_terry_S{s}.csv"),
    ("HBT",      "hierarchical_bt", "hierarchical_bt_S{s}.csv"),
    ("KSVM",     "kernel_ranksvm",  "kernel_ranksvm_S{s}.csv"),
    ("LGBM",     "lightgbm_reg",    "lightgbm_reg_S{s}.csv"),
    ("Cos",      "dist_pred_cosine", "dist_pred_cosine_{s}.csv"),
    ("L2",       "dist_pred_l2",    "dist_pred_l2_{s}.csv"),
]


def _short(v: float) -> str:
    if np.isnan(v):
        return "--"
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0") else s


def _overlaps(lo_a: float, hi_a: float,
              lo_b: float, hi_b: float) -> bool:
    """Two CIs overlap iff max(lo_a, lo_b) <= min(hi_a, hi_b)."""
    if any(np.isnan(v) for v in (lo_a, hi_a, lo_b, hi_b)):
        return False
    return max(lo_a, lo_b) <= min(hi_a, hi_b)


def _ci_cell(point: float, lo: float, hi: float,
             bold: bool = False, overlap: bool = False) -> str:
    """Stacked two-line cell: point on top, CI in \\scriptsize below.
    Stacked layout lets the 8-column table fit a single NeurIPS column
    width at \\footnotesize without scaling the whole table down via
    \\resizebox (which pushes effective text size below 6pt). Uses
    built-in \\shortstack — no extra package dependency.

    ``overlap=True`` appends ``$^\\dag$`` to mark cells whose CI overlaps
    the column leader's CI."""
    p_str = _short(point)
    if bold:
        p_str = f"\\textbf{{{p_str}}}"
    elif overlap:
        p_str = f"{p_str}$^\\dag$"
    if np.isnan(lo) or np.isnan(hi):
        return p_str
    return (f"\\shortstack{{{p_str} \\\\ "
            f"{{\\scriptsize [{_short(lo)},{_short(hi)}]}}}}")


def _resolve_path(pattern: str, subset: str) -> Path:
    """Resolve OOF path. For full-SNCTI rows we prefer the `_bench` file
    (full 215-product LOOCV used by render_table_results.py). For partial
    subsets, only the non-bench variant exists. Distance predictors do not
    have `_bench` variants at all."""
    primary = OOF / pattern.format(s=subset)
    bench = primary.with_name(primary.stem + "_bench" + primary.suffix)
    # Prefer _bench when it exists (canonical full-LOOCV).
    if bench.exists():
        return bench
    if primary.exists():
        return primary
    raise FileNotFoundError(
        f"No OOF found for pattern {pattern!r} subset={subset!r} "
        f"(tried {primary.name} and {bench.name})"
    )


def _bca_pw_acc(oof_path: Path) -> tuple[float, float, float]:
    df = pd.read_csv(oof_path).dropna(subset=["predicted_score", "true_score"])
    point = compute_all_metrics(df)["pairwise_accuracy"]
    cis = compute_bca_cis(df, n_bootstrap=N_BOOT, seed=SEED)
    lo, hi = cis["pairwise_accuracy"]
    return point, lo, hi


def _load_cache() -> dict[tuple[str, str], tuple[float, float, float]] | None:
    """Load committed BCa CIs keyed on (model, subset). Return None if
    the cache is missing or doesn't cover every (model, subset) pair
    the table needs."""
    if not CACHE.exists():
        return None
    df = pd.read_csv(CACHE)
    by_key = {(r.model, r.subset): (float(r.point), float(r.ci_lo), float(r.ci_hi))
              for r in df.itertuples(index=False)}
    flat = [s for group in SUBSETS for s in group]
    expected = {(c, s) for c, _, _ in COLS for s in flat}
    if not expected.issubset(by_key.keys()):
        missing = expected - by_key.keys()
        print(f"Cache missing {len(missing)} entries (e.g. {next(iter(missing))}); "
              f"recomputing all.", file=sys.stderr)
        return None
    return by_key


def _save_cache(by_key: dict[tuple[str, str], tuple[float, float, float]]) -> None:
    rows = [(c, s, p, lo, hi) for (c, s), (p, lo, hi) in sorted(by_key.items())]
    df = pd.DataFrame(rows, columns=["model", "subset", "point", "ci_lo", "ci_hi"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False, float_format="%.6f")
    print(f"Wrote cache → {CACHE.relative_to(SUPERVISED_DIR.parent)}")


def main() -> int:
    flat_subsets = [s for group in SUBSETS for s in group]

    by_key = _load_cache()
    if by_key is not None:
        print(f"Loaded {len(by_key)} cached BCa CIs from "
              f"{CACHE.relative_to(SUPERVISED_DIR.parent)}; "
              f"skipping bootstrap.")
    else:
        todo = []
        for col_label, _, pattern in COLS:
            for subset in flat_subsets:
                path = _resolve_path(pattern, subset)
                todo.append((col_label, subset, path))

        print(f"Bootstrapping {len(todo)} OOFs in parallel "
              f"({N_BOOT} resamples, seed {SEED}). "
              f"~25 min on 8 cores; output cached to "
              f"{CACHE.relative_to(SUPERVISED_DIR.parent)} for instant re-renders.")
        cis = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(_bca_pw_acc)(t[2]) for t in todo
        )
        by_key = {(t[0], t[1]): cis[i] for i, t in enumerate(todo)}
        _save_cache(by_key)

    # Per-column best (max pw_acc), used to bold the best subset per model.
    # Compare in 3-decimal precision so display ties are all bolded
    # (matches the original frozen-artifact convention).
    best_by_col: dict[str, set] = {}
    leader_ci_by_col: dict[str, tuple[float, float]] = {}
    for col_label, _, _ in COLS:
        rounded = {s: round(by_key[(col_label, s)][0], 3) for s in flat_subsets}
        max_r = max(rounded.values())
        best_by_col[col_label] = {s for s, r in rounded.items() if r == max_r}
        leader_subset = next(s for s in flat_subsets if rounded[s] == max_r)
        _, lo_l, hi_l = by_key[(col_label, leader_subset)]
        leader_ci_by_col[col_label] = (lo_l, hi_l)

    lines = [
        r"\begin{center}",
        r"\captionof{table}{Feature ablation: pairwise accuracy (point on top, "
        r"95\% BCa CI from 10{,}000 resamples below) for supervised models "
        r"and unsupervised distance predictors. S (category subset) is "
        r"always included for supervised models. N = nutrition, C = "
        r"compound, T = text, I = image. BT = Bradley--Terry, HBT = "
        r"Hierarchical BT, KSVM = Kernel RankSVM, LGBM = LightGBM. "
        r"\textbf{Bold} = best subset per model; $^\dag$ = CI overlaps the "
        r"column leader (no significant difference at 95\%). Values below "
        r".500 indicate worse-than-random.}",
        r"\label{tab:ablation}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r" & \multicolumn{5}{c}{Supervised (S+subset)} & \multicolumn{2}{c}{MMRF} \\",
        r"\cmidrule(lr){2-6} \cmidrule(lr){7-8}",
        r"Subset & " + " & ".join(c for c, _, _ in COLS) + r" \\",
        r"\midrule",
    ]

    for gi, group in enumerate(SUBSETS):
        if gi > 0:
            lines.append(r"\midrule")
        for subset in group:
            cells = []
            for col_label, _, _ in COLS:
                p, lo, hi = by_key[(col_label, subset)]
                bold = subset in best_by_col[col_label]
                ovl = (not bold) and _overlaps(
                    lo, hi, *leader_ci_by_col[col_label]
                )
                cells.append(_ci_cell(p, lo, hi, bold=bold, overlap=ovl))
            lines.append(f"{subset} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines))
    print(f"\nWrote {OUT_TEX.relative_to(SUPERVISED_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
