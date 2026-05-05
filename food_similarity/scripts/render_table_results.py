"""Render the food-similarity results table (table_results.tex).

A single combined 7-column table sized for the NeurIPS 1-column layout:

    Model | Pw.Acc | rho_S | tau_K | R@1 | R@2 | R@3

Inline ``X.XXX [lo, hi]`` CIs for the bootstrapped metrics
(pairwise accuracy, recall@k); point estimates for the rank
correlations.

Bootstrap method
----------------
Computes BCa CIs via the original ``compute_bca_cis`` in
``evaluation/bootstrap.py`` (Python-loop bootstrap, the implementation
used to generate the submitted-paper numbers). The vectorized
``compute_bca_pw_acc`` in ``evaluation/bootstrap_fast.py`` is ~8x faster
but uses a different internal RNG-call pattern; even with the same
seed=42 it produces CI bounds that drift by approximately +/-0.001 in
the third decimal. Point estimates match exactly between the two.

Inputs
------
OOF prediction CSVs in ``results/oof_predictions/`` (one per row
defined in ``MAIN_ROWS`` below). Each CSV has columns:
``category, product_code, true_score, predicted_score``.

Output
------
``paper/model_results_tables/table_results.tex`` (label: tab:results)
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
OUT_MAIN = SUPERVISED_DIR.parent / "paper" / "model_results_tables" / "table_results.tex"
CACHE = SUPERVISED_DIR / "results" / "cis_table_results.csv"

N_BOOTSTRAP = 10_000
SEED = 42


def _short(v: float) -> str:
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0") else ("$-" + s[2:] + "$" if s.startswith("-0") else s)


def fmt_pw(p: float, lo: float, hi: float) -> str:
    if np.isnan(p):
        return "--"
    if np.isnan(lo) or np.isnan(hi):
        return _short(p)
    return f"{_short(p)} [{_short(lo)}, {_short(hi)}]"


def fmt_stack(p: float, lo: float, hi: float, bold: bool = False) -> str:
    """Stacked makecell: point estimate above [lo, hi] in scriptsize.

    Used for two-column NeurIPS layout where the inline ``p [lo, hi]``
    form would overflow ``\\columnwidth``.
    """
    if np.isnan(p):
        return "--"
    pt = _short(p)
    if bold:
        pt = r"\textbf{" + pt + "}"
    if np.isnan(lo) or np.isnan(hi):
        return pt
    return r"\makecell{" + pt + r" \\ {\scriptsize [" + _short(lo) + ", " + _short(hi) + r"]}}"


# row key -> (display label, OOF filename)
# Section headers and the trivially-derivable Random baseline are
# elided to save vertical space. Random performance is documented in
# the table caption. Section groupings are conveyed via thin
# \midrule separators between rows of similar provenance.
MAIN_ROWS = [
    ("gemini",       "Gemini 3.1 Pro",   "llm_gemini_3_1_pro_preview_ingredients_image.csv"),
    ("qwen",         "Qwen 3.5 397B",    "llm_qwen3_5_397b_a17b_ingredients_image.csv"),
    ("mmrf_cos",     "MMRF (cosine)",    "dist_pred_cosine_NCI.csv"),
    ("mmrf_l2",      "MMRF (L2)",        "dist_pred_l2_NCI.csv"),
    ("midrule", None, None),
    ("ridge",        "Ridge",            "ridge_SNCTI_bench.csv"),
    ("bt",           "Bradley--Terry",   "bradley_terry_SNCTI_bench.csv"),
    ("hbt",          "Hierarchical BT",  "hierarchical_bt_SNCTI_bench.csv"),
    ("ksvm",         "Kernel RankSVM",   "kernel_ranksvm_SNCTI_bench.csv"),
    ("lgbm",         "LightGBM",         "lightgbm_reg_SNCTI_bench.csv"),
    ("midrule", None, None),
    ("nnls",         "NNLS (BT+Gemini)", "nested_bt_gemini_nnls.csv"),
    ("rankavg",      "Rank avg.",        "nested_bt_gemini_rank.csv"),
    ("mean",         "Mean",             "nested_bt_gemini_mean.csv"),
    ("midrule", None, None),
    ("random",       "Random (analytical)", None),
]

def theoretical_random_baseline() -> dict[str, str]:
    """Closed-form macro-averaged baselines for a uniformly-random ranker.

    - Pairwise accuracy: $\\Pr[\\hat{y}_i > \\hat{y}_j \\mid y_i > y_j] = 1/2$.
    - Spearman / Kendall: no rank correlation in expectation.
    - R@$k$ in category $c$ of size $n_c$ is the probability that the
      true best product lands in the top $k$ of a uniform random
      permutation, $= \\min(k, n_c) / n_c$. Macro-averaging gives
      $\\bar{R}_k = (1/|C|) \\sum_c \\min(k, n_c)/n_c$.

    Computed at render time from the actual category sizes recorded
    in any of the LOOCV OOF CSVs (taking ``bradley_terry_SNCTI_bench``
    as the canonical reference, since it covers all 215 products).
    """
    df = pd.read_csv(OOF / "bradley_terry_SNCTI_bench.csv")
    n_per_cat = df.groupby("category").size()
    return {
        "pairwise_accuracy": _short(0.5),
        "spearman":          _short(0.0),
        "kendall_tau":       _short(0.0),
        "recall_at_1":       _short((np.minimum(1, n_per_cat) / n_per_cat).mean()),
        "recall_at_2":       _short((np.minimum(2, n_per_cat) / n_per_cat).mean()),
        "recall_at_3":       _short((np.minimum(3, n_per_cat) / n_per_cat).mean()),
    }


def compute_row(oof_file: str) -> dict:
    """Compute point estimates + BCa CIs for all 6 metrics on one OOF."""
    df = pd.read_csv(OOF / oof_file).dropna(subset=["predicted_score", "true_score"])
    metrics = compute_all_metrics(df)
    cis = compute_bca_cis(df, n_bootstrap=N_BOOTSTRAP, seed=SEED)
    return {"metrics": metrics, "cis": cis}


METRICS = [
    ("pairwise_accuracy", r"Pw.\ Acc.",            True),
    ("spearman",          r"$\rho_{\mathrm{S}}$",  True),
    ("kendall_tau",       r"$\tau_{\mathrm{K}}$",  True),
    ("recall_at_1",       "R@1",                   True),
    ("recall_at_2",       "R@2",                   True),
    ("recall_at_3",       "R@3",                   True),
]


def render_main(rows_data: dict) -> str:
    """Single combined 7-column table for the 1-column NeurIPS layout."""
    best_per: dict[str, set] = {}
    leader_ci: dict[str, tuple[float, float]] = {}
    for metric, _, has_ci in METRICS:
        max_v = max(d["metrics"][metric] for d in rows_data.values())
        best_per[metric] = {key for key, d in rows_data.items()
                            if abs(d["metrics"][metric] - max_v) < 1e-9}
        if has_ci:
            for key, d in rows_data.items():
                if abs(d["metrics"][metric] - max_v) < 1e-9:
                    leader_ci[metric] = d["cis"][metric]
                    break

    def cell(key: str, metric: str, has_ci: bool) -> str:
        d = rows_data[key]
        is_best = key in best_per[metric]
        if has_ci:
            lo, hi = d["cis"][metric]
            overlap = (not is_best) and metric in leader_ci \
                and _overlaps(lo, hi, *leader_ci[metric])
            return fmt_pw_bold(d["metrics"][metric], lo, hi,
                               bold=is_best, overlap=overlap)
        s = _short(d["metrics"][metric])
        if is_best:
            s = f"\\textbf{{{s}}}"
        return s

    header = " & ".join(["Model"] + [h for _, h, _ in METRICS]) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\caption{Ranking metrics on 215 NECTAR plant-based products "
        r"(935 within-category pairs; LOOCV) with 95\% BCa CIs. "
        r"\textbf{Bold} = best in column; $^\dag$ = CI overlaps the leader "
        r"(no significant difference at 95\%)}",
        r"\label{tab:results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}l" + "c" * len(METRICS) + r"@{}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    random_baseline = theoretical_random_baseline()
    for key, lbl, f in MAIN_ROWS:
        if key == "midrule":
            lines.append(r"\midrule"); continue
        if key == "random":
            random_cells = [random_baseline[m] for m, _, _ in METRICS]
            lines.append(f"{lbl} & " + " & ".join(random_cells) + r" \\")
            continue
        cells = [cell(key, m, has_ci) for m, _, has_ci in METRICS]
        lines.append(f"{lbl} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}", ""]
    return "\n".join(lines)


def fmt_pw_bold(p: float, lo: float, hi: float,
                bold: bool, overlap: bool = False) -> str:
    """Compact inline cell: ``X.XXX [lo,hi]`` (no inner spaces).

    The parent table's \\scriptsize already keeps the cell narrow; an
    extra \\scriptsize wrap on the bracket would push it to \\tiny and
    hurt legibility, so we just rely on the parent font size.

    ``overlap=True`` appends ``$^\\dag$`` to the point estimate to mark
    cells whose 95% BCa CI overlaps the column leader's CI (no
    significant difference at 95%).
    """
    if np.isnan(p):
        return "--"
    pt = _short(p)
    if bold:
        pt = f"\\textbf{{{pt}}}"
    elif overlap:
        pt = f"{pt}$^\\dag$"
    if np.isnan(lo) or np.isnan(hi):
        return pt
    return f"{pt} [{_short(lo)},{_short(hi)}]"


def _overlaps(lo_a: float, hi_a: float,
              lo_b: float, hi_b: float) -> bool:
    """Two intervals [lo_a, hi_a] and [lo_b, hi_b] overlap iff
    max(lo_a, lo_b) <= min(hi_a, hi_b). NaN inputs return False."""
    if any(np.isnan(v) for v in (lo_a, hi_a, lo_b, hi_b)):
        return False
    return max(lo_a, lo_b) <= min(hi_a, hi_b)


def _load_cache(rows_with_files) -> dict | None:
    """Read CSV cache of point + 6-metric BCa CIs keyed on row name.

    CSV schema: row, metric, point, ci_lo, ci_hi
    Returns ``rows_data`` in the format ``compute_row`` produces, or
    ``None`` if the cache is missing / doesn't cover every requested row.
    """
    if not CACHE.exists():
        return None
    df = pd.read_csv(CACHE)
    rows_data: dict = {}
    for key, _ in rows_with_files:
        sub = df[df["row"] == key]
        if sub.empty:
            print(f"Cache missing row {key!r}; recomputing all.", file=sys.stderr)
            return None
        metrics, cis = {}, {}
        for r in sub.itertuples(index=False):
            metrics[r.metric] = float(r.point)
            cis[r.metric] = (float(r.ci_lo), float(r.ci_hi))
        rows_data[key] = {"metrics": metrics, "cis": cis}
    return rows_data


def _save_cache(rows_data: dict) -> None:
    rows = []
    for key, d in sorted(rows_data.items()):
        for metric, point in d["metrics"].items():
            lo, hi = d["cis"].get(metric, (float("nan"), float("nan")))
            rows.append((key, metric, point, lo, hi))
    out = pd.DataFrame(rows, columns=["row", "metric", "point", "ci_lo", "ci_hi"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CACHE, index=False, float_format="%.6f")
    print(f"Wrote cache → {CACHE.relative_to(SUPERVISED_DIR.parent)}")


def main() -> int:
    rows_with_files = [(key, f) for key, _, f in MAIN_ROWS if f]

    rows_data = _load_cache(rows_with_files)
    if rows_data is not None:
        print(f"Loaded BCa CIs for {len(rows_data)} rows from "
              f"{CACHE.relative_to(SUPERVISED_DIR.parent)}; skipping bootstrap.")
    else:
        print(f"Computing BCa CIs for {len(rows_with_files)} OOF files in parallel "
              f"({N_BOOTSTRAP} resamples, seed {SEED}, Python-loop bootstrap). "
              f"Output cached for instant re-renders.")
        # Each compute_row is independent: parallelize across rows. Loky
        # backend avoids sklearn/numpy thread oversubscription.
        results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(compute_row)(f) for _, f in rows_with_files
        )
        rows_data = dict(zip([k for k, _ in rows_with_files], results))
        for key in [k for k, _ in rows_with_files]:
            m = rows_data[key]["metrics"]
            print(f"  {key:<12}  pw_acc={m['pairwise_accuracy']:.4f}  "
                  f"R@1={m['recall_at_1']:.3f}")
        _save_cache(rows_data)

    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    OUT_MAIN.write_text(render_main(rows_data))
    print(f"\nWrote {OUT_MAIN.relative_to(SUPERVISED_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
