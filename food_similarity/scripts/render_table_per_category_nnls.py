"""Per-category metrics for the NNLS ensemble (BT + Gemini).

Single table covering both the per-category evaluation metrics
(pairwise accuracy, Spearman rho) and the inputs to the
theoretical-analysis bound (sigma_S/mu_S, Pearson r). Also reports
the rank the model assigns to the true best-rated product in each
category (this encapsulates R@k since R@k = [rank <= k]).

Skipped intentionally:
  - Kendall tau: highly correlated with Spearman at n = 5-18
  - Per-category R@1/R@2/R@3: degenerate to binary {0, 1} per category;
    macro-aggregated values appear once in the caption.

Run: python food_similarity/scripts/render_table_per_category_nnls.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from joblib import Parallel, delayed

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERVISED_DIR))

from evaluation.bootstrap_fast import compute_bca_pw_acc

NNLS_OOF = SUPERVISED_DIR / "results" / "oof_predictions" / "nested_bt_gemini_nnls.csv"
OUT = SUPERVISED_DIR.parent / "paper" / "model_results_tables" / "table_per_category_nnls.tex"
CACHE = SUPERVISED_DIR / "results" / "cis_per_category_nnls.csv"

N_BOOTSTRAP = 10_000
SEED = 42


def _short(v: float) -> str:
    if np.isnan(v):
        return "--"
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0") else ("$-$" + s[3:] if s.startswith("-0") else s)


def fmt_signed_short(v: float) -> str:
    if np.isnan(v):
        return "--"
    sign = "$+$" if v >= 0 else "$-$"
    return f"{sign}.{int(round(abs(v) * 1000)):03d}"


def fmt_cv(cv: float) -> str:
    if np.isnan(cv):
        return "--"
    return f".{int(round(cv * 1000)):03d}"


def fmt_pwacc(p: float, lo: float, hi: float) -> str:
    """Stacked cell: point on top, CI in \\scriptsize below. Lets the
    7-column table fit a single NeurIPS column at \\footnotesize without
    \\resizebox (which would scale text inconsistently across appendix
    tables). Uses built-in \\shortstack — no extra package."""
    if np.isnan(p):
        return "--"
    pt = _short(p)
    if np.isnan(lo) or np.isnan(hi):
        return pt
    return (f"\\shortstack{{{pt} \\\\ "
            f"{{\\scriptsize [{_short(lo)},{_short(hi)}]}}}}")


def cat_display(c: str) -> str:
    return c.replace("_", " ")


def _bca_for_category(category: str) -> tuple:
    df = pd.read_csv(NNLS_OOF).dropna(subset=["predicted_score", "true_score"])
    df_cat = df[df["category"] == category]
    if len(df_cat) < 2:
        return (np.nan, np.nan, np.nan)
    return compute_bca_pw_acc(df_cat, n_bootstrap=N_BOOTSTRAP, seed=SEED)


def true_best_rank(sub: pd.DataFrame) -> int:
    """Rank assigned to the true-best-rated product (1 = predicted best)."""
    s = sub.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    true_best_code = sub.loc[sub["true_score"].idxmax(), "product_code"]
    return int(s[s["product_code"] == true_best_code].index[0]) + 1


def main() -> None:
    nnls = pd.read_csv(NNLS_OOF).dropna(subset=["predicted_score", "true_score"])

    rows = []
    cats = sorted(nnls["category"].unique())

    bca_results = None
    if CACHE.exists():
        df_cache = pd.read_csv(CACHE)
        cached = {r.category: (float(r.point), float(r.ci_lo), float(r.ci_hi))
                  for r in df_cache.itertuples(index=False)}
        if set(cats).issubset(cached.keys()):
            print(f"Loaded {len(cached)} cached BCa CIs from "
                  f"{CACHE.relative_to(SUPERVISED_DIR.parent)}; skipping bootstrap.")
            bca_results = [cached[c] for c in cats]

    if bca_results is None:
        print(f"Computing BCa CIs for {len(cats)} categories "
              f"(~5 min; cached for instant re-renders)...", flush=True)
        bca_results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(_bca_for_category)(c) for c in cats
        )
        rows_cache = [(c, p, lo, hi) for c, (p, lo, hi) in zip(cats, bca_results)]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows_cache, columns=["category", "point", "ci_lo", "ci_hi"]) \
          .to_csv(CACHE, index=False, float_format="%.6f")
        print(f"Wrote cache → {CACHE.relative_to(SUPERVISED_DIR.parent)}")

    for c, (p, lo, hi) in zip(cats, bca_results):
        sub = nnls[nnls["category"] == c]
        n = len(sub)
        mu = float(sub["true_score"].mean())
        sigma = float(sub["true_score"].std(ddof=1))
        cv = sigma / mu if mu else float("nan")
        if n >= 3:
            r_p = float(pearsonr(sub["true_score"], sub["predicted_score"])[0])
            r_s = float(spearmanr(sub["true_score"], sub["predicted_score"])[0])
        else:
            r_p = float("nan")
            r_s = float("nan")
        rank = true_best_rank(sub) if n >= 1 else float("nan")
        rows.append((c, n, cv, p, lo, hi, r_s, r_p, rank))

    # Macro-aggregated R@k for caption
    ranks = [r[-1] for r in rows]
    r_at_1 = np.mean([1 if rk <= 1 else 0 for rk in ranks])
    r_at_2 = np.mean([1 if rk <= 2 else 0 for rk in ranks])
    r_at_3 = np.mean([1 if rk <= 3 else 0 for rk in ranks])

    lines = [
        r"\begin{center}",
        rf"\captionof{{table}}{{Per-category metrics for the NNLS ensemble (BT $+$ Gemini), "
        rf"covering both the per-category evaluation metrics (Section~\ref{{sec:Tasks}}) "
        rf"and the inputs used in the worked example "
        rf"(Section~\ref{{sec:theoretical}}). $n$ = number of products in "
        rf"category; Pairs $= \binom{{n}}{{2}}$ within-category ranking pairs. "
        rf"$\sigma_S/\mu_S$ is the "
        rf"coefficient of variation of the panel-mean similarity within the "
        rf"category. $\rho_{{\mathrm{{S}}}}$ is Spearman rank correlation; "
        rf"$r$ is Pearson correlation between predicted and panel-mean scores. "
        rf"``True-best rank'' is the position of the highest-rated product in "
        rf"the model's ranking (1 = top). Pairwise accuracy CIs are 95\% BCa "
        rf"(10{{,}}000 resamples). "
        rf"Aggregated across these 24 categories, NNLS achieves "
        rf"R@1 $=$ {_short(r_at_1)}, "
        rf"R@2 $=$ {_short(r_at_2)}, "
        rf"R@3 $=$ {_short(r_at_3)}.}}",
        r"\label{tab:per-category-nnls}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{@{}lrrccccc@{}}",
        r"\toprule",
        r"Category & $n$ & Pairs & $\sigma_S/\mu_S$ & Pw.\ Acc.\ & "
        r"$\rho_{\mathrm{S}}$ & $r$ & True-best rank \\",
        r"\midrule",
    ]

    for c, n, cv, p, lo, hi, r_s, r_p, rank in rows:
        lines.append(
            f"{cat_display(c)} & {n} & {n*(n-1)//2} & {fmt_cv(cv)} & {fmt_pwacc(p, lo, hi)} "
            f"& {fmt_signed_short(r_s)} & {fmt_signed_short(r_p)} "
            f"& {int(rank)} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]

    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT.relative_to(SUPERVISED_DIR.parent)}")


if __name__ == "__main__":
    main()
