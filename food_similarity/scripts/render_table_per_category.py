"""Regenerate table_per_category.tex with 95% BCa CIs.

Per-category pairwise accuracy for Gemini (LLM baseline), BT (SNCTI bench),
and BT+Gemini NNLS ensemble. Each cell shows the point estimate above a
95% BCa CI (10,000 resamples, seed 42) computed on a category-restricted
bootstrap. Categories with very small n (e.g. n=5) produce visibly wide
CIs; reported faithfully without thresholding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERVISED_DIR))

from evaluation.bootstrap_fast import compute_bca_pw_acc
from evaluation.metrics import compute_per_category_metrics

OOF = SUPERVISED_DIR / "results" / "oof_predictions"
OUT = SUPERVISED_DIR.parent / "paper" / "model_results_tables" / "table_per_category.tex"
CACHE = SUPERVISED_DIR / "results" / "cis_per_category.csv"

N_BOOTSTRAP = 10_000
SEED = 42

COLS = [
    ("Gemini",    "llm_gemini_3_1_pro_preview_ingredients_image.csv"),
    ("BT",        "bradley_terry_SNCTI_bench.csv"),
    ("BT+Gemini", "nested_bt_gemini_nnls.csv"),
]


def _short(v: float) -> str:
    if np.isnan(v):
        return "--"
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0") else ("$-" + s[2:] + "$" if s.startswith("-0") else s)


def _bca_for_category(oof_path: Path, category: str) -> tuple:
    df = pd.read_csv(oof_path).dropna(subset=["predicted_score", "true_score"])
    df_cat = df[df["category"] == category]
    if len(df_cat) < 2:
        return (np.nan, np.nan, np.nan)
    return compute_bca_pw_acc(df_cat, n_bootstrap=N_BOOTSTRAP, seed=SEED)


def _overlaps(lo_a: float, hi_a: float,
              lo_b: float, hi_b: float) -> bool:
    """Two CIs overlap iff max(lo_a, lo_b) <= min(hi_a, hi_b)."""
    if any(np.isnan(v) for v in (lo_a, hi_a, lo_b, hi_b)):
        return False
    return max(lo_a, lo_b) <= min(hi_a, hi_b)


def fmt_stack(p: float, lo: float, hi: float,
              bold: bool = False, overlap: bool = False) -> str:
    """Stacked cell: point on top, CI in \\scriptsize below. Lets the
    table fit a single NeurIPS column width at \\footnotesize without
    \\resizebox (which would scale text inconsistently across appendix
    tables). Uses built-in \\shortstack — no extra package.

    ``overlap=True`` appends ``$^\\dag$`` to the point estimate to mark
    cells whose 95% BCa CI overlaps the row leader's CI."""
    if np.isnan(p):
        return "--"
    pt = _short(p)
    if bold:
        pt = r"\textbf{" + pt + "}"
    elif overlap:
        pt = pt + r"$^\dag$"
    if np.isnan(lo) or np.isnan(hi):
        return pt
    return (f"\\shortstack{{{pt} \\\\ "
            f"{{\\scriptsize [{_short(lo)},{_short(hi)}]}}}}")


def cat_display(c: str) -> str:
    return c.replace("_", " ")


def main() -> int:
    per_cat = {}
    n_lookup = {}
    cats_seen = set()
    for label, fname in COLS:
        df = pd.read_csv(OOF / fname).dropna(subset=["predicted_score", "true_score"])
        m = compute_per_category_metrics(df)
        per_cat[label] = {r["category"]: r["pairwise_accuracy"] for _, r in m.iterrows()}
        for _, r in m.iterrows():
            n_lookup[r["category"]] = max(n_lookup.get(r["category"], 0), int(r["n_products"]))
            cats_seen.add(r["category"])

    all_cats = sorted(cats_seen)

    jobs = [(label, fname, c) for label, fname in COLS for c in all_cats]

    # Cache layer: load (model, category) -> (point, ci_lo, ci_hi) if complete.
    cis = None
    if CACHE.exists():
        df_cache = pd.read_csv(CACHE)
        cached = {(r.model, r.category): (float(r.point), float(r.ci_lo), float(r.ci_hi))
                  for r in df_cache.itertuples(index=False)}
        needed = {(label, c) for label, _, c in jobs}
        if needed.issubset(cached.keys()):
            print(f"Loaded {len(cached)} cached BCa CIs from "
                  f"{CACHE.relative_to(SUPERVISED_DIR.parent)}; skipping bootstrap.")
            cis = {(label, c): cached[(label, c)] for label, _, c in jobs}

    if cis is None:
        print(f"Computing BCa CIs for {len(jobs)} per-category cells "
              f"(~3 min; cached for instant re-renders)...", flush=True)
        results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(_bca_for_category)(OOF / fname, c) for _, fname, c in jobs
        )
        cis = {(label, c): r for (label, _, c), r in zip(jobs, results)}
        # Save cache
        rows = [(lbl, cat, p, lo, hi) for (lbl, cat), (p, lo, hi) in sorted(cis.items())]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["model", "category", "point", "ci_lo", "ci_hi"]) \
          .to_csv(CACHE, index=False, float_format="%.6f")
        print(f"Wrote cache → {CACHE.relative_to(SUPERVISED_DIR.parent)}")

    n_cols = 3 + len(COLS)
    header_line = (r"Category & $n$ & Pairs & "
                   + " & ".join(lbl for lbl, _ in COLS)
                   + r" \\")

    lines = [
        r"\begin{center}",
        r"\captionof{table}{Per-category pairwise accuracy (point on top, "
        r"95\% BCa CI from 10{,}000 resamples below; bootstrap restricted "
        r"to within-category products) for one model per family. "
        r"\textbf{Bold} = best model per category; $^\dag$ = CI overlaps "
        r"the row leader (no significant difference at 95\%). $n$ = number "
        r"of products in category; Pairs $= \binom{n}{2}$ within-category "
        r"ranking pairs. Wide CIs on small-$n$ categories reflect the "
        r"small effective sample size and are reported faithfully.}",
        r"\label{tab:per-category}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\begin{tabular}{@{}lrr" + "c" * len(COLS) + r"@{}}",
        r"\toprule",
        header_line,
        r"\midrule",
    ]

    for c in all_cats:
        n = n_lookup.get(c, 0)
        vals = [per_cat[lbl].get(c, np.nan) for lbl, _ in COLS]
        valid = [v for v in vals if not np.isnan(v)]
        max_v = max(valid) if valid else np.nan
        # Leader CI for overlap testing within this category row: the first
        # column whose point matches the row max.
        leader_lo, leader_hi = np.nan, np.nan
        for lbl, _ in COLS:
            v = per_cat[lbl].get(c, np.nan)
            if not np.isnan(v) and not np.isnan(max_v) \
                    and abs(v - max_v) < 1e-9:
                leader_lo, leader_hi = cis.get((lbl, c),
                                               (np.nan, np.nan, np.nan))[1:]
                break
        cells = []
        for lbl, _ in COLS:
            v = per_cat[lbl].get(c, np.nan)
            lo, hi = cis.get((lbl, c), (np.nan, np.nan, np.nan))[1:]
            is_best = (not np.isnan(v) and not np.isnan(max_v)
                       and abs(v - max_v) < 1e-9)
            ovl = (not is_best) and _overlaps(lo, hi, leader_lo, leader_hi)
            cells.append(fmt_stack(v, lo, hi, bold=is_best, overlap=ovl))
        lines.append(f"{cat_display(c)} & {n} & {n*(n-1)//2} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]
    OUT.write_text("\n".join(lines))
    print(f"Wrote {OUT.relative_to(SUPERVISED_DIR.parent)}")

    for lbl, _ in COLS:
        vals = list(per_cat[lbl].values())
        print(f"  {lbl:<14} mean={np.nanmean(vals):.4f}  n={len(vals)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
