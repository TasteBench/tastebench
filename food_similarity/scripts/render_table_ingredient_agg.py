"""Render table_ingredient_agg.tex.

Compares pairwise accuracy of five compound-to-ingredient aggregation
variants for two models: Bradley--Terry alone, and the BT + Gemini
NNLS ensemble. 95% BCa CIs (10,000 resamples, seed 42) on every cell.

The five variants are produced by running
``run_ingredient_agg_ablation.sh`` first, which writes per-variant OOFs:

    results/oof_predictions/bradley_terry_SNCTI_<variant>.csv
    results/oof_predictions/nested_bt_gemini_nnls_<variant>.csv

This script reads those, computes BCa CIs, caches them to
``cis_ingredient_agg.csv`` for instant re-renders, and emits
``paper/model_results_tables/table_ingredient_agg.tex``.

Usage:
    cd food_similarity
    python scripts/render_table_ingredient_agg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

FOOD_SIM_DIR = Path(__file__).resolve().parent.parent
HUMAN_BASELINE_DIR = FOOD_SIM_DIR.parent / "human_baseline"
sys.path.insert(0, str(FOOD_SIM_DIR))
sys.path.insert(0, str(HUMAN_BASELINE_DIR))

from evaluation.bootstrap_fast import compute_bca_pw_acc  # noqa: E402
from scipy.stats import norm  # noqa: E402

OOF      = FOOD_SIM_DIR / "results" / "oof_predictions"
OUT_TEX  = FOOD_SIM_DIR.parent / "paper" / "model_results_tables" / "table_ingredient_agg.tex"
CACHE    = FOOD_SIM_DIR / "results" / "cis_ingredient_agg.csv"

N_BOOT = 10_000
SEED   = 42

# (variant key, display label, formula for per-ingredient embedding z_I)
VARIANTS = [
    ("mean",                  "Uniform mean",     r"$\tfrac{1}{n}\sum_i z_i$"),
    ("max",                   "Element-wise max", r"$\max_i z_i$ (per coord.)"),
    ("top3_by_conc",          "Top-3 by conc.",   r"$\tfrac{1}{3}\sum_{i \in \mathrm{top}_3(c)} z_i$"),
    ("weighted_average",      "Linear conc.",     r"$\sum_i \tfrac{c_i}{\sum_j c_j}\, z_i$"),
    ("log_weighted_average",  "Log conc.",        r"$\sum_i \tfrac{\log(1+c_i)}{\sum_j \log(1+c_j)}\, z_i$"),
]

MODELS = [
    ("bt",   r"BT",                "bradley_terry_SNCTI_{v}.csv"),
    ("nnls", r"BT $+$ Gemini NNLS","nested_bt_gemini_nnls_{v}.csv"),
]


def _short(v: float) -> str:
    s = f"{v:.3f}"
    return s[1:] if s.startswith("0") else s


def _bca_all_pairs(oof_path: Path) -> tuple[float, float, float]:
    """All-pairs pw-acc + 95% BCa CI (point, lo, hi)."""
    df = pd.read_csv(oof_path).dropna(subset=["predicted_score", "true_score"])
    return compute_bca_pw_acc(df, n_bootstrap=N_BOOT, seed=SEED)


def _within_block_per_block_counts(oof_path: Path, blocks) -> np.ndarray:
    """Per-block (n_correct, n_pairs) for within-block pw-acc.

    Returns shape (B, 2). Used as the bootstrap unit for cluster
    bootstrap on blocks below.
    """
    from human_panelist_baseline import _pairwise_accuracy
    df = pd.read_csv(oof_path)
    scores = {(r.category, r.product_code): (r.true_score, r.predicted_score)
              for r in df.itertuples(index=False)}

    rows = []
    for cat, cat_blocks in blocks.items():
        for block in cat_blocks:
            true_vals, pred_vals = [], []
            for p in block.analog_products:
                key = (cat, p)
                if key in scores:
                    t, pr = scores[key]
                    true_vals.append(t)
                    pred_vals.append(pr)
            if len(true_vals) < 2:
                continue
            c, tot = _pairwise_accuracy(np.array(true_vals), np.array(pred_vals))
            rows.append((float(c), int(tot)))
    return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 2))


def _bca_within_block(oof_path: Path, blocks) -> tuple[float, float, float]:
    """Within-block pw-acc + 95% BCa CI via cluster bootstrap on blocks.

    Block is the natural resampling unit: each block contributes
    n*(n-1)/2 pairs, and blocks are independent (different panelist
    subsets, different product subsets).
    """
    counts = _within_block_per_block_counts(oof_path, blocks)
    if counts.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    correct = counts[:, 0]
    total   = counts[:, 1]
    observed = correct.sum() / total.sum()

    rng = np.random.default_rng(SEED)
    B = counts.shape[0]
    boot = np.empty(N_BOOT, dtype=np.float64)
    for b in range(N_BOOT):
        idx = rng.integers(0, B, size=B)
        s = correct[idx].sum()
        t = total[idx].sum()
        boot[b] = s / t if t > 0 else np.nan

    # Jackknife: leave-one-block-out
    total_c, total_p = correct.sum(), total.sum()
    jack = np.empty(B, dtype=np.float64)
    for i in range(B):
        s = total_c - correct[i]
        t = total_p - total[i]
        jack[i] = s / t if t > 0 else np.nan

    z0 = norm.ppf((boot < observed).mean() + 0.5 * (boot == observed).mean())
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0

    def _q(z):
        p = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        return float(np.quantile(boot, np.clip(p, 0.0, 1.0)))

    return float(observed), _q(norm.ppf(0.025)), _q(norm.ppf(0.975))


def _load_cache() -> dict[tuple[str, str, str], tuple[float, float, float]] | None:
    """Cache key: (ingredient_agg, model, scope) where scope ∈ {all_pairs, within_block}.
    All rows now carry full BCa CIs (point, ci_lo, ci_hi)."""
    if not CACHE.exists():
        return None
    df = pd.read_csv(CACHE)
    if "scope" not in df.columns:
        return None
    by_key = {(r.ingredient_agg, r.model, r.scope):
              (float(r.point), float(r.ci_lo), float(r.ci_hi))
              for r in df.itertuples(index=False)}
    expected = {(v, m, s) for v, _, _ in VARIANTS for m, _, _ in MODELS
                for s in ("all_pairs", "within_block")}
    if not expected.issubset(by_key.keys()):
        return None
    # Old caches stored within_block rows as point-only (NaN CIs); force
    # regen so we get the cluster-bootstrap CIs added in this revision.
    for (_, _, scope), (_, lo, hi) in by_key.items():
        if scope == "within_block" and (np.isnan(lo) or np.isnan(hi)):
            return None
    return by_key


def _save_cache(by_key: dict) -> None:
    rows = [(v, m, s, p, lo, hi) for (v, m, s), (p, lo, hi) in sorted(by_key.items())]
    df = pd.DataFrame(rows, columns=["ingredient_agg", "model", "scope",
                                     "point", "ci_lo", "ci_hi"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False, float_format="%.6f")
    print(f"Wrote cache → {CACHE.relative_to(FOOD_SIM_DIR.parent)}")


def main() -> int:
    by_key = _load_cache()
    if by_key is not None:
        print(f"Loaded {len(by_key)} cached entries from "
              f"{CACHE.relative_to(FOOD_SIM_DIR.parent)}; skipping bootstrap.")
    else:
        # Verify all OOFs are present
        todo = []
        for v, _, _ in VARIANTS:
            for m, _, fname in MODELS:
                path = OOF / fname.format(v=v)
                if not path.exists():
                    print(f"Missing {path.relative_to(FOOD_SIM_DIR.parent)}.\n"
                          f"Run scripts/run_ingredient_agg_ablation.sh first.",
                          file=sys.stderr)
                    return 1
                todo.append((v, m, path))

        print(f"Bootstrapping {len(todo)} all-pairs OOFs in parallel "
              f"({N_BOOT} resamples, seed {SEED}). ~3-5 min on 8 cores; "
              f"cached to {CACHE.relative_to(FOOD_SIM_DIR.parent)} for re-renders.")
        results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
            delayed(_bca_all_pairs)(t[2]) for t in todo
        )
        by_key = {(t[0], t[1], "all_pairs"): results[i] for i, t in enumerate(todo)}

        # Within-block BCa CIs via cluster bootstrap on BIBD blocks (the
        # apples-to-apples comparison with panelists). Cheap relative to
        # all-pairs since per-block n is small; run sequentially with a
        # shared blocks structure to avoid re-parsing the sensory CSV.
        print(f"Bootstrapping {len(todo)} within-block OOFs sequentially "
              f"({N_BOOT} block-resamples, seed {SEED}).")
        from human_panelist_baseline import load_sensory_data, identify_blocks
        sensory = load_sensory_data()
        blocks  = identify_blocks(sensory)
        for v, m, path in todo:
            by_key[(v, m, "within_block")] = _bca_within_block(path, blocks)

        _save_cache(by_key)

    # Per-(model, scope) best-row (max point) for bolding. Bold ties too:
    # at .3f rounding, multiple variants can land on the same value.
    best = {}
    leader_ci = {}
    for m, _, _ in MODELS:
        for s in ("all_pairs", "within_block"):
            top = max(by_key[(v, m, s)][0] for v, _, _ in VARIANTS)
            best[(m, s)] = {v for v, _, _ in VARIANTS
                            if abs(by_key[(v, m, s)][0] - top) < 5e-4}
            leader_v = next(v for v, _, _ in VARIANTS
                            if v in best[(m, s)])
            _, lo_l, hi_l = by_key[(leader_v, m, s)]
            leader_ci[(m, s)] = (lo_l, hi_l)

    def _overlaps(lo_a, hi_a, lo_b, hi_b):
        if any(np.isnan(x) for x in (lo_a, hi_a, lo_b, hi_b)):
            return False
        return max(lo_a, lo_b) <= min(hi_a, hi_b)

    def cell(v: str, m: str, scope: str) -> str:
        p, lo, hi = by_key[(v, m, scope)]
        pt = _short(p)
        is_best = v in best[(m, scope)]
        if is_best:
            pt = rf"\textbf{{{pt}}}"
        elif _overlaps(lo, hi, *leader_ci[(m, scope)]):
            pt = pt + r"$^\dag$"
        return f"{pt} [{_short(lo)},{_short(hi)}]"

    n_metric_cols = 2 * len(MODELS)  # all-pairs + within-block per model

    # Multicolumn group header: Aggregation | Definition | BT (2 cols) | NNLS (2 cols)
    group_header_parts = [r"\multicolumn{2}{c}{}"]
    for _, lbl, _ in MODELS:
        group_header_parts.append(rf"\multicolumn{{2}}{{c}}{{{lbl}}}")
    group_header = " & ".join(group_header_parts) + r" \\"

    cmidrules = " ".join(
        rf"\cmidrule(lr){{{3 + 2*i}-{4 + 2*i}}}" for i in range(len(MODELS))
    )

    sub_header = " & ".join(
        ["Aggregation", r"Embedding $z_I$"]
        + ["All-pairs", "Within-block"] * len(MODELS)
    ) + r" \\"

    lines = [
        r"\begin{center}",
        r"\captionof{table}{Compound-to-ingredient aggregation ablation on the "
        r"food-similarity task (LOOCV, $n=215$ NECTAR products, 935 within-category pairs). For "
        r"ingredient $I$ with compounds $i \in I$ at concentrations $c_i$ "
        r"and compound embeddings $z_i$, the per-ingredient embedding "
        r"$z_I$ follows one of the rules below. \emph{All-pairs} covers "
        r"all within-category pairs; \emph{within-block} restricts to the "
        r"BIBD blocks panelists saw (apples-to-apples with humans). 95\% "
        r"BCa CIs in brackets (10{,}000 resamples; cluster bootstrap on "
        r"blocks for within-block). "
        r"\textbf{Bold} = best per (model, scope); $^\dag$ = CI overlaps "
        r"the leader (no significant difference at 95\%).}",
        r"\label{tab:ingredient-agg-ablation}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{0.95}",
        r"\begin{tabular}{@{}ll" + "c" * n_metric_cols + r"@{}}",
        r"\toprule",
        group_header,
        cmidrules,
        sub_header,
        r"\midrule",
    ]

    for v, lbl, defn in VARIANTS:
        cells = [lbl, defn]
        for m, _, _ in MODELS:
            cells.append(cell(v, m, "all_pairs"))
            cells.append(cell(v, m, "within_block"))
        lines.append(" & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.write_text("\n".join(lines))
    print(f"Wrote {OUT_TEX.relative_to(FOOD_SIM_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
