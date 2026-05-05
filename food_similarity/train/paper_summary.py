"""Paper-ready results tables for NeurIPS 2026.

Produces LaTeX tables that match the final paper:
  table_results.tex, table_results_recall.tex, table_per_category.tex

Usage:
    cd food_similarity
    python -m train.paper_summary                          # full (10k bootstrap)
    python -m train.paper_summary --n-bootstrap 100        # fast debug
    python -m train.paper_summary --table-only             # skip bootstrap, use cached
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERVISED_DIR))

from data.loocv import get_analog_keys, load_product_features
from evaluation.bootstrap import compute_bca_cis
from evaluation.metrics import compute_all_metrics

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OOF_DIR = SUPERVISED_DIR / "results" / "oof_predictions"
RESULTS_DIR = SUPERVISED_DIR / "results"

METRIC_NAMES = ["pairwise_accuracy", "spearman", "kendall_tau", "recall_at_1", "recall_at_2", "recall_at_3"]

# ---------------------------------------------------------------------------
# Model catalogs — exactly the 12 models in the main table
# ---------------------------------------------------------------------------

# (oof_file, display_name, group, description)
# display_name is the LaTeX-ready name (use -- for en-dash)
MAIN_CATALOG = [
    # Unsupervised
    ("llm_gemini_3_1_pro_preview_ingredients_image.csv",
     "Gemini 3.1 Pro", "Unsupervised",
     "Zero-shot pairwise preference judgments via LLM, aggregated into a "
     "ranking with Copeland win-rate. Input: ingredient list and product image."),
    ("llm_qwen3_5_397b_a17b_ingredients_image.csv",
     "Qwen 3.5 397B", "Unsupervised",
     "Same protocol as Gemini 3.1 Pro."),
    ("dist_pred_cosine_NCI.csv",
     r"MMRF (cosine)", "Unsupervised",
     "Per-modality cosine distance to animal-reference centroid (nutrition, "
     "compound, image), rank-normalized per category, averaged across modalities."),
    ("dist_pred_l2_NCI.csv",
     r"MMRF (L2)", "Unsupervised",
     "Same as cosine variant using Euclidean distance."),
    (None,  # Random — generated inline
     "Random", "Unsupervised",
     "Theoretical expected value of uniformly random predictions."),
    # Supervised — linear
    ("ridge_SNCTI_bench.csv",
     "Ridge", "Supervised -- linear",
     r"L2-regularized linear regression on mean similarity ($\alpha{=}1.0$)."),
    # Supervised — pairwise
    ("bradley_terry_SNCTI_bench.csv",
     "Bradley--Terry", "Supervised -- pairwise",
     r"Logistic regression on within-category feature-pair differences ($C{=}1.0$)."),
    ("hierarchical_bt_SNCTI_bench.csv",
     "Hierarchical BT", "Supervised -- pairwise",
     "Per-category-subset Bradley--Terry with empirical Bayes shrinkage toward "
     "the global coefficient vector (shrinkage strength set to median subset size)."),
    ("kernel_ranksvm_SNCTI_bench.csv",
     "Kernel RankSVM", "Supervised -- pairwise",
     r"RBF-kernel SVM on within-category pair differences ($C{=}1.0$, $\gamma{=}\texttt{scale}$)."),
    # Supervised — nonlinear
    ("lightgbm_reg_SNCTI_bench.csv",
     "LightGBM", "Supervised -- nonlinear",
     "Gradient-boosted trees with pointwise MSE objective (100 trees, 31 leaves)."),
]

# Ensemble models shown in main table (subset of ENSEMBLE_FULL)
MAIN_ENSEMBLE = [
    ("nested_bt_gemini_nnls.csv", "NNLS"),
    ("nested_bt_gemini_rank.csv", "Rank average"),
    ("nested_bt_gemini_mean.csv", "Mean"),
]

# Full ensemble comparison (extended set of meta-learner variants)
ENSEMBLE_FULL = [
    ("nested_bt_gemini_nnls.csv", "NNLS",
     "Non-negative least squares on nested BT and Gemini scores. Two weights, closed-form."),
    ("nested_bt_gemini_ridge.csv", r"Ridge ($\alpha{=}1.0$)",
     r"Ridge regression ($\alpha{=}1.0$) on nested BT and Gemini scores."),
    ("nested_bt_gemini_rank.csv", "Rank average",
     "Equal-weight average of within-category percentile ranks. Parameter-free."),
    ("nested_bt_gemini_linear.csv", "Linear regression",
     "Unregularized OLS on nested BT and Gemini scores."),
    ("nested_bt_gemini_mean.csv", "Mean",
     "Arithmetic mean of BT and Gemini predicted scores. Parameter-free."),
    ("nested_bt_gemini_lgbm20.csv", "LightGBM (20 trees)",
     "LightGBM meta-learner (20 trees, 4 leaves). Included as counter-example: "
     "nonlinear meta-learning does not help in this 2-feature regime."),
]

# Ensemble descriptions for the main descriptions table (matches MAIN_ENSEMBLE)
MAIN_ENSEMBLE_DESCRIPTIONS = {
    "NNLS": "Non-negative least squares on nested BT and Gemini scores. Two weights, closed-form.",
    "Rank average": "Equal-weight average of within-category percentile ranks. Parameter-free.",
    "Mean": "Arithmetic mean of BT and Gemini predicted scores. Parameter-free.",
}

GROUP_ORDER = [
    "Unsupervised",
    "Supervised -- linear",
    "Supervised -- pairwise",
    "Supervised -- nonlinear",
    "Ensemble (BT + Gemini, nested LOOCV)",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_oof(filename: str) -> Optional[pd.DataFrame]:
    path = OOF_DIR / filename
    if not path.exists():
        logger.warning(f"Missing: {filename}")
        return None
    df = pd.read_csv(path).dropna(subset=["predicted_score"])
    return df if len(df) >= 20 else None


def random_theoretical_metrics(product_features: dict) -> Dict[str, float]:
    """Compute theoretical (expected) metrics for a uniformly random baseline.

    E[Pairwise Acc] = 0.5, E[Spearman] = E[Kendall] = 0,
    E[R@k] = mean(min(k, n_c) / n_c) over categories.
    No CIs needed — these are population expectations.
    """
    analog_keys = sorted(get_analog_keys(product_features))
    cats: Dict[str, int] = {}
    for k in analog_keys:
        c = product_features[k]["category"]
        cats[c] = cats.get(c, 0) + 1
    sizes = np.array(list(cats.values()))

    return {
        "pairwise_accuracy": 0.5,
        "spearman": 0.0,
        "kendall_tau": 0.0,
        "recall_at_1": float(np.mean(np.minimum(1, sizes) / sizes)),
        "recall_at_2": float(np.mean(np.minimum(2, sizes) / sizes)),
        "recall_at_3": float(np.mean(np.minimum(3, sizes) / sizes)),
    }


# ---------------------------------------------------------------------------
# Compute all metrics + CIs
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# LaTeX rendering — table_results.tex
# ---------------------------------------------------------------------------

def fmt3(val: float) -> str:
    """Format as .xxx (no leading zero)."""
    if np.isnan(val):
        return "nan"
    if abs(val) < 0.0005:
        return ".000"
    s = f"{val:.3f}"
    if s.startswith("0"):
        return s[1:]  # .654
    if s.startswith("-0"):
        return "$-" + s[2:] + "$"  # $-.013$
    return s


def _render_results_core(rows: List[dict], full: bool) -> str:
    """Core renderer for results tables. full=True includes R@1/R@2/R@3."""
    ncols = 7 if full else 4
    env = "table"
    float_spec = "H" if full else "t"

    lines = []
    lines.append(rf"\begin{{{env}}}[{float_spec}]")
    lines.append(r"\centering")
    if full:
        lines.append(r"\scriptsize")
        lines.append(r"\setlength{\tabcolsep}{2pt}")
    else:
        lines.append(r"\small")

    if full:
        lines.append(
            r"\caption{All ranking metrics with 95\% BCa CIs on 215 NECTAR products (LOOCV). "
            r"Pairwise accuracy, Spearman $\rho_{\mathrm{S}}$, and Kendall $\tau_{\mathrm{K}}$ are "
            r"category-weighted; R@$k$ is macro-averaged across 24 categories. "
            r"BT = Bradley--Terry, HBT = Hierarchical BT, KSVM = Kernel RankSVM. "
            r"\textbf{Bold} = best in column.}")
        lines.append(r"\label{tab:results-full}")
        lines.append(r"\begin{tabular}{@{}lcccccc@{}}")
        lines.append(r"\toprule")
        lines.append(
            r"Model & Pairwise Acc.\ [95\% CI] & $\rho_{\mathrm{S}}$ "
            r"& $\tau_{\mathrm{K}}$ & R@1 & R@2 & R@3 \\")
    else:
        lines.append(
            r"\caption{Ranking performance on 215 NECTAR plant-based products (LOOCV). "
            r"All supervised models use SNCTI features with per-modality PCA (95\% variance) "
            r"and KNN image imputation ($k{=}5$); distance predictors use NCI. "
            r"95\% BCa CIs (10{,}000 resamples). "
            r"\textbf{Bold} = best in column. "
            r"Recall metrics in Table~\ref{tab:results-recall}; "
            r"all metrics with CIs in Table~\ref{tab:results-full}.}")
        lines.append(r"\label{tab:results}")
        lines.append(r"\begin{tabular}{@{}lccc@{}}")
        lines.append(r"\toprule")
        lines.append(
            r"Model & Pw.\ Acc.\ [95\% CI] & $\rho_{\mathrm{S}}$ "
            r"& $\tau_{\mathrm{K}}$ \\")

    lines.append(r"\midrule")

    # Find column-wise bests
    best = {}
    for metric in METRIC_NAMES:
        best[metric] = max(r[metric] for r in rows)

    # Shorten ensemble header for single-column main table
    GROUP_SHORT = {
        "Ensemble (BT + Gemini, nested LOOCV)": "Ensemble (BT + Gemini)",
    }

    current_group = None
    for row in rows:
        if row["group"] != current_group:
            if current_group is not None:
                lines.append(r"\midrule")
            current_group = row["group"]
            display_group = current_group if full else GROUP_SHORT.get(current_group, current_group)
            lines.append(
                rf"\multicolumn{{{ncols}}}{{@{{}}l}}{{\textit{{{display_group}}}}} \\")

        name = row["display_name"]
        pw = row["pairwise_accuracy"]
        lo, hi = row["pairwise_accuracy_ci_lo"], row["pairwise_accuracy_ci_hi"]

        pw_s = fmt3(pw)
        has_ci = not (np.isnan(lo) or np.isnan(hi))
        ci_s = f" [{fmt3(lo)}, {fmt3(hi)}]" if has_ci else ""
        if abs(pw - best["pairwise_accuracy"]) < 1e-6:
            pw_ci = rf"\textbf{{{pw_s}}}{ci_s}"
            name = rf"\textbf{{{name}}}"
        else:
            pw_ci = f"{pw_s}{ci_s}"

        def maybe_bold(val, metric):
            s = fmt3(val)
            if abs(val - best[metric]) < 1e-6:
                return rf"\textbf{{{s}}}"
            return s

        sp = maybe_bold(row["spearman"], "spearman")
        kt = maybe_bold(row["kendall_tau"], "kendall_tau")

        if full:
            r1 = maybe_bold(row["recall_at_1"], "recall_at_1")
            r2 = maybe_bold(row["recall_at_2"], "recall_at_2")
            r3 = maybe_bold(row["recall_at_3"], "recall_at_3")
            lines.append(rf"\quad {name} & {pw_ci} & {sp} & {kt} & {r1} & {r2} & {r3} \\")
        else:
            lines.append(rf"\quad {name} & {pw_ci} & {sp} & {kt} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines)


def render_results_table(rows: List[dict]) -> str:
    """Main body table: single-column, 3 metrics."""
    return _render_results_core(rows, full=False)


def render_results_recall_table(rows: List[dict]) -> str:
    """Companion table for main results: R@1, R@2, R@3 with 95% CIs (single-column)."""
    recall_metrics = ["recall_at_1", "recall_at_2", "recall_at_3"]

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(
        r"\caption{Recall metrics on 215 NECTAR products (LOOCV) with 95\% BCa CIs. "
        r"R@$k$ = fraction of categories where the true best product is ranked "
        r"in the top $k$ by the model (macro-averaged). \textbf{Bold} = best in column.}")
    lines.append(r"\label{tab:results-recall}")
    lines.append(r"\begin{tabular}{@{}lccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Model & R@1 & R@2 & R@3 \\")
    lines.append(r"\midrule")

    # Column-wise bests
    best = {m: max(r[m] for r in rows) for m in recall_metrics}

    current_group = None
    for row in rows:
        if row["group"] != current_group:
            if current_group is not None:
                lines.append(r"\midrule")
            current_group = row["group"]
            display_group = "Ensemble (BT + Gemini)" if "Ensemble" in current_group else current_group
            lines.append(
                rf"\multicolumn{{4}}{{@{{}}l}}{{\textit{{{display_group}}}}} \\")

        name = row["display_name"]

        def fmt_with_ci(val, metric):
            s = fmt3(val)
            lo = row.get(f"{metric}_ci_lo", np.nan)
            hi = row.get(f"{metric}_ci_hi", np.nan)
            if not np.isnan(lo) and not np.isnan(hi):
                s += f" [{fmt3(lo)}, {fmt3(hi)}]"
            if abs(val - best[metric]) < 1e-6:
                return rf"\textbf{{{s}}}"
            return s

        r1 = fmt_with_ci(row["recall_at_1"], "recall_at_1")
        r2 = fmt_with_ci(row["recall_at_2"], "recall_at_2")
        r3 = fmt_with_ci(row["recall_at_3"], "recall_at_3")

        lines.append(rf"\quad {name} & {r1} & {r2} & {r3} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LaTeX rendering — per-category breakdown
# ---------------------------------------------------------------------------

# Top models for per-category breakdown
PER_CATEGORY_MODELS = [
    ("llm_gemini_3_1_pro_preview_ingredients_image.csv", "Gemini"),
    ("bradley_terry_SNCTI_bench.csv", "BT"),
    ("nested_bt_gemini_nnls.csv", "BT+Gemini"),
]


def render_per_category_table() -> str:
    """Render per-category pairwise accuracy for top models."""
    from evaluation.metrics import compute_per_category_metrics

    # Collect per-category metrics
    cats = None
    n_products = None
    model_data = {}  # display -> list of pw_acc per category

    for oof_file, display in PER_CATEGORY_MODELS:
        df = load_oof(oof_file)
        if df is None:
            continue
        pc = compute_per_category_metrics(df).sort_values("category")
        if cats is None:
            cats = pc["category"].tolist()
            n_products = pc["n_products"].tolist()
        model_data[display] = pc["pairwise_accuracy"].tolist()

    if not model_data:
        return ""

    displays = [d for _, d in PER_CATEGORY_MODELS if d in model_data]
    n_models = len(displays)

    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(
        r"\caption{Per-category pairwise accuracy for selected models "
        r"(one per family). \textbf{Bold} = best model per category. "
        r"$n$ = number of products in category.}")
    lines.append(r"\label{tab:per-category}")
    lines.append(r"\begin{tabular}{@{}lr" + "r" * n_models + "@{}}")
    lines.append(r"\toprule")
    lines.append("Category & $n$ & " + " & ".join(displays) + r" \\")
    lines.append(r"\midrule")

    for i, cat in enumerate(cats):
        n = n_products[i]
        vals = [model_data[d][i] for d in displays]
        best_val = max(vals)
        cells = []
        for v in vals:
            s = fmt3(v)
            if abs(v - best_val) < 1e-6:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        # Clean up category name for LaTeX (underscores)
        cat_tex = cat.replace("_", " ")
        lines.append(f"{cat_tex} & {n} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CI caching
# ---------------------------------------------------------------------------

CACHE_CSV = RESULTS_DIR / "paper_summary_extended.csv"


def load_cached_cis() -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Load cached CIs from paper_summary_extended.csv.

    Returns dict mapping display_name -> {metric_name: (ci_lo, ci_hi)}.
    Returns empty dict if cache doesn't exist or has no CIs.
    """
    if not CACHE_CSV.exists():
        return {}
    df = pd.read_csv(CACHE_CSV)
    cache = {}
    for _, row in df.iterrows():
        name = row["display_name"]
        cis = {}
        for metric in METRIC_NAMES:
            lo_col = f"{metric}_ci_lo"
            hi_col = f"{metric}_ci_hi"
            # Also support legacy "pw_ci_lo" / "pw_ci_hi" column names
            if metric == "pairwise_accuracy" and lo_col not in row.index:
                lo_col, hi_col = "pw_ci_lo", "pw_ci_hi"
            lo = row.get(lo_col, np.nan)
            hi = row.get(hi_col, np.nan)
            if not np.isnan(lo) and not np.isnan(hi):
                cis[metric] = (lo, hi)
        if cis:
            cache[name] = cis
    return cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NeurIPS paper tables")
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--table-only", action="store_true",
                        help="Skip bootstrap, use cached CIs from previous run")
    args = parser.parse_args()

    n_bootstrap = args.n_bootstrap
    use_cache = args.table_only
    ci_cache = load_cached_cis() if use_cache else {}
    if use_cache:
        if ci_cache:
            logger.info(f"Loaded cached CIs for {len(ci_cache)} models from {CACHE_CSV}")
        else:
            logger.warning(f"No cached CIs found in {CACHE_CSV} — CIs will be nan. "
                           f"Run without --table-only first to compute them.")

    product_features = load_product_features()
    logger.info(f"Loaded {len(product_features)} products")

    def get_all_cis(display_name: str, df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        """Get CIs for all metrics from cache or compute fresh."""
        if use_cache and display_name in ci_cache:
            return ci_cache[display_name]
        if use_cache:
            logger.warning(f"No cached CIs for '{display_name}' — will be nan")
            return {}
        ci = compute_bca_cis(df, n_bootstrap=n_bootstrap)
        return ci

    def build_row(display_name: str, df: pd.DataFrame, metrics: Dict[str, float],
                  **extra) -> dict:
        """Build a row dict with metrics + CIs for all metrics."""
        cis = get_all_cis(display_name, df)
        row = {"display_name": display_name, "n": len(df), **metrics, **extra}
        for metric in METRIC_NAMES:
            if metric in cis:
                row[f"{metric}_ci_lo"] = cis[metric][0]
                row[f"{metric}_ci_hi"] = cis[metric][1]
            else:
                row[f"{metric}_ci_lo"] = np.nan
                row[f"{metric}_ci_hi"] = np.nan
        return row

    # ---- Build main table rows ----
    main_rows = []
    for spec in MAIN_CATALOG:
        oof_file, display, group, description = spec

        if oof_file is None:  # Random — theoretical expectations, no CIs
            logger.info(f"Computing: {display} (theoretical)")
            m = random_theoretical_metrics(product_features)
            n_analogs = len(get_analog_keys(product_features))
            row = {"display_name": display, "group": group,
                   "description": description, "n": n_analogs, **m}
            # No CIs for theoretical baseline
            for metric in METRIC_NAMES:
                row[f"{metric}_ci_lo"] = np.nan
                row[f"{metric}_ci_hi"] = np.nan
            main_rows.append(row)
            continue

        df = load_oof(oof_file)
        if df is None:
            continue

        logger.info(f"Computing: {display} (n={len(df)})")
        m = compute_all_metrics(df)
        main_rows.append(build_row(display, df, m,
                                   group=group, description=description))

    # Ensemble rows for main table
    for oof_file, display in MAIN_ENSEMBLE:
        df = load_oof(oof_file)
        if df is None:
            continue
        logger.info(f"Computing: {display} (ensemble, n={len(df)})")
        m = compute_all_metrics(df)
        main_rows.append(build_row(display, df, m,
                                   group="Ensemble (BT + Gemini, nested LOOCV)",
                                   description=MAIN_ENSEMBLE_DESCRIPTIONS[display]))

    # ---- Full ensemble rows (extended meta-learner variants) ----
    ensemble_rows = []
    for oof_file, display, description in ENSEMBLE_FULL:
        df = load_oof(oof_file)
        if df is None:
            continue
        logger.info(f"Computing ensemble: {display} (n={len(df)})")
        m = compute_all_metrics(df)
        ensemble_rows.append(build_row(display, df, m, description=description))

    # ---- Render LaTeX ----
    results_tex = render_results_table(main_rows)
    results_recall_tex = render_results_recall_table(main_rows)
    per_cat_tex = render_per_category_table()

    # ---- Save individual .tex files ----
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(exist_ok=True)

    files = {
        "table_results.tex": results_tex,
        "table_results_recall.tex": results_recall_tex,
        "table_per_category.tex": per_cat_tex,
    }
    for fname, content in files.items():
        (tables_dir / fname).write_text(content + "\n")
        logger.info(f"Saved {tables_dir / fname}")

    # Save CSV (this is the CI cache for future --table-only runs)
    pd.DataFrame(main_rows + ensemble_rows).to_csv(CACHE_CSV, index=False)
    logger.info(f"Saved {CACHE_CSV}")

    # Print to stdout
    for content in files.values():
        print(content)
        print()


if __name__ == "__main__":
    main()
