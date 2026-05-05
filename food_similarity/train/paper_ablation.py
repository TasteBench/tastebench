"""Feature ablation table for the food-similarity benchmark.

Generates any missing/stale OOF predictions (n < 215) using proper
KNN image pre-imputation, then builds pairwise accuracy tables:
  1. Combined table: supervised models + distance predictors (15 subsets)
  2. LLM input ablation (7 combinations)

Output: results/table_ablation.tex

Usage:
    cd food_similarity
    python -m train.paper_ablation                 # generate missing + table
    python -m train.paper_ablation --table-only     # table from existing files
"""

import argparse
import logging
import sys
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SUPERVISED_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SUPERVISED_DIR))

from data.loocv import get_analog_keys, load_product_features
from evaluation.bootstrap_fast import compute_bca_pw_acc
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


def fmt3(val: float) -> str:
    """Format as .xxx (no leading zero)."""
    if np.isnan(val):
        return "--"
    s = f"{val:.3f}"
    if s.startswith("0"):
        return s[1:]
    if s.startswith("-0"):
        return "$-" + s[2:] + "$"
    return s


# ---------------------------------------------------------------------------
# Feature subsets (15 = all non-empty subsets of {N, C, T, I})
# ---------------------------------------------------------------------------

NCTI_SUBSETS = []
for r in range(1, 5):
    for c in combinations("NCTI", r):
        NCTI_SUBSETS.append("".join(c))

# S always included for supervised models
SNCTI_SUBSETS = ["S" + s for s in NCTI_SUBSETS]

# ---------------------------------------------------------------------------
# Model definitions — matches the 12-row main table
# ---------------------------------------------------------------------------

# Supervised models for ablation (model_key, display_name for LaTeX)
SUPERVISED_MODELS = [
    ("ridge", "Ridge"),
    ("bradley_terry", "Bradley--Terry"),
    ("hierarchical_bt", "Hierarchical BT"),
    ("kernel_ranksvm", "Kernel RankSVM"),
    ("lightgbm_reg", "LightGBM"),
]

# Distance predictors (short names for ablation table rows)
DIST_PRED_MODELS = [
    ("dist_pred_cosine", "Cosine"),
    ("dist_pred_l2", "L2"),
]

LLM_MODELS = [
    ("llm_gemini_3_1_pro_preview", "Gemini 3.1 Pro"),
    ("llm_qwen3_5_397b_a17b", "Qwen 3.5 397B"),
]

LLM_INPUT_COMBOS = [
    ("ingredients", "Ingr."),
    ("nutrition", "Nutr."),
    ("image", "Img."),
    ("ingredients_nutrition", "Ingr.+Nutr."),
    ("ingredients_image", "Ingr.+Img."),
    ("nutrition_image", "Nutr.+Img."),
    ("ingredients_nutrition_image", "All"),
]

# ---------------------------------------------------------------------------
# File lookup
# ---------------------------------------------------------------------------

MIN_N = 210


def find_oof(model_key: str, feature_code: str) -> tuple:
    """Find a clean OOF file (n >= MIN_N). Returns (path, DataFrame) or (None, None)."""
    for cand in (f"{model_key}_{feature_code}_bench.csv",
                 f"{model_key}_{feature_code}.csv"):
        path = OOF_DIR / cand
        if path.exists():
            df = pd.read_csv(path)
            if len(df) >= MIN_N:
                return path, df
    return None, None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_missing(product_features: dict) -> int:
    """Generate missing/stale supervised ablation OOF predictions."""
    from train.paper_table import impute_missing_images_inplace
    import train.run_loocv as loocv_mod
    from train.run_loocv import run_single

    impute_missing_images_inplace(product_features)

    loocv_mod._PCA_VARIANCE = 0.95
    loocv_mod._SKIP_BOOTSTRAP = True
    loocv_mod._KNN_IMPUTE = 0
    loocv_mod._DIM_REDUCTION = "pca"
    loocv_mod._PAIR_WEIGHTING = None
    loocv_mod._PCA_N_COMPONENTS = None

    to_generate = []
    for model_key, _ in SUPERVISED_MODELS:
        for fc in SNCTI_SUBSETS:
            path, df = find_oof(model_key, fc)
            if path is None:
                to_generate.append((model_key, fc))

    if not to_generate:
        logger.info("All supervised ablation files are up to date")
        return 0

    logger.info(f"Need to generate {len(to_generate)} prediction files")
    n_generated = 0
    for i, (model_key, fc) in enumerate(to_generate):
        logger.info(f"  [{i+1}/{len(to_generate)}] {model_key} / {fc}")
        try:
            run_single(model_key, fc, product_features)
            n_generated += 1
        except Exception as e:
            logger.error(f"  FAILED: {model_key}/{fc}: {e}")

    logger.info(f"Generated {n_generated}/{len(to_generate)} files")
    return n_generated


# ---------------------------------------------------------------------------
# Table building
# ---------------------------------------------------------------------------

def get_pairwise_acc(model_key: str, feature_code: str, is_distpred: bool = False) -> float:
    """Get pairwise accuracy for one model × feature subset."""
    if is_distpred:
        path = OOF_DIR / f"{model_key}_{feature_code}.csv"
        if not path.exists():
            return np.nan
        df = pd.read_csv(path)
    else:
        _, df = find_oof(model_key, feature_code)
        if df is None:
            return np.nan
    return compute_all_metrics(df)["pairwise_accuracy"]


def get_llm_pairwise_acc(model_key: str, combo_key: str) -> float:
    path = OOF_DIR / f"{model_key}_{combo_key}.csv"
    if not path.exists():
        return np.nan
    df = pd.read_csv(path)
    return compute_all_metrics(df)["pairwise_accuracy"]


# ---------------------------------------------------------------------------
# BCa CI computation
# ---------------------------------------------------------------------------

N_BOOTSTRAP = 10_000
SEED = 42


def _bca_for_oof(path) -> tuple:
    if path is None or not path.exists():
        return (np.nan, np.nan, np.nan)
    df = pd.read_csv(path).dropna(subset=["predicted_score", "true_score"])
    return compute_bca_pw_acc(df, n_bootstrap=N_BOOTSTRAP, seed=SEED)


def _resolve_sup_path(model_key: str, feature_code: str):
    for cand in (f"{model_key}_{feature_code}_bench.csv",
                 f"{model_key}_{feature_code}.csv"):
        p = OOF_DIR / cand
        if p.exists():
            return p
    return None


def _resolve_dist_path(model_key: str, feature_code: str):
    p = OOF_DIR / f"{model_key}_{feature_code}.csv"
    return p if p.exists() else None


def compute_features_cis() -> dict:
    """Parallel BCa CIs for the 15 x 7 cells of the features ablation.

    Returns dict: (column_short_name, subset_letter_only) -> (point, lo, hi).
    """
    sup_short = [("ridge", "Ridge"), ("bradley_terry", "BT"),
                 ("hierarchical_bt", "HBT"), ("kernel_ranksvm", "KSVM"),
                 ("lightgbm_reg", "LGBM")]
    dp_short = [("dist_pred_cosine", "Cos"), ("dist_pred_l2", "L2")]

    jobs = []
    for mk, short in sup_short:
        for fc in NCTI_SUBSETS:
            jobs.append((short, fc, _resolve_sup_path(mk, "S" + fc)))
    for mk, short in dp_short:
        for fc in NCTI_SUBSETS:
            jobs.append((short, fc, _resolve_dist_path(mk, fc)))

    logger.info(f"Computing BCa CIs for {len(jobs)} feature-ablation cells...")
    results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
        delayed(_bca_for_oof)(p) for _, _, p in jobs
    )
    return {(short, fc): r for (short, fc, _), r in zip(jobs, results)}


def compute_llm_cis() -> dict:
    """Parallel BCa CIs for the 7 x 2 cells of the LLM ablation."""
    jobs = []
    for mk, display in LLM_MODELS:
        for ck, _ in LLM_INPUT_COMBOS:
            jobs.append((display, ck, OOF_DIR / f"{mk}_{ck}.csv"))

    logger.info(f"Computing BCa CIs for {len(jobs)} LLM-ablation cells...")
    results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
        delayed(_bca_for_oof)(p) for _, _, p in jobs
    )
    return {(d, ck): r for (d, ck, _), r in zip(jobs, results)}


def fmt3_stack(p: float, lo: float, hi: float, bold: bool = False) -> str:
    """Stacked makecell: point estimate above [lo, hi] in scriptsize."""
    if np.isnan(p):
        return "--"
    pt = fmt3(p)
    if bold:
        pt = r"\textbf{" + pt + "}"
    if np.isnan(lo) or np.isnan(hi):
        return pt
    return r"\makecell{" + pt + r" \\ {\scriptsize [" + fmt3(lo) + ", " + fmt3(hi) + r"]}}"


# ---------------------------------------------------------------------------
# LaTeX rendering — combined ablation table
# ---------------------------------------------------------------------------

def render_combined_ablation() -> str:
    """Render features ablation with stacked BCa CIs in a full-width table*."""
    sup_short = [("ridge", "Ridge"), ("bradley_terry", "BT"),
                 ("hierarchical_bt", "HBT"), ("kernel_ranksvm", "KSVM"),
                 ("lightgbm_reg", "LGBM")]
    dp_short = [("dist_pred_cosine", "Cos"), ("dist_pred_l2", "L2")]
    all_models = sup_short + dp_short
    col_names = [s for _, s in all_models]
    n_sup = len(sup_short)
    n_dp = len(dp_short)

    model_vals = {}
    for model_key, short in sup_short:
        model_vals[short] = [get_pairwise_acc(model_key, "S" + fc) for fc in NCTI_SUBSETS]
    for model_key, short in dp_short:
        model_vals[short] = [get_pairwise_acc(model_key, fc, is_distpred=True)
                             for fc in NCTI_SUBSETS]

    cis = compute_features_cis()

    best_per = {}
    for short, vals in model_vals.items():
        valid = [v for v in vals if not np.isnan(v)]
        best_per[short] = max(valid) if valid else np.nan

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Feature ablation: pairwise accuracy with 95\% BCa CIs "
        r"(10{,}000 resamples) for supervised models and unsupervised "
        r"distance predictors. S (category subset) is always included for "
        r"supervised models. N = nutrition, C = compound, T = text, I = image. "
        r"BT = Bradley--Terry, HBT = Hierarchical BT, KSVM = Kernel RankSVM, "
        r"LGBM = LightGBM. \textbf{Bold} = best subset per model. "
        r"Values below .500 indicate worse-than-random.}",
        r"\label{tab:ablation}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1pt}",
        r"\begin{tabular}{@{}l" + "c" * len(all_models) + r"@{}}",
        r"\toprule",
        rf" & \multicolumn{{{n_sup}}}{{c}}{{Supervised (S+subset)}} "
        rf"& \multicolumn{{{n_dp}}}{{c}}{{MMRF}} \\",
        rf"\cmidrule(lr){{2-{1+n_sup}}} \cmidrule(lr){{{2+n_sup}-{1+n_sup+n_dp}}}",
        "Subset & " + " & ".join(col_names) + r" \\",
        r"\midrule",
    ]

    groups = [
        NCTI_SUBSETS[:4],     # singles
        NCTI_SUBSETS[4:10],   # pairs
        NCTI_SUBSETS[10:14],  # triples
        NCTI_SUBSETS[14:],    # full (NCTI)
    ]

    for gi, group in enumerate(groups):
        if gi > 0:
            lines.append(r"\midrule")
        for subset in group:
            idx = NCTI_SUBSETS.index(subset)
            cells = []
            for short in col_names:
                v = model_vals[short][idx]
                lo, hi = cis.get((short, subset), (np.nan, np.nan, np.nan))[1:]
                is_best = (not np.isnan(v) and not np.isnan(best_per[short])
                           and abs(v - best_per[short]) < 1e-6)
                cells.append(fmt3_stack(v, lo, hi, bold=is_best))
            lines.append(f"{subset} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def render_llm_ablation() -> str:
    """Render LLM input ablation with stacked BCa CIs (single-column table[H])."""
    model_vals = {}
    for mk, display in LLM_MODELS:
        model_vals[display] = [
            get_llm_pairwise_acc(mk, ck) for ck, _ in LLM_INPUT_COMBOS
        ]
    col_names = [d.split()[0] for d in model_vals]  # "Gemini", "Qwen"

    cis = compute_llm_cis()

    best_per_model = {}
    for display, vals in model_vals.items():
        valid = [v for v in vals if not np.isnan(v)]
        best_per_model[display] = max(valid) if valid else np.nan

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Input ablation: pairwise accuracy with 95\% BCa CIs "
        r"(10{,}000 resamples) for zero-shot LLMs across modality subsets. "
        r"\textbf{Bold} = best per model.}",
        r"\label{tab:ablation-llm}",
        r"\begin{tabular}{@{}l" + "c" * len(LLM_MODELS) + r"@{}}",
        r"\toprule",
        "Input & " + " & ".join(col_names) + r" \\",
        r"\midrule",
    ]

    for i, (combo_key, combo_display) in enumerate(LLM_INPUT_COMBOS):
        cells = []
        row_label = combo_display
        all_best = True
        for display, vals in model_vals.items():
            v = vals[i]
            lo, hi = cis.get((display, combo_key), (np.nan, np.nan, np.nan))[1:]
            is_best = (not np.isnan(v) and not np.isnan(best_per_model[display])
                       and abs(v - best_per_model[display]) < 1e-6)
            if not is_best:
                all_best = False
            cells.append(fmt3_stack(v, lo, hi, bold=is_best))
        if all_best:
            row_label = r"\textbf{" + row_label + "}"
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Feature ablation table for the food-similarity benchmark")
    parser.add_argument("--table-only", action="store_true",
                        help="Skip generation, build table from existing files")
    args = parser.parse_args()

    if not args.table_only:
        logger.info("Loading product features...")
        product_features = load_product_features()
        logger.info(f"Loaded {len(product_features)} products")
        generate_missing(product_features)

    # Render
    logger.info("Building ablation tables...")
    features_tex = render_combined_ablation()
    llm_tex = render_llm_ablation()

    # Save individual files
    tables_dir = RESULTS_DIR / "tables"
    tables_dir.mkdir(exist_ok=True)

    files = {
        "table_ablation_features.tex": features_tex,
        "table_ablation_llm.tex": llm_tex,
    }
    for fname, content in files.items():
        (tables_dir / fname).write_text(content + "\n")
        logger.info(f"Saved {tables_dir / fname}")

    # Print
    for content in files.values():
        print(content)
        print()


if __name__ == "__main__":
    main()
