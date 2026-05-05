"""Human panelist baseline analysis for NeurIPS 2026.

Computes individual and group-level pairwise ranking accuracy against panel
consensus, plus inter-rater reliability metrics.

Self-contained: no imports from food_similarity/.

Usage:
    cd human_baseline
    python human_panelist_baseline.py
"""

import json
import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import krippendorff
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

# --- Paths ---
BASELINES_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = BASELINES_DIR.parent
DATA_DIR = NEURIPS_DIR / "data"
SHARED_DIR = NEURIPS_DIR / "shared"
RESULTS_DIR = BASELINES_DIR / "results"
PAPER_DIR = NEURIPS_DIR / "paper" / "human_baseline"
MODEL_OOF_CSV = NEURIPS_DIR / "food_similarity" / "results" / "oof_predictions" / "nested_bt_gemini_nnls.csv"

SENSORY_CSV = DATA_DIR / "consolidated_datasets" / "nectar_consolidated_sensory_rating.csv"
PRODUCT_LABELS_CSV = SHARED_DIR / "data" / "nectar_product_labels.csv"

DROP_CATEGORIES = {"Cold_Unbreaded_Chicken_Breast", "Tenders"}

MEAT_CATS = {
    "Bacon", "Bratwurst", "Breakfast_Sausages", "Burgers", "Chicken_Strips",
    "Breaded_Chicken_Filet", "Unbreaded_Chicken_Breast", "Deli_Ham", "Deli_Turkey",
    "Hot_Dogs", "Meatballs", "Nuggets", "Pulled_Pork", "Steak",
}
DAIRY_CATS = {
    "Butter", "Cream_Cheese", "Sour_Cream", "Creamer", "Milk",
    "Barista_Milk", "Cheddar_Cheese", "Mozzarella", "Ice_Cream_Hard_Serve", "Yogurt",
}

DEFAULT_K_VALUES = list(range(1, 76))
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Block:
    """A group of panelists who rated the exact same set of products."""
    category: str
    block_id: int
    all_products: List[int]       # All products in block (incl. reference)
    analog_products: List[int]    # Plant-based analogs only
    respondents: List[int]        # Panelist IDs


def load_sensory_data() -> pd.DataFrame:
    """Load and filter sensory rating data.

    Filters: non-null similarity, year in [2025, 2026], drops excluded categories.
    Joins with product labels to mark references and analogs.

    Returns:
        DataFrame with columns: product_category, respondent, product_code,
        similarity, is_reference, is_analog
    """
    sensory = pd.read_csv(SENSORY_CSV)
    sensory = sensory[sensory["similarity"].notna()]
    sensory = sensory[sensory["year"].isin([2025, 2026])]
    sensory = sensory[~sensory["product_category"].isin(DROP_CATEGORIES)]
    sensory = sensory[["product_category", "respondent", "product_code", "similarity"]]

    labels = pd.read_csv(PRODUCT_LABELS_CSV)
    labels = labels[["category", "product_code", "has_meat", "has_dairy", "is_reference"]]

    sensory = sensory.merge(
        labels,
        left_on=["product_category", "product_code"],
        right_on=["category", "product_code"],
        how="left",
    ).drop(columns=["category"])

    # Analog = not (meat in meat category OR dairy in dairy category)
    sensory["is_analog"] = ~(
        ((sensory["product_category"].isin(MEAT_CATS)) & (sensory["has_meat"] == True))
        | ((sensory["product_category"].isin(DAIRY_CATS)) & (sensory["has_dairy"] == True))
    )

    logger.info(
        f"Loaded {len(sensory)} ratings, "
        f"{sensory['respondent'].nunique()} panelists, "
        f"{sensory['product_category'].nunique()} categories"
    )
    return sensory


def identify_blocks(
    sensory: pd.DataFrame,
    min_block_size: int = 20,
) -> Dict[str, List[Block]]:
    """Identify BIBD blocks from sensory data.

    Groups respondents by the exact set of products they rated within each
    category. Drops blocks with fewer than min_block_size respondents.

    Returns:
        Dict mapping category -> list of Block objects.
    """
    blocks_by_cat: Dict[str, List[Block]] = {}
    total_respondents = 0
    dropped_respondents = 0

    for cat in sorted(sensory["product_category"].unique()):
        cat_data = sensory[sensory["product_category"] == cat]
        # Group respondents by exact product set
        resp_products = cat_data.groupby("respondent")["product_code"].apply(
            lambda x: tuple(sorted(x))
        )
        product_set_counts = resp_products.value_counts()

        cat_blocks = []
        block_id = 0
        for product_set, count in product_set_counts.items():
            total_respondents += count
            if count < min_block_size:
                dropped_respondents += count
                continue

            respondent_ids = resp_products[resp_products == product_set].index.tolist()
            all_prods = list(product_set)

            # Identify which products are analogs
            analog_prods = []
            for p in all_prods:
                row = cat_data[cat_data["product_code"] == p].iloc[0]
                if row["is_analog"]:
                    analog_prods.append(p)

            if len(analog_prods) < 2:
                dropped_respondents += count
                continue

            cat_blocks.append(Block(
                category=cat,
                block_id=block_id,
                all_products=all_prods,
                analog_products=sorted(analog_prods),
                respondents=sorted(respondent_ids),
            ))
            block_id += 1

        if cat_blocks:
            blocks_by_cat[cat] = cat_blocks

    n_blocks = sum(len(bs) for bs in blocks_by_cat.values())
    logger.info(
        f"Identified {n_blocks} major blocks across {len(blocks_by_cat)} categories "
        f"(dropped {dropped_respondents}/{total_respondents} respondents in minor blocks)"
    )
    return blocks_by_cat


def compute_panel_means(sensory: pd.DataFrame) -> Dict[Tuple[str, int], Tuple[float, float, int]]:
    """Compute per-product panel mean similarity, sum, and count.

    Returns:
        Dict mapping (category, product_code) -> (mean, sum, count).
        Sum and count are needed for LOO correction.
    """
    stats = (
        sensory.groupby(["product_category", "product_code"])["similarity"]
        .agg(["mean", "sum", "count"])
    )
    return {
        (cat, code): (row["mean"], row["sum"], int(row["count"]))
        for (cat, code), row in stats.iterrows()
    }


# ---------------------------------------------------------------------------
# Metric functions (self-contained copies from food_similarity/evaluation/metrics.py)
# ---------------------------------------------------------------------------

def _pairwise_accuracy(true: np.ndarray, pred: np.ndarray) -> Tuple[float, int]:
    """Pairwise accuracy for one set of products. Returns (correct, total)."""
    n = len(true)
    correct = 0.0
    total = 0
    for i, j in combinations(range(n), 2):
        true_diff = true[i] - true[j]
        pred_diff = pred[i] - pred[j]
        if abs(true_diff) < 1e-10:
            correct += 0.5
            total += 1
        elif abs(pred_diff) < 1e-10:
            correct += 0.5
            total += 1
        else:
            if true_diff * pred_diff > 0:
                correct += 1.0
            total += 1
    return correct, total


def _spearman(true: np.ndarray, pred: np.ndarray) -> float:
    """Spearman correlation for one set of products."""
    if len(true) < 2:
        return np.nan
    mask = ~(np.isnan(true) | np.isnan(pred))
    t, p = true[mask], pred[mask]
    if len(t) < 2:
        return np.nan
    rho, _ = spearmanr(t, p)
    return rho


def _recall_at_k(true: np.ndarray, pred: np.ndarray, k: int) -> float:
    """Recall@k: is the best product in the top-k predictions?"""
    if len(true) == 0:
        return np.nan
    best_idx = np.argmax(true)
    best_pred = pred[best_idx]
    n_strictly_better = np.sum(pred > best_pred + 1e-10)
    if n_strictly_better >= k:
        return 0.0
    n_tied = np.sum(np.abs(pred - best_pred) < 1e-10)
    slots_available = k - n_strictly_better
    return min(slots_available, n_tied) / n_tied


def compute_metrics(
    true_scores: np.ndarray,
    pred_scores: np.ndarray,
) -> Dict[str, float]:
    """Compute all four metrics for a single set of products.

    Args:
        true_scores: ground truth similarity scores
        pred_scores: predicted similarity scores

    Returns:
        Dict with keys: pairwise_accuracy, spearman, recall_at_1, recall_at_3
    """
    correct, total = _pairwise_accuracy(true_scores, pred_scores)
    return {
        "pairwise_accuracy": correct / total if total > 0 else np.nan,
        "n_correct": correct,
        "n_pairs": total,
        "spearman": _spearman(true_scores, pred_scores),
        "recall_at_1": _recall_at_k(true_scores, pred_scores, 1),
        "recall_at_3": _recall_at_k(true_scores, pred_scores, 3),
    }


def _aggregate_category_weighted(
    results_df: pd.DataFrame,
    include_recall: bool = False,
) -> Dict[str, float]:
    """Aggregate metrics across categories, weighted by number of pairs.

    Args:
        results_df: DataFrame with columns: category, true_score, predicted_score
        include_recall: whether to compute recall@1 and recall@3

    Returns:
        Dict with pairwise_accuracy, spearman, and optionally recall_at_1, recall_at_3
    """
    total_correct = 0.0
    total_pairs = 0
    spearman_num = 0.0
    spearman_den = 0
    recall_1_sum = 0.0
    recall_3_sum = 0.0
    n_cats_recall = 0

    for _, group in results_df.groupby("category"):
        t = group["true_score"].values
        p = group["predicted_score"].values
        c, tot = _pairwise_accuracy(t, p)
        total_correct += c
        total_pairs += tot

        rho = _spearman(t, p)
        if not np.isnan(rho):
            spearman_num += rho * len(t)
            spearman_den += len(t)

        if include_recall:
            r1 = _recall_at_k(t, p, 1)
            r3 = _recall_at_k(t, p, 3)
            if not np.isnan(r1):
                recall_1_sum += r1
                recall_3_sum += r3
                n_cats_recall += 1

    result = {
        "pairwise_accuracy": total_correct / total_pairs if total_pairs > 0 else np.nan,
        "spearman": spearman_num / spearman_den if spearman_den > 0 else np.nan,
    }
    if include_recall:
        result["recall_at_1"] = recall_1_sum / n_cats_recall if n_cats_recall > 0 else np.nan
        result["recall_at_3"] = recall_3_sum / n_cats_recall if n_cats_recall > 0 else np.nan
    return result


def _interpolate_k_star(
    k_medians: pd.Series,
    target_accuracy: float,
) -> float:
    """Find k* by linear interpolation: the k at which median accuracy reaches the target.

    Args:
        k_medians: Series indexed by k with median pairwise accuracy values
        target_accuracy: the accuracy threshold to interpolate to

    Returns:
        Interpolated k*, or NaN if never reached.
    """
    for k_val in sorted(k_medians.index):
        if k_medians[k_val] >= target_accuracy:
            prev_k_vals = [k for k in k_medians.index if k < k_val]
            if prev_k_vals:
                prev_k = max(prev_k_vals)
                prev_acc = k_medians[prev_k]
                curr_acc = k_medians[k_val]
                if curr_acc > prev_acc:
                    frac = (target_accuracy - prev_acc) / (curr_acc - prev_acc)
                    return prev_k + frac * (k_val - prev_k)
                return float(k_val)
            return float(k_val)
    return np.nan


# ---------------------------------------------------------------------------
# Individual Panelist LOO Accuracy
# ---------------------------------------------------------------------------

def compute_individual_accuracy(
    sensory: pd.DataFrame,
    blocks: Dict[str, List[Block]],
    panel_means: Dict[Tuple[str, int], Tuple[float, float, int]],
) -> pd.DataFrame:
    """Compute per-panelist LOO pairwise accuracy.

    For each panelist i in each block:
        prediction = panelist i's similarity ratings for analog products
        ground_truth = full panel mean with LOO correction (excluding panelist i)

    Returns:
        DataFrame with columns: category, block_id, respondent, n_analogs,
        n_pairs, pairwise_accuracy, spearman, recall_at_1, recall_at_3
    """
    rows = []

    for cat, cat_blocks in blocks.items():
        cat_data = sensory[sensory["product_category"] == cat]

        for block in cat_blocks:
            analogs = block.analog_products
            n_analogs = len(analogs)
            if n_analogs < 2:
                continue

            # Build rating lookup: (respondent, product_code) -> similarity
            block_data = cat_data[
                cat_data["respondent"].isin(block.respondents)
                & cat_data["product_code"].isin(analogs)
            ]
            ratings = block_data.set_index(["respondent", "product_code"])["similarity"]

            for resp in block.respondents:
                # Get this panelist's ratings for each analog
                pred_scores = []
                true_scores = []
                valid_products = []

                for p in analogs:
                    key = (resp, p)
                    if key not in ratings.index:
                        continue
                    rating_i = ratings.loc[key]
                    mean_p, sum_p, count_p = panel_means[(cat, p)]

                    # LOO correction: remove this panelist from the mean
                    if count_p > 1:
                        loo_mean = (sum_p - rating_i) / (count_p - 1)
                    else:
                        continue  # Skip if only one panelist rated this product

                    pred_scores.append(rating_i)
                    true_scores.append(loo_mean)
                    valid_products.append(p)

                if len(valid_products) < 2:
                    continue

                true_arr = np.array(true_scores)
                pred_arr = np.array(pred_scores)
                metrics = compute_metrics(true_arr, pred_arr)

                rows.append({
                    "category": cat,
                    "block_id": block.block_id,
                    "respondent": resp,
                    "n_analogs": len(valid_products),
                    "n_pairs": metrics["n_pairs"],
                    "pairwise_accuracy": metrics["pairwise_accuracy"],
                    "spearman": metrics["spearman"],
                    "recall_at_1": metrics["recall_at_1"],
                    "recall_at_3": metrics["recall_at_3"],
                })

    result = pd.DataFrame(rows)
    logger.info(
        f"Individual accuracy: {len(result)} panelist-block observations, "
        f"median pairwise accuracy = {result['pairwise_accuracy'].median():.3f}"
    )
    return result


# ---------------------------------------------------------------------------
# Group-Size k Curve
# ---------------------------------------------------------------------------

def compute_group_size_curve(
    sensory: pd.DataFrame,
    blocks: Dict[str, List[Block]],
    panel_means: Dict[Tuple[str, int], Tuple[float, float, int]],
    k_values: List[int] = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Compute pairwise accuracy as a function of group size k.

    For each k, bootstrap: sample k panelists per block, compute group mean
    as prediction, full panel mean as ground truth. As k approaches N, the
    group mean converges to the panel mean and accuracy approaches 1.0.
    The human agreement upper bound (reported separately) shows the ceiling
    for an external predictor like the model.

    Returns:
        DataFrame with columns: k, iteration, pairwise_accuracy, spearman,
        recall_at_1, recall_at_3
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    # Pre-build rating matrices per block for efficiency
    block_ratings = {}  # block_key -> {respondent -> {product -> rating}}
    for cat, cat_blocks in blocks.items():
        cat_data = sensory[sensory["product_category"] == cat]
        for block in cat_blocks:
            key = (cat, block.block_id)
            block_data = cat_data[
                cat_data["respondent"].isin(block.respondents)
                & cat_data["product_code"].isin(block.analog_products)
            ]
            # Build nested dict
            resp_ratings = {}
            for _, row in block_data.iterrows():
                resp = row["respondent"]
                if resp not in resp_ratings:
                    resp_ratings[resp] = {}
                resp_ratings[resp][row["product_code"]] = row["similarity"]
            block_ratings[key] = resp_ratings

    rows = []

    for k in k_values:
        logger.info(f"Group-size curve: k={k}")
        for iteration in range(n_bootstrap):
            rng = np.random.default_rng(seed * 10000 + k * 100 + iteration)

            # Accumulate category-weighted metrics
            all_true = []
            all_pred = []
            all_cats = []

            for cat, cat_blocks in blocks.items():
                for block in cat_blocks:
                    if len(block.respondents) < k + 1:
                        continue

                    # Sample k respondents
                    sampled = rng.choice(
                        block.respondents, size=k, replace=False
                    ).tolist()
                    key = (cat, block.block_id)
                    resp_ratings = block_ratings[key]

                    for p in block.analog_products:
                        # Group prediction = mean of sampled panelists
                        sampled_ratings = [
                            resp_ratings[r][p]
                            for r in sampled
                            if r in resp_ratings and p in resp_ratings[r]
                        ]
                        if not sampled_ratings:
                            continue
                        group_pred = np.mean(sampled_ratings)

                        # Ground truth = full panel mean (same target the model predicts against)
                        mean_p, _, _ = panel_means[(cat, p)]

                        all_true.append(mean_p)
                        all_pred.append(group_pred)
                        all_cats.append(cat)

            if not all_true:
                continue

            iter_df = pd.DataFrame({
                "category": all_cats,
                "true_score": all_true,
                "predicted_score": all_pred,
            })
            agg = _aggregate_category_weighted(iter_df, include_recall=True)
            rows.append({
                "k": k,
                "iteration": iteration,
                **agg,
            })

    result = pd.DataFrame(rows)
    for k_val in k_values:
        k_data = result[result["k"] == k_val]
        if len(k_data) > 0:
            logger.info(
                f"  k={k_val:>3}: median pairwise_accuracy = "
                f"{k_data['pairwise_accuracy'].median():.3f}"
            )
    return result


# ---------------------------------------------------------------------------
# Inter-Rater Reliability
# ---------------------------------------------------------------------------

def compute_inter_rater_reliability(
    sensory: pd.DataFrame,
    blocks: Dict[str, List[Block]],
) -> pd.DataFrame:
    """Compute Krippendorff's alpha (ordinal) and Kendall's W per block.

    Returns:
        DataFrame with columns: category, block_id, n_respondents, n_products,
        krippendorff_alpha, kendall_w
    """
    rows = []

    for cat, cat_blocks in blocks.items():
        cat_data = sensory[sensory["product_category"] == cat]

        for block in cat_blocks:
            analogs = block.analog_products
            respondents = block.respondents

            # Build rating matrix: (n_respondents, n_analogs)
            block_data = cat_data[
                cat_data["respondent"].isin(respondents)
                & cat_data["product_code"].isin(analogs)
            ]
            pivot = block_data.pivot_table(
                index="respondent",
                columns="product_code",
                values="similarity",
                aggfunc="first",
            )
            # Reindex to ensure consistent column order
            pivot = pivot.reindex(columns=analogs)
            rating_matrix = pivot.values  # (n_respondents, n_analogs)

            # Krippendorff's alpha (ordinal)
            # krippendorff package expects (n_raters, n_units) with NaN for missing
            try:
                alpha = krippendorff.alpha(
                    reliability_data=rating_matrix,
                    level_of_measurement="ordinal",
                )
            except Exception:
                alpha = np.nan

            # Kendall's W (coefficient of concordance)
            n_raters, n_items = rating_matrix.shape
            if n_raters < 2 or n_items < 2:
                w = np.nan
            else:
                # Handle NaN by replacing with row mean for ranking
                masked = np.where(
                    np.isnan(rating_matrix),
                    np.nanmean(rating_matrix, axis=1, keepdims=True),
                    rating_matrix,
                )
                ranks = np.apply_along_axis(rankdata, 1, masked)
                rank_sums = ranks.sum(axis=0)
                mean_rank_sum = rank_sums.mean()
                ss = np.sum((rank_sums - mean_rank_sum) ** 2)
                w = (12 * ss) / (n_raters ** 2 * (n_items ** 3 - n_items))

            rows.append({
                "category": cat,
                "block_id": block.block_id,
                "n_respondents": len(respondents),
                "n_products": len(analogs),
                "krippendorff_alpha": alpha,
                "kendall_w": w,
            })

    result = pd.DataFrame(rows)
    logger.info(
        f"Inter-rater reliability: {len(result)} blocks, "
        f"mean alpha = {result['krippendorff_alpha'].mean():.3f}, "
        f"mean W = {result['kendall_w'].mean():.3f}"
    )
    return result


def compute_split_half_reliability(
    sensory: pd.DataFrame,
    blocks: Dict[str, List[Block]],
    panel_means: Dict[Tuple[str, int], Tuple[float, float, int]],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> Dict[str, float]:
    """Compute split-half ranking reliability.

    Measures how reproducible the panel mean is as a ground truth target.
    Within each block, panelists are randomly split into two equal halves;
    each half's mean rating per product produces a ranking. We then check
    how often the two halves' rankings agree on pairwise comparisons —
    averaging across many random splits and all blocks. A value of 1.0
    would mean two independent halves of the panel produce identical
    rankings. Reports the raw split-half pairwise agreement (no correction).

    Returns:
        Dict with pairwise_accuracy (float) and n_iterations (int).
    """
    rng = np.random.default_rng(seed)

    # Pre-build rating data per block
    block_data_cache = {}
    for cat, cat_blocks in blocks.items():
        cat_data = sensory[sensory["product_category"] == cat]
        for block in cat_blocks:
            key = (cat, block.block_id)
            bd = cat_data[
                cat_data["respondent"].isin(block.respondents)
                & cat_data["product_code"].isin(block.analog_products)
            ]
            pivot = bd.pivot_table(
                index="respondent", columns="product_code",
                values="similarity", aggfunc="first",
            ).reindex(columns=block.analog_products)
            block_data_cache[key] = pivot

    half_accuracies = []

    for iteration in range(n_bootstrap):
        total_correct = 0.0
        total_pairs = 0

        for cat, cat_blocks in blocks.items():
            for block in cat_blocks:
                key = (cat, block.block_id)
                pivot = block_data_cache[key]
                respondents = pivot.index.tolist()
                n = len(respondents)
                if n < 4:
                    continue

                # Split into two halves
                perm = rng.permutation(n)
                half1_idx = perm[: n // 2]
                half2_idx = perm[n // 2:]

                half1_means = pivot.iloc[half1_idx].mean(axis=0).values
                half2_means = pivot.iloc[half2_idx].mean(axis=0).values

                # Remove NaN products
                valid = ~(np.isnan(half1_means) | np.isnan(half2_means))
                if valid.sum() < 2:
                    continue

                c, t = _pairwise_accuracy(half1_means[valid], half2_means[valid])
                total_correct += c
                total_pairs += t

        if total_pairs > 0:
            half_accuracies.append(total_correct / total_pairs)

    median_half = np.median(half_accuracies)

    result = {
        "pairwise_accuracy": float(median_half),
        "n_iterations": len(half_accuracies),
    }
    logger.info(
        f"Split-half ranking reliability: {median_half:.3f}"
    )
    return result


# ---------------------------------------------------------------------------
# Per-Category Model vs. Human Comparison
# ---------------------------------------------------------------------------

def compute_per_category_comparison(
    individual_df: pd.DataFrame,
    model_oof_path: Path = MODEL_OOF_CSV,
) -> pd.DataFrame:
    """Compare model vs. median human accuracy per category.

    Returns:
        DataFrame with columns: category, human_median_pairwise_accuracy,
        model_pairwise_accuracy, human_median_spearman, model_spearman
    """
    model_oof = pd.read_csv(model_oof_path)

    rows = []
    for cat in sorted(individual_df["category"].unique()):
        # Human: median of individual panelist accuracies in this category
        cat_human = individual_df[individual_df["category"] == cat]
        human_median_pa = cat_human["pairwise_accuracy"].median()
        human_median_sp = cat_human["spearman"].median()

        # Model: compute from OOF predictions
        cat_model = model_oof[model_oof["category"] == cat]
        if len(cat_model) < 2:
            continue
        true = cat_model["true_score"].values
        pred = cat_model["predicted_score"].values
        model_metrics = compute_metrics(true, pred)

        rows.append({
            "category": cat,
            "human_median_pairwise_accuracy": human_median_pa,
            "model_pairwise_accuracy": model_metrics["pairwise_accuracy"],
            "human_median_spearman": human_median_sp,
            "model_spearman": model_metrics["spearman"],
            "n_products": len(cat_model),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Bootstrap CIs for the primary metrics
# ---------------------------------------------------------------------------

def compute_bootstrap_cis(
    individual_df: pd.DataFrame,
    group_df: pd.DataFrame,
    blocks: Dict[str, List[Block]],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> Dict:
    """Bootstrap CIs and one-sided bootstrap test for the primary metrics.

    Resamples panelist-block observations with replacement for CIs.
    Tests whether the model's all-pairs accuracy significantly exceeds
    the median individual panelist accuracy via a one-sided bootstrap
    test: the null distribution is the bootstrap distribution of the
    median panelist accuracy under resampling, and the p-value is the
    fraction of bootstrap medians that meet or exceed the model
    accuracy. (This is a bootstrap test, not a permutation test --
    permutation would shuffle model/human labels under exchangeability;
    here we resample observations to estimate a sampling distribution.)

    Returns:
        Dict with median_pairwise_accuracy_ci, k_star_ci,
        model_vs_human_p_value.
    """
    rng = np.random.default_rng(seed + 99999)  # Different seed from other analyses
    n = len(individual_df)

    boot_medians = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_sample = individual_df.iloc[idx]
        boot_medians.append(boot_sample["pairwise_accuracy"].median())

    boot_arr = np.array(boot_medians)

    # Also bootstrap k* from group-size curve
    # Use all-pairs accuracy as the k* target (matches the plot reference line)
    model_oof = pd.read_csv(MODEL_OOF_CSV)
    all_pairs_acc = _aggregate_category_weighted(model_oof)["pairwise_accuracy"]

    k_medians_by_iter = group_df.groupby(["k", "iteration"])["pairwise_accuracy"].first().unstack(level="iteration")
    n_iters = k_medians_by_iter.shape[1]
    boot_k_stars = []
    for _ in range(n_bootstrap):
        iter_idx = rng.choice(n_iters, size=n_iters, replace=True)
        boot_k_med = k_medians_by_iter.iloc[:, iter_idx].median(axis=1)
        boot_k_stars.append(_interpolate_k_star(boot_k_med, all_pairs_acc))

    k_star_arr = np.array([x for x in boot_k_stars if not np.isnan(x)])

    # One-sided bootstrap test: is model accuracy > median individual accuracy?
    # H0: model accuracy <= median individual accuracy
    # Uses all-pairs accuracy. The null distribution is the bootstrap
    # distribution of the median panelist accuracy; the p-value is the
    # fraction of bootstrap medians at or above the model's all-pairs accuracy.
    observed_diff = all_pairs_acc - individual_df["pairwise_accuracy"].median()

    n_exceed = np.sum(boot_arr >= all_pairs_acc)
    p_value = float(n_exceed / len(boot_arr))

    logger.info(
        f"One-sided bootstrap test: model all-pairs ({all_pairs_acc:.4f}) "
        f"vs median individual ({individual_df['pairwise_accuracy'].median():.4f}), "
        f"diff={observed_diff:.4f}, p={p_value:.4f}"
    )

    return {
        "median_pairwise_accuracy_ci": [
            float(np.percentile(boot_arr, 2.5)),
            float(np.percentile(boot_arr, 97.5)),
        ],
        "k_star_ci": [
            float(np.percentile(k_star_arr, 2.5)) if len(k_star_arr) > 0 else None,
            float(np.percentile(k_star_arr, 97.5)) if len(k_star_arr) > 0 else None,
        ],
        "model_vs_human_p_value": p_value,
        "model_all_pairs_accuracy": float(all_pairs_acc),
        "observed_diff": float(observed_diff),
    }


# ---------------------------------------------------------------------------
# Pair Difficulty Analysis
# ---------------------------------------------------------------------------

def compute_pair_difficulty(
    sensory: pd.DataFrame,
    blocks: Dict[str, List[Block]],
    panel_means: Dict[Tuple[str, int], Tuple[float, float, int]],
    model_oof_path: Path = MODEL_OOF_CSV,
) -> pd.DataFrame:
    """Analyze accuracy as a function of pair difficulty.

    Difficulty = |true_score_A - true_score_B| for each within-category pair.
    Larger difference = easier pair.

    Returns:
        DataFrame with columns: difficulty_bin, human_accuracy, model_accuracy, n_pairs
    """
    model_oof = pd.read_csv(model_oof_path)
    model_scores = {
        (row["category"], row["product_code"]): row["predicted_score"]
        for _, row in model_oof.iterrows()
    }

    # Compute per-pair difficulty and accuracy for both human and model
    pair_records = []

    for cat, cat_blocks in blocks.items():
        # Get all analog products across blocks in this category
        cat_analogs = set()
        for b in cat_blocks:
            cat_analogs.update(b.analog_products)
        cat_analogs = sorted(cat_analogs)

        for p1, p2 in combinations(cat_analogs, 2):
            key1 = (cat, p1)
            key2 = (cat, p2)
            if key1 not in panel_means or key2 not in panel_means:
                continue

            true1 = panel_means[key1][0]
            true2 = panel_means[key2][0]
            difficulty = abs(true1 - true2)

            # Model accuracy for this pair
            if key1 in model_scores and key2 in model_scores:
                pred1 = model_scores[key1]
                pred2 = model_scores[key2]
                true_diff = true1 - true2
                pred_diff = pred1 - pred2
                if abs(true_diff) < 1e-10:
                    model_correct = 0.5
                elif abs(pred_diff) < 1e-10:
                    model_correct = 0.5
                else:
                    model_correct = 1.0 if true_diff * pred_diff > 0 else 0.0
            else:
                model_correct = np.nan

            # Human accuracy for this pair: average across all panelists who rated BOTH products
            human_corrects = []
            for b in cat_blocks:
                if p1 not in b.analog_products or p2 not in b.analog_products:
                    continue
                # This block has both products — compute per-panelist accuracy
                cat_data = sensory[
                    (sensory["product_category"] == cat)
                    & sensory["respondent"].isin(b.respondents)
                ]
                for resp in b.respondents:
                    r_data = cat_data[cat_data["respondent"] == resp]
                    r1 = r_data[r_data["product_code"] == p1]["similarity"]
                    r2 = r_data[r_data["product_code"] == p2]["similarity"]
                    if len(r1) == 0 or len(r2) == 0:
                        continue
                    r1_val = r1.values[0]
                    r2_val = r2.values[0]
                    pred_diff_h = r1_val - r2_val
                    true_diff_h = true1 - true2
                    if abs(true_diff_h) < 1e-10:
                        human_corrects.append(0.5)
                    elif abs(pred_diff_h) < 1e-10:
                        human_corrects.append(0.5)
                    else:
                        human_corrects.append(1.0 if true_diff_h * pred_diff_h > 0 else 0.0)

            human_acc = np.mean(human_corrects) if human_corrects else np.nan

            pair_records.append({
                "category": cat,
                "product_code_1": p1,
                "product_code_2": p2,
                "difficulty": difficulty,
                "human_accuracy": human_acc,
                "model_accuracy": model_correct,
                "n_panelists": len(human_corrects),
            })

    pairs_df = pd.DataFrame(pair_records)

    # Bin into quintiles by difficulty
    pairs_df["difficulty_bin"] = pd.qcut(
        pairs_df["difficulty"], q=5, labels=["Very hard", "Hard", "Medium", "Easy", "Very easy"]
    )

    binned = pairs_df.groupby("difficulty_bin", observed=True).agg(
        human_accuracy=("human_accuracy", "mean"),
        model_accuracy=("model_accuracy", "mean"),
        n_pairs=("difficulty", "count"),
        mean_difficulty=("difficulty", "mean"),
    ).reset_index()

    logger.info(f"Pair difficulty analysis:\n{binned.to_string(index=False)}")

    return binned


# ---------------------------------------------------------------------------
# Meat vs. Dairy Meta-Category
# ---------------------------------------------------------------------------

def compute_meta_category_comparison(
    individual_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    model_oof_path: Path = MODEL_OOF_CSV,
) -> pd.DataFrame:
    """Compare meat vs. dairy meta-categories.

    Returns:
        DataFrame with columns: meta_category, human_median_accuracy,
        model_accuracy, mean_alpha, n_categories
    """
    model_oof = pd.read_csv(model_oof_path)

    rows = []
    for meta_name, meta_cats in [("Meat", MEAT_CATS), ("Dairy", DAIRY_CATS)]:
        # Human
        human_meta = individual_df[individual_df["category"].isin(meta_cats)]
        human_acc = human_meta["pairwise_accuracy"].median()

        # Model
        model_meta = model_oof[model_oof["category"].isin(meta_cats)]
        if len(model_meta) >= 2:
            model_acc = _aggregate_category_weighted(model_meta)["pairwise_accuracy"]
        else:
            model_acc = np.nan

        # Reliability
        rel_meta = reliability_df[reliability_df["category"].isin(meta_cats)]
        mean_alpha = rel_meta["krippendorff_alpha"].mean()

        rows.append({
            "meta_category": meta_name,
            "human_median_accuracy": human_acc,
            "model_accuracy": model_acc,
            "mean_alpha": mean_alpha,
            "n_categories": len(human_meta["category"].unique()),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison Table
# ---------------------------------------------------------------------------

def compute_model_within_block_metrics(
    blocks: Dict[str, List[Block]],
    model_oof_path: Path = MODEL_OOF_CSV,
) -> Dict[str, float]:
    """Evaluate model on within-block pairs only (same pairs panelists see).

    For each block, restrict model predictions to the block's analog products
    and compute metrics. Aggregate category-weighted across all blocks.

    Returns:
        Dict with pairwise_accuracy, spearman (category-weighted).
    """
    model_oof = pd.read_csv(model_oof_path)
    model_scores = {
        (row["category"], row["product_code"]): (row["true_score"], row["predicted_score"])
        for _, row in model_oof.iterrows()
    }

    total_correct = 0.0
    total_pairs = 0
    sp_num = 0.0
    sp_den = 0

    for cat, cat_blocks in blocks.items():
        for block in cat_blocks:
            true_vals = []
            pred_vals = []
            for p in block.analog_products:
                key = (cat, p)
                if key in model_scores:
                    t, pr = model_scores[key]
                    true_vals.append(t)
                    pred_vals.append(pr)

            if len(true_vals) < 2:
                continue

            true_arr = np.array(true_vals)
            pred_arr = np.array(pred_vals)

            c, tot = _pairwise_accuracy(true_arr, pred_arr)
            total_correct += c
            total_pairs += tot

            rho = _spearman(true_arr, pred_arr)
            if not np.isnan(rho):
                sp_num += rho * len(true_arr)
                sp_den += len(true_arr)

    return {
        "pairwise_accuracy": total_correct / total_pairs if total_pairs > 0 else np.nan,
        "spearman": sp_num / sp_den if sp_den > 0 else np.nan,
    }


def format_comparison_table(table_df: pd.DataFrame) -> pd.DataFrame:
    """Format comparison table for paper-ready CSV output.

    - Round pairwise_accuracy to 3 decimal places
    - Format CI as "[0.640; 0.725]" or em-dash for missing
    - Format IQR as "0.50–0.75" or em-dash for missing
    - Replace all NaN values with em-dash
    - Drop Spearman column (computed at different granularities across rows,
      not apples-to-apples)

    Returns:
        DataFrame with columns: method, pairwise_accuracy, 95% CI, IQR, eval_set
    """
    EMDASH = "—"
    rows = []
    for _, row in table_df.iterrows():
        pa = f"{row['pairwise_accuracy']:.3f}"
        if pd.notna(row.get('ci_lo')) and pd.notna(row.get('ci_hi')):
            ci = f"[{row['ci_lo']:.3f}; {row['ci_hi']:.3f}]"
        else:
            ci = EMDASH
        if pd.notna(row.get('iqr_lo')) and pd.notna(row.get('iqr_hi')):
            iqr = f"{row['iqr_lo']:.2f}–{row['iqr_hi']:.2f}"
        else:
            iqr = EMDASH
        rows.append({
            "method": row["method"],
            "pairwise_accuracy": pa,
            "95% CI": ci,
            "IQR": iqr,
            "eval_set": row["eval_set"],
        })
    return pd.DataFrame(rows)


def generate_latex_table(
    table_df: pd.DataFrame,
    split_half: Dict[str, float],
    k_star: float,
    n_observations: int,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> str:
    """Generate a LaTeX *fragment* (just the tabular + dagger footnote).

    The paper-side wrapper (\\begin{table}/\\caption/\\label) is intentionally
    omitted so this fragment can either (a) be wrapped by the user in their
    paper as a standalone table, or (b) be \\input{} inside a side-by-side
    minipage with group_size_curve.pdf for a combined human-baseline float
    (see human_baseline/README.md for the snippet).

    The k* / split-half values are no longer baked into the file via a
    \\caption — bake them into your paper-side caption instead. They're
    available in results/summary.json and results/split_half_reliability.json
    if you want to interpolate at compile time.
    """
    def _short(v: float) -> str:
        s = f"{v:.3f}"
        return s[1:] if s.startswith("0") else s

    lines = [
        r"\begin{tabular}{@{}lc@{}}",
        r"\toprule",
        r"Method & Pairwise Acc.\ [95\% CI] \\",
        r"\midrule",
        r"\multicolumn{2}{@{}l}{\textit{Evaluated on within-block pairs}} \\",
    ]

    for _, row in table_df.iterrows():
        if row["eval_set"] == "all within-category":
            continue  # Handle separately below
        if "split-half" in row["method"].lower():
            continue  # Already in caption
        pa = _short(row['pairwise_accuracy'])
        ci_lo, ci_hi = row.get("ci_lo"), row.get("ci_hi")
        iqr_lo, iqr_hi = row.get("iqr_lo"), row.get("iqr_hi")

        if pd.notna(ci_lo) and pd.notna(ci_hi):
            val = f"{pa} [{_short(ci_lo)},{_short(ci_hi)}]"
        elif pd.notna(iqr_lo) and pd.notna(iqr_hi):
            val = f"{pa}" + r"\textsuperscript{\dag}"
        else:
            val = pa

        method = row["method"]
        lines.append(f"\\quad {method} & {val} \\\\")

    # All-pairs model row
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{2}{@{}l}{\textit{Evaluated on all within-category pairs}} \\"
    )
    for _, row in table_df.iterrows():
        if row["eval_set"] != "all within-category":
            continue
        pa = _short(row['pairwise_accuracy'])
        ci_lo, ci_hi = row.get("ci_lo"), row.get("ci_hi")
        if pd.notna(ci_lo) and pd.notna(ci_hi):
            val = f"{pa} [{_short(ci_lo)},{_short(ci_hi)}]"
        else:
            val = pa
        lines.append(f"\\quad {row['method']} & {val} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\vspace{2pt}",
        (
            r"{\scriptsize \textsuperscript{\dag}IQR: "
            f"{table_df.iloc[0]['iqr_lo']:.2f}--{table_df.iloc[0]['iqr_hi']:.2f} "
            f"across {n_observations:,d} panelist-block observations.}}"
        ),
        "",
    ])
    return "\n".join(lines)


def generate_comparison_table(
    individual_df: pd.DataFrame,
    group_df: pd.DataFrame,
    split_half: Dict[str, float],
    bootstrap_cis: Dict,
    blocks: Dict[str, List[Block]],
) -> pd.DataFrame:
    """Generate the summary comparison table for the paper.

    All human-baseline rows and the within-block model row are evaluated on the
    same within-block pairs (~10 pairs per block, 5 analogs).

    Columns: method, pairwise_accuracy, ci_lo, ci_hi, iqr_lo, iqr_hi, eval_set.
    Recall@k is omitted because it is not comparable across different set sizes.

    Returns:
        DataFrame ready for CSV/LaTeX export.
    """
    # Get group-size medians for specific k values
    k_stats = {}
    for k_val in [3, 5]:
        k_data = group_df[group_df["k"] == k_val]
        if len(k_data) > 0:
            k_stats[k_val] = {
                "pairwise_accuracy": k_data["pairwise_accuracy"].median(),
                "pa_ci_lo": np.percentile(k_data["pairwise_accuracy"], 2.5),
                "pa_ci_hi": np.percentile(k_data["pairwise_accuracy"], 97.5),
            }

    # Individual panelist distribution stats for IQR
    pa_q25 = individual_df["pairwise_accuracy"].quantile(0.25)
    pa_q75 = individual_df["pairwise_accuracy"].quantile(0.75)

    rows = []

    # Individual panelist (median + IQR, no CI since bootstrap CI is degenerate)
    rows.append({
        "method": "Individual panelist (median)",
        "pairwise_accuracy": individual_df["pairwise_accuracy"].median(),
        "ci_lo": None, "ci_hi": None,
        "iqr_lo": pa_q25, "iqr_hi": pa_q75,
        "eval_set": "within-block",
    })

    # Panel of k (k=3 and k=5 bracket the model's effective panel size k*)
    for k_val in [3, 5]:
        if k_val in k_stats:
            s = k_stats[k_val]
            rows.append({
                "method": f"Panel of {k_val}",
                "pairwise_accuracy": s["pairwise_accuracy"],
                "ci_lo": s["pa_ci_lo"], "ci_hi": s["pa_ci_hi"],
                "iqr_lo": None, "iqr_hi": None,
                "eval_set": "within-block",
            })

    # Best model — within-block evaluation (apples-to-apples with humans)
    # Bootstrap CI by resampling blocks with replacement
    model_oof = pd.read_csv(MODEL_OOF_CSV)
    model_scores = {
        (row["category"], row["product_code"]): (row["true_score"], row["predicted_score"])
        for _, row in model_oof.iterrows()
    }
    wb_metrics = compute_model_within_block_metrics(blocks)
    all_blocks = [
        (cat, block) for cat, cat_blocks in blocks.items() for block in cat_blocks
    ]
    rng_wb = np.random.default_rng(DEFAULT_SEED + 77777)
    boot_wb = []
    for _ in range(DEFAULT_N_BOOTSTRAP):
        idx = rng_wb.choice(len(all_blocks), size=len(all_blocks), replace=True)
        tc, tt = 0.0, 0
        for i in idx:
            cat, block = all_blocks[i]
            tv, pv = [], []
            for p in block.analog_products:
                key = (cat, p)
                if key in model_scores:
                    t, pr = model_scores[key]
                    tv.append(t)
                    pv.append(pr)
            if len(tv) >= 2:
                c, t = _pairwise_accuracy(np.array(tv), np.array(pv))
                tc += c
                tt += t
        if tt > 0:
            boot_wb.append(tc / tt)
    boot_wb_arr = np.array(boot_wb)

    rows.append({
        "method": "Best model (within-block)",
        "pairwise_accuracy": wb_metrics["pairwise_accuracy"],
        "ci_lo": float(np.percentile(boot_wb_arr, 2.5)),
        "ci_hi": float(np.percentile(boot_wb_arr, 97.5)),
        "iqr_lo": None, "iqr_hi": None,
        "eval_set": "within-block",
    })

    # Best model — all within-category pairs (includes cross-block pairs
    # that no individual panelist judged). CI is the BCa bootstrap on
    # the model OOF, computed inline so it tracks whichever model_oof
    # is canonical at the time.
    all_pairs_metrics = _aggregate_category_weighted(model_oof)
    import sys as _sys
    _sup = NEURIPS_DIR / "food_similarity"
    if str(_sup) not in _sys.path:
        _sys.path.insert(0, str(_sup))
    from evaluation.bootstrap_fast import compute_bca_pw_acc as _bca
    _, _ci_lo, _ci_hi = _bca(model_oof, n_bootstrap=10_000, seed=42)
    rows.append({
        "method": "Best model (all pairs)",
        "pairwise_accuracy": all_pairs_metrics["pairwise_accuracy"],
        "ci_lo": _ci_lo, "ci_hi": _ci_hi,
        "iqr_lo": None, "iqr_hi": None,
        "eval_set": "all within-category",
    })

    # Split-half ranking reliability
    rows.append({
        "method": "Split-half ranking reliability",
        "pairwise_accuracy": split_half["pairwise_accuracy"],
        "ci_lo": None, "ci_hi": None,
        "iqr_lo": None, "iqr_hi": None,
        "eval_set": "within-block",
    })

    return pd.DataFrame(rows)


def save_results(
    individual_df: pd.DataFrame,
    group_df: pd.DataFrame,
    reliability_df: pd.DataFrame,
    split_half: Dict[str, float],
    blocks: Dict[str, List[Block]],
    per_category_df: pd.DataFrame,
    bootstrap_cis: Dict,
    pair_difficulty_df: pd.DataFrame,
    meta_category_df: pd.DataFrame,
) -> None:
    """Save all results to CSV/JSON files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    individual_df.to_csv(RESULTS_DIR / "individual_loo_accuracy.csv", index=False)
    group_df.to_csv(RESULTS_DIR / "group_size_curve.csv", index=False)
    reliability_df.to_csv(RESULTS_DIR / "inter_rater_reliability.csv", index=False)
    per_category_df.to_csv(RESULTS_DIR / "per_category_comparison.csv", index=False)
    pair_difficulty_df.to_csv(RESULTS_DIR / "pair_difficulty.csv", index=False)
    meta_category_df.to_csv(RESULTS_DIR / "meta_category.csv", index=False)

    with open(RESULTS_DIR / "split_half_reliability.json", "w") as f:
        json.dump(split_half, f, indent=2)
    with open(RESULTS_DIR / "bootstrap_cis.json", "w") as f:
        json.dump(bootstrap_cis, f, indent=2)

    # Compute best-model all-pairs accuracy
    model_oof = pd.read_csv(MODEL_OOF_CSV)
    all_pairs_metrics = _aggregate_category_weighted(model_oof)
    all_pairs_acc = float(all_pairs_metrics["pairwise_accuracy"])

    # Compute within-block accuracy (apples-to-apples: model evaluated only
    # on the pair set each panelist rated within a session). Tracks the
    # current MODEL_OOF_CSV instead of being hardcoded.
    wb_metrics = compute_model_within_block_metrics(blocks)

    # Compute effective panel size k* (based on all-pairs accuracy,
    # matching the model reference line shown on the group-size curve)
    k_medians = group_df.groupby("k")["pairwise_accuracy"].median()
    k_star = _interpolate_k_star(k_medians, all_pairs_acc)

    # Model percentile (what fraction of individual panelists does the model exceed?)
    model_exceeds_pct = float(
        (individual_df["pairwise_accuracy"] < all_pairs_acc).mean() * 100
    )

    summary = {
        "individual_panelist": {
            "median_pairwise_accuracy": float(individual_df["pairwise_accuracy"].median()),
            "mean_pairwise_accuracy": float(individual_df["pairwise_accuracy"].mean()),
            "std_pairwise_accuracy": float(individual_df["pairwise_accuracy"].std()),
            "percentiles": {
                str(p): float(individual_df["pairwise_accuracy"].quantile(p / 100))
                for p in [25, 50, 75, 90]
            },
            "model_exceeds_pct": model_exceeds_pct,
            "n_panelists": int(individual_df["respondent"].nunique()),
        },
        "group_size_curve": {
            "effective_panel_size_k_star": float(k_star) if not np.isnan(k_star) else None,
            "k_values_median": {
                str(k): float(v) for k, v in k_medians.items()
            },
        },
        "inter_rater_reliability": {
            "overall_krippendorff_alpha": float(reliability_df["krippendorff_alpha"].mean()),
            "overall_kendall_w": float(reliability_df["kendall_w"].mean()),
        },
        "split_half_reliability": split_half,
        "best_model_pairwise_accuracy": all_pairs_acc,
        # within-block: model evaluated only on the pair set each panelist
        # rated within a session (the "matched" comparison surface). Pulled
        # from wb_metrics (computed earlier in main) so the number tracks
        # the model_oof currently in use; not hardcoded.
        "best_model_within_block_accuracy": float(wb_metrics["pairwise_accuracy"]),
        "best_model_all_pairs_accuracy": all_pairs_acc,
        "n_blocks": sum(len(bs) for bs in blocks.values()),
        "n_categories": len(blocks),
        "seed": DEFAULT_SEED,
        "n_bootstrap": DEFAULT_N_BOOTSTRAP,
    }

    summary["bootstrap_cis"] = bootstrap_cis

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    sensory = load_sensory_data()
    blocks = identify_blocks(sensory)
    panel_means = compute_panel_means(sensory)

    logger.info("=== Individual Panelist LOO Accuracy ===")
    individual_df = compute_individual_accuracy(sensory, blocks, panel_means)

    logger.info("=== Group-Size k Curve ===")
    group_df = compute_group_size_curve(sensory, blocks, panel_means)

    logger.info("=== Inter-Rater Reliability ===")
    reliability_df = compute_inter_rater_reliability(sensory, blocks)

    logger.info("=== Split-Half Ranking Reliability ===")
    split_half = compute_split_half_reliability(sensory, blocks, panel_means)

    logger.info("=== Per-Category Comparison ===")
    per_category_df = compute_per_category_comparison(individual_df)

    logger.info("=== Bootstrap CIs ===")
    bootstrap_cis = compute_bootstrap_cis(individual_df, group_df, blocks)

    logger.info("=== Pair Difficulty ===")
    pair_difficulty_df = compute_pair_difficulty(sensory, blocks, panel_means)

    logger.info("=== Meat vs Dairy ===")
    meta_category_df = compute_meta_category_comparison(individual_df, reliability_df)

    logger.info("=== Saving Results ===")
    save_results(
        individual_df, group_df, reliability_df, split_half, blocks,
        per_category_df, bootstrap_cis, pair_difficulty_df, meta_category_df,
    )

    # Print summary metrics
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path) as f:
        summary = json.load(f)
    pa_q25 = individual_df["pairwise_accuracy"].quantile(0.25)
    pa_q75 = individual_df["pairwise_accuracy"].quantile(0.75)
    print("\n" + "=" * 55)
    print("HUMAN PANELIST BASELINE — NEURIPS RESULTS")
    print("=" * 55)
    print(f"  Median individual panelist:       {summary['individual_panelist']['median_pairwise_accuracy']:.3f} (IQR {pa_q25:.2f}–{pa_q75:.2f})")
    print(f"  Best model (all pairs):           {summary['best_model_all_pairs_accuracy']:.3f}")
    print(f"  Best model (within-block):        {summary['best_model_within_block_accuracy']:.3f}")
    k_star = summary['group_size_curve']['effective_panel_size_k_star']
    if k_star is not None:
        print(f"  Effective panel size k*:          {k_star:.1f}")
    if 'bootstrap_cis' in summary and 'model_vs_human_p_value' in summary['bootstrap_cis']:
        p_val = summary['bootstrap_cis']['model_vs_human_p_value']
        p_str = "< 0.001" if p_val < 0.001 else f"= {p_val:.3f}"
        print(f"  Model vs. median human:           p {p_str}")
    print(f"  Split-half ranking reliability:   {summary['split_half_reliability']['pairwise_accuracy']:.3f}")
    print(f"  Krippendorff's alpha:             {summary['inter_rater_reliability']['overall_krippendorff_alpha']:.3f}")
    print("=" * 55)

    logger.info("=== Comparison Table ===")
    comparison_table_df = generate_comparison_table(individual_df, group_df, split_half, bootstrap_cis, blocks)
    comparison_table_formatted = format_comparison_table(comparison_table_df)
    comparison_table_formatted.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

    # Generate LaTeX table using within-block k* for apples-to-apples comparison
    k_medians = group_df.groupby("k")["pairwise_accuracy"].median()
    wb_acc = summary.get("best_model_within_block_accuracy", summary["best_model_pairwise_accuracy"])
    k_star_wb = _interpolate_k_star(k_medians, wb_acc)
    latex_str = generate_latex_table(
        comparison_table_df, split_half,
        k_star=k_star_wb if not np.isnan(k_star_wb) else 2.0,
        n_observations=len(individual_df),
        n_bootstrap=DEFAULT_N_BOOTSTRAP,
    )
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    with open(PAPER_DIR / "human_baseline_table.tex", "w") as f:
        f.write(latex_str)

    print("\nComparison Table:")
    print(comparison_table_formatted.to_string(index=False))
