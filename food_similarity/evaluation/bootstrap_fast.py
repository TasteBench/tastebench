"""Vectorized pairwise_accuracy BCa bootstrap, ~50× faster than bootstrap.py.

Outputs match compute_bca_cis()['pairwise_accuracy'] to 1e-6 on the Phase A
anchor OOFs. Used by compute_cis_parallel.py to compute just pw_acc CIs for
large batches of OOFs; if other metrics' CIs are needed, use bootstrap.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _pairwise_acc_vectorized(true_vals: np.ndarray, pred_vals: np.ndarray,
                             cat_idx: np.ndarray) -> float:
    """Category-weighted pairwise accuracy over within-category pairs.

    Matches compute_all_metrics.pairwise_accuracy semantics:
      - iterate (i, j) within each category
      - tied truth → 0.5 credit, count as pair
      - tied pred (but truth non-tied) → 0.5 credit, count as pair
      - otherwise sign-match = 1, mismatch = 0
      - aggregate: (sum correct) / (sum total) across all categories
    """
    total_correct = 0.0
    total_pairs = 0
    for c in np.unique(cat_idx):
        mask = cat_idx == c
        t = true_vals[mask]
        p = pred_vals[mask]
        n = len(t)
        if n < 2:
            continue
        i_idx, j_idx = np.triu_indices(n, k=1)
        true_diff = t[i_idx] - t[j_idx]
        pred_diff = p[i_idx] - p[j_idx]
        # ties in truth → 0.5, ties in pred (non-tied truth) → 0.5
        truth_tied = np.abs(true_diff) < 1e-10
        pred_tied = np.abs(pred_diff) < 1e-10
        # correct = 1 where sign match, 0.5 on any tie
        signs_agree = (true_diff * pred_diff) > 0
        correct = np.where(truth_tied | pred_tied, 0.5,
                           np.where(signs_agree, 1.0, 0.0))
        total_correct += correct.sum()
        total_pairs += n * (n - 1) // 2
    return total_correct / total_pairs if total_pairs else np.nan


def _resample_idx(cat_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Product indices resampled with replacement WITHIN each category."""
    out = np.empty(len(cat_idx), dtype=np.int64)
    for c in np.unique(cat_idx):
        mask = cat_idx == c
        pos = np.where(mask)[0]
        k = len(pos)
        idx = rng.choice(k, size=k, replace=True)
        out[pos] = pos[idx]
    return out


def _recall_at_k_vectorized(true_vals: np.ndarray, pred_vals: np.ndarray,
                             cat_idx: np.ndarray, k: int) -> float:
    """Macro-averaged recall@k matching compute_all_metrics semantics."""
    per_cat = []
    for c in np.unique(cat_idx):
        mask = cat_idx == c
        if not mask.any():
            continue
        t = true_vals[mask]
        p = pred_vals[mask]
        if len(t) == 0:
            continue
        best_idx = np.argmax(t)
        best_pred = p[best_idx]
        n_strictly_better = int((p > best_pred + 1e-10).sum())
        if n_strictly_better >= k:
            per_cat.append(0.0); continue
        n_tied = int((np.abs(p - best_pred) < 1e-10).sum())
        if n_tied == 0:
            per_cat.append(np.nan); continue
        slots = k - n_strictly_better
        per_cat.append(min(slots, n_tied) / n_tied)
    per_cat = [x for x in per_cat if not np.isnan(x)]
    return float(np.mean(per_cat)) if per_cat else np.nan


def _bca_ci(boot: np.ndarray, jack: np.ndarray, observed: float, alpha: float) -> tuple:
    """BCa (bias-corrected accelerated) CI from bootstrap + jackknife samples."""
    z0 = norm.ppf((boot < observed).mean() + 0.5 * (boot == observed).mean())
    jack_mean = jack.mean()
    num = ((jack_mean - jack) ** 3).sum()
    den = 6.0 * (((jack_mean - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0

    def _q(z):
        p = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        return np.quantile(boot, np.clip(p, 0.0, 1.0))

    return _q(norm.ppf(alpha / 2)), _q(norm.ppf(1 - alpha / 2))


def compute_bca_pw_acc(results_df: pd.DataFrame, n_bootstrap: int = 10_000,
                      seed: int = 42, alpha: float = 0.05) -> tuple:
    """Return (point_estimate, ci_lo, ci_hi) for pairwise_accuracy via BCa."""
    cats = results_df["category"].values
    uniq = {c: i for i, c in enumerate(sorted(set(cats)))}
    cat_idx = np.array([uniq[c] for c in cats], dtype=np.int32)
    true = results_df["true_score"].values.astype(np.float64)
    pred = results_df["predicted_score"].values.astype(np.float64)

    observed = _pairwise_acc_vectorized(true, pred, cat_idx)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = _resample_idx(cat_idx, rng)
        boot[b] = _pairwise_acc_vectorized(true[idx], pred[idx], cat_idx[idx])

    # Jackknife values (drop one product at a time)
    n = len(true)
    jack = np.empty(n, dtype=np.float64)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        keep[i] = False
        jack[i] = _pairwise_acc_vectorized(true[keep], pred[keep], cat_idx[keep])
        keep[i] = True

    lo, hi = _bca_ci(boot, jack, observed, alpha)
    return observed, lo, hi


def compute_bca_recall_at_k(results_df: pd.DataFrame, k: int,
                            n_bootstrap: int = 10_000,
                            seed: int = 42, alpha: float = 0.05) -> tuple:
    """Return (point_estimate, ci_lo, ci_hi) for recall@k via BCa."""
    cats = results_df["category"].values
    uniq = {c: i for i, c in enumerate(sorted(set(cats)))}
    cat_idx = np.array([uniq[c] for c in cats], dtype=np.int32)
    true = results_df["true_score"].values.astype(np.float64)
    pred = results_df["predicted_score"].values.astype(np.float64)

    observed = _recall_at_k_vectorized(true, pred, cat_idx, k)

    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = _resample_idx(cat_idx, rng)
        boot[b] = _recall_at_k_vectorized(true[idx], pred[idx], cat_idx[idx], k)

    n = len(true)
    jack = np.empty(n, dtype=np.float64)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        keep[i] = False
        jack[i] = _recall_at_k_vectorized(true[keep], pred[keep], cat_idx[keep], k)
        keep[i] = True

    lo, hi = _bca_ci(boot, jack, observed, alpha)
    return observed, lo, hi


if __name__ == "__main__":
    # Sanity-check against bootstrap.py on Phase A anchor OOFs
    import sys, time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evaluation.bootstrap import compute_bca_cis
    from evaluation.metrics import compute_all_metrics

    OOF = Path(__file__).resolve().parent.parent / "results" / "oof_predictions"
    for name in ["bradley_terry_SNCTI_bench",
                 "nested_bt_gemini_nnls",
                 "llm_gemini_3_1_pro_preview_ingredients_image"]:
        df = pd.read_csv(OOF / f"{name}.csv").dropna(subset=["predicted_score", "true_score"])

        t0 = time.time()
        fast = compute_bca_pw_acc(df, n_bootstrap=10_000, seed=42)
        t_fast = time.time() - t0

        t0 = time.time()
        slow_cis = compute_bca_cis(df, n_bootstrap=10_000, seed=42)["pairwise_accuracy"]
        slow_pt = compute_all_metrics(df)["pairwise_accuracy"]
        t_slow = time.time() - t0

        print(f"{name}")
        print(f"  fast: point={fast[0]:.6f} lo={fast[1]:.6f} hi={fast[2]:.6f}  ({t_fast:.1f}s)")
        print(f"  slow: point={slow_pt:.6f} lo={slow_cis[0]:.6f} hi={slow_cis[1]:.6f}  ({t_slow:.1f}s)")
        print(f"  Δ: {abs(fast[0]-slow_pt):.6f} {abs(fast[1]-slow_cis[0]):.6f} {abs(fast[2]-slow_cis[1]):.6f}")
        print(f"  speedup: {t_slow/t_fast:.1f}x")
