"""Classification metrics: accuracy, per-class P/R/F1/AUC, macro-F1, weighted-F1, ECE,
plus BCa bootstrap CIs over the test set."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.stats import norm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from molecular.src.data.dataset import LABEL_ORDER


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    """Top-label expected calibration error with equal-width bins."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if not in_bin.any():
            continue
        acc_in_bin = correct[in_bin].mean()
        conf_in_bin = conf[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(acc_in_bin - conf_in_bin)
    return float(ece)


def compute_metrics(
    y_true: np.ndarray, probs: np.ndarray, n_classes: int = 5
) -> dict:
    """Return a dict with all reported classifier metrics for multiclass."""
    y_pred = probs.argmax(axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=list(range(n_classes)), zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=list(range(n_classes)), zero_division=0))

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )

    per_class = []
    for i in range(n_classes):
        mask = (y_true == i)
        if mask.any():
            try:
                auc = float(roc_auc_score((y_true == i).astype(int), probs[:, i]))
            except ValueError:
                auc = None
        else:
            auc = None
        per_class.append({
            "class":       LABEL_ORDER[i],
            "precision":   float(p[i]),
            "recall":      float(r[i]),
            "f1":          float(f[i]),
            "support":     int(s[i]),
            "roc_auc_ovr": auc,
        })

    ece = expected_calibration_error(y_true, probs)

    conf_mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p_ in zip(y_true, y_pred):
        conf_mat[t, p_] += 1

    return {
        "accuracy":         acc,
        "macro_f1":         macro_f1,
        "weighted_f1":      weighted_f1,
        "ece":              float(ece),
        "per_class":        per_class,
        "confusion_matrix": conf_mat.tolist(),
        "n_samples":        int(len(y_true)),
    }


# ----------------------------------------------------------------------------
# Bootstrap CIs over the test set
# ----------------------------------------------------------------------------

def _macro_metrics(y_true: np.ndarray, probs: np.ndarray, n_classes: int = 5) -> dict:
    """Point estimates of accuracy + macro {P, R, F1, AUROC} on a single sample.

    AUROC: per-class one-vs-rest AUC, averaged over classes that have at
    least one positive in this sample (degenerate classes are skipped).
    Returns NaN for any metric that is undefined on this sample.
    """
    y_pred = probs.argmax(axis=1)
    acc = float((y_pred == y_true).mean())
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    macro_p, macro_r, macro_f1 = float(p.mean()), float(r.mean()), float(f.mean())

    aucs: list[float] = []
    for i in range(n_classes):
        bin_y = (y_true == i).astype(int)
        if bin_y.sum() == 0 or bin_y.sum() == len(bin_y):
            continue
        try:
            aucs.append(float(roc_auc_score(bin_y, probs[:, i])))
        except ValueError:
            continue
    macro_auroc = float(np.mean(aucs)) if aucs else float("nan")

    return {"accuracy": acc, "precision": macro_p, "recall": macro_r,
            "f1": macro_f1, "auroc": macro_auroc}


def _bca_ci(theta_hat: float, theta_boot: np.ndarray,
            theta_jack: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap CI."""
    boot = theta_boot[~np.isnan(theta_boot)]
    if len(boot) < 10 or np.isnan(theta_hat):
        return float("nan"), float("nan")
    # bias correction z0
    p_lt = float((boot < theta_hat).mean())
    p_lt = min(max(p_lt, 1e-6), 1 - 1e-6)
    z0 = norm.ppf(p_lt)
    # acceleration a from jackknife
    jack = theta_jack[~np.isnan(theta_jack)]
    if len(jack) < 2:
        a = 0.0
    else:
        jbar = jack.mean()
        num = float(((jbar - jack) ** 3).sum())
        den = 6.0 * float(((jbar - jack) ** 2).sum()) ** 1.5
        a = num / den if den > 0 else 0.0
    z_lo, z_hi = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    alpha_lo = float(norm.cdf(z0 + (z0 + z_lo) / max(1 - a * (z0 + z_lo), 1e-9)))
    alpha_hi = float(norm.cdf(z0 + (z0 + z_hi) / max(1 - a * (z0 + z_hi), 1e-9)))
    lo = float(np.quantile(boot, alpha_lo))
    hi = float(np.quantile(boot, alpha_hi))
    return lo, hi


def bootstrap_classification_cis(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    n_classes: int = 5,
) -> dict[str, dict[str, float]]:
    """BCa bootstrap CIs over test-set resamples for accuracy + macro P/R/F1/AUROC.

    Returns a dict keyed by metric name: {metric: {point, ci_lo, ci_hi}}.
    Resampling is non-stratified at the molecule level; bootstrap iterations
    where a class has zero positives skip that class's AUROC contribution
    (handled inside ``_macro_metrics``). 10,000 resamples + 2,254-fold
    jackknife matches the BCa configuration used by Tables 2 / 3.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    n = len(y_true)
    point = _macro_metrics(y_true, probs, n_classes)

    rng = np.random.default_rng(seed)
    metric_names = ("accuracy", "precision", "recall", "f1", "auroc")
    boot = {m: np.empty(n_bootstrap, dtype=np.float64) for m in metric_names}
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = _macro_metrics(y_true[idx], probs[idx], n_classes)
        for k in metric_names:
            boot[k][b] = m[k]

    jack = {m: np.empty(n, dtype=np.float64) for m in metric_names}
    full_idx = np.arange(n)
    for j in range(n):
        idx = np.delete(full_idx, j)
        m = _macro_metrics(y_true[idx], probs[idx], n_classes)
        for k in metric_names:
            jack[k][j] = m[k]

    out: dict[str, dict[str, float]] = {}
    for k in metric_names:
        lo, hi = _bca_ci(point[k], boot[k], jack[k])
        out[k] = {"point": float(point[k]), "ci_lo": lo, "ci_hi": hi}
    return out


def _per_class_metrics(y_true: np.ndarray, probs: np.ndarray, n_classes: int = 5) -> list[dict]:
    """Per-class precision / recall / F1 / AUROC on a single sample.

    Class entries with zero positives in this sample carry NaN for AUROC
    (and for P/R/F1, sklearn returns 0 under zero_division=0; we surface
    NaN instead so the BCa percentile-on-NaN handling kicks in cleanly).
    """
    y_pred = probs.argmax(axis=1)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    out: list[dict] = []
    for i in range(n_classes):
        bin_y = (y_true == i).astype(int)
        positives = int(bin_y.sum())
        if positives == 0:
            out.append({"precision": float("nan"), "recall": float("nan"),
                        "f1": float("nan"), "auroc": float("nan"),
                        "support": 0})
            continue
        try:
            auroc = float(roc_auc_score(bin_y, probs[:, i])) if 0 < positives < len(bin_y) else float("nan")
        except ValueError:
            auroc = float("nan")
        out.append({
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
            "auroc": auroc, "support": positives,
        })
    return out


def bootstrap_per_class_cis(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    n_classes: int = 5,
) -> list[dict[str, dict[str, float]]]:
    """BCa bootstrap CIs over test-set resamples for per-class P/R/F1/AUROC.

    Returns a length-n_classes list; each entry maps metric name -> dict
    with point/ci_lo/ci_hi/support. Non-stratified resampling at the
    molecule level (consistent with macro-CI bootstrap); resamples where
    a class has 0 positives contribute NaN to that class's metrics, which
    are filtered before the BCa quantiles -- so the rare-class CIs are
    honest about the rare-zero-positive case rather than artificially
    stabilised by stratification.
    """
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    n = len(y_true)
    point = _per_class_metrics(y_true, probs, n_classes)

    rng = np.random.default_rng(seed)
    metric_names = ("precision", "recall", "f1", "auroc")
    boot = {(c, m): np.empty(n_bootstrap, dtype=np.float64)
            for c in range(n_classes) for m in metric_names}
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        per_class = _per_class_metrics(y_true[idx], probs[idx], n_classes)
        for c in range(n_classes):
            for m in metric_names:
                boot[(c, m)][b] = per_class[c][m]

    jack = {(c, m): np.empty(n, dtype=np.float64)
            for c in range(n_classes) for m in metric_names}
    full_idx = np.arange(n)
    for j in range(n):
        idx = np.delete(full_idx, j)
        per_class = _per_class_metrics(y_true[idx], probs[idx], n_classes)
        for c in range(n_classes):
            for m in metric_names:
                jack[(c, m)][j] = per_class[c][m]

    out: list[dict[str, dict[str, float]]] = []
    for c in range(n_classes):
        per: dict[str, dict[str, float]] = {"support": {"point": float(point[c]["support"])}}
        for m in metric_names:
            lo, hi = _bca_ci(point[c][m], boot[(c, m)], jack[(c, m)])
            per[m] = {"point": float(point[c][m]), "ci_lo": lo, "ci_hi": hi}
        out.append(per)
    return out


def mcnemar_accuracy(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """McNemar's test on paired correct/incorrect outcomes for two models.

    Certifies the FART vs taste_gnn accuracy gap.
    Returns the discordant counts plus the chi-square statistic with Yates'
    continuity correction and its p-value (df=1). For ``b + c >= 25`` the
    chi-square approximation is reliable; otherwise the function still
    returns the statistic but a binomial exact test would be preferable.
    """
    from scipy.stats import chi2

    correct_a = np.asarray(correct_a, dtype=bool)
    correct_b = np.asarray(correct_b, dtype=bool)
    if correct_a.shape != correct_b.shape:
        raise ValueError("paired arrays must be same length")
    b = int((correct_a & ~correct_b).sum())   # A correct, B wrong
    c = int((~correct_a & correct_b).sum())   # A wrong, B correct
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "n_discordant": 0, "chi2": 0.0, "p_value": 1.0}
    chi2_stat = (abs(b - c) - 1) ** 2 / n_disc
    p_value = float(1 - chi2.cdf(chi2_stat, df=1))
    return {"b": b, "c": c, "n_discordant": n_disc,
            "chi2": float(chi2_stat), "p_value": p_value}
