import numpy as np
import pytest

from molecular.src.eval.metrics import (
    compute_metrics,
    expected_calibration_error,
)


def test_perfect_predictions_give_perfect_metrics():
    y_true = np.array([0, 1, 2, 3, 4])
    probs = np.eye(5, dtype=np.float32)
    m = compute_metrics(y_true, probs, n_classes=5)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["weighted_f1"] == 1.0
    assert all(c["f1"] == 1.0 for c in m["per_class"])


def test_all_same_prediction_on_imbalanced():
    y_true = np.array([0]*8 + [1]*2)
    probs = np.zeros((10, 5), dtype=np.float32)
    probs[:, 0] = 1.0
    m = compute_metrics(y_true, probs, n_classes=5)
    assert m["accuracy"] == 0.8
    c1 = m["per_class"][1]
    assert c1["recall"] == 0.0
    assert c1["f1"] == 0.0


def test_ece_with_perfectly_calibrated_binary():
    y_true = np.array([0] * 90 + [1] * 10)
    probs = np.zeros((100, 5), dtype=np.float32)
    probs[:, 0] = 0.9
    probs[:, 1] = 0.1
    ece = expected_calibration_error(y_true, probs, n_bins=10)
    assert ece < 0.05


def test_metrics_includes_roc_auc_per_class():
    y_true = np.array([0, 0, 1, 1, 2])
    probs = np.array(
        [[0.9, 0.05, 0.05, 0.0, 0.0],
         [0.8, 0.1, 0.1, 0.0, 0.0],
         [0.1, 0.8, 0.1, 0.0, 0.0],
         [0.2, 0.7, 0.1, 0.0, 0.0],
         [0.1, 0.1, 0.8, 0.0, 0.0]],
        dtype=np.float32,
    )
    m = compute_metrics(y_true, probs, n_classes=5)
    per_class = m["per_class"]
    assert per_class[0]["roc_auc_ovr"] is not None
    assert per_class[0]["roc_auc_ovr"] >= 0.9
    assert per_class[3]["roc_auc_ovr"] is None
    assert per_class[4]["roc_auc_ovr"] is None
