"""Regression test against the double-softmax bug.

Chemprop's MulticlassClassificationFFN.forward already applies softmax,
so predict_proba must NOT apply it a second time. The check: synthesize
non-uniform logits via the FFN, push them through both predict_proba
and the manual ``softmax(predictor.ffn(z))`` reference, and assert they
match. Under the old double-softmax bug they would not match (predict_proba
would apply softmax to already-softmax output).
"""
from __future__ import annotations

import numpy as np
import torch

from chemprop.data import build_dataloader

from molecular.src.models.dmpnn import (
    build_model, featurize_smiles, predict_proba, _safe_batch_size,
)


def _reference_probs(model, smis):
    """Compute softmax(ffn_logits) without going through predict_proba."""
    ds = featurize_smiles(smis)
    dl = build_dataloader(ds, batch_size=_safe_batch_size(len(smis), 8),
                          num_workers=0, shuffle=False)
    model.eval()
    out = []
    with torch.no_grad():
        for batch in dl:
            bmg, *_ = batch
            z = model.fingerprint(bmg)
            logits = model.predictor.ffn(z)  # raw (N, K) logits
            out.append(torch.softmax(logits, dim=-1).numpy())
    return np.concatenate(out, axis=0)


def test_predict_proba_rows_sum_to_one():
    torch.manual_seed(0)
    np.random.seed(0)
    model = build_model(hidden_dim=64, depth=2, dropout=0.0,
                        n_classes=5, init_lr=1e-4)
    smis = ["CCO", "C(=O)O", "c1ccccc1", "CC(=O)O", "CCN"]
    probs = predict_proba(model, smis, batch_size=8, device="cpu")
    assert probs.shape == (5, 5)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0).all()
    assert (probs <= 1).all()


def test_predict_proba_matches_softmax_of_raw_logits():
    """Reproduces the double-softmax bug: predict_proba(model, smis) must
    equal softmax(ffn(z)) (single softmax). Under the old buggy implementation
    predict_proba returned softmax(softmax(ffn(z))), which differs."""
    torch.manual_seed(0)
    np.random.seed(0)
    model = build_model(hidden_dim=64, depth=2, dropout=0.0,
                        n_classes=5, init_lr=1e-4)
    smis = ["CCO", "C(=O)O", "c1ccccc1", "CC(=O)O", "CCN", "CCCN", "CCCO",
            "C1CCCCC1", "C1CCNCC1", "C1=CC=CN=C1"]
    actual = predict_proba(model, smis, batch_size=8, device="cpu")
    expected = _reference_probs(model, smis)
    np.testing.assert_allclose(actual, expected, atol=1e-5)
