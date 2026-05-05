"""Tests for FartDataset: label encoding + row iteration."""

from pathlib import Path

import pandas as pd
import pytest

from molecular.src.data.dataset import FartDataset, LABEL_ORDER, encode_label

FIXTURES = Path(__file__).parent / "fixtures" / "tiny_splits"


def test_label_order_is_canonical_and_frozen():
    assert LABEL_ORDER == ("sweet", "bitter", "sour", "umami", "undefined")


def test_encode_label_known():
    assert encode_label("sweet") == 0
    assert encode_label("bitter") == 1
    assert encode_label("sour") == 2
    assert encode_label("umami") == 3
    assert encode_label("undefined") == 4


def test_encode_label_unknown_raises():
    with pytest.raises(KeyError):
        encode_label("salty")


def test_fart_dataset_loads_and_yields_tuples():
    ds = FartDataset(FIXTURES / "train.csv")
    assert len(ds) == 2
    smi, label = ds[0]
    assert isinstance(smi, str)
    assert isinstance(label, int)
    assert 0 <= label < 5


def test_fart_dataset_smiles_column_and_label_column():
    ds = FartDataset(FIXTURES / "train.csv")
    smis = [s for s, _ in ds]
    labels = [l for _, l in ds]
    assert smis == ["OC1=CC=CC=C1", "CC(C)CC(=O)O"]
    assert labels == [encode_label("sweet"), encode_label("bitter")]


def test_fart_dataset_class_weights_inverse_frequency():
    ds = FartDataset(FIXTURES / "train.csv")
    weights = ds.class_weights(strategy="inverse_frequency")
    # 2 rows, 1 of each class → equal weights over present classes; others = 1.0 baseline
    assert weights.shape == (5,)
    assert weights[encode_label("sweet")] > 0
    assert weights[encode_label("bitter")] > 0
