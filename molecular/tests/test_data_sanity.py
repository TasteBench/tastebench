"""Tests for data download + sanity-check logic using tiny fixtures."""

from pathlib import Path

import pandas as pd
import pytest

from molecular.src.data.download import (
    compute_class_distribution,
    verify_no_smiles_leakage,
    verify_counts_within_tolerance,
    sha256_file,
)

FIXTURES = Path(__file__).parent / "fixtures" / "tiny_splits"


def test_compute_class_distribution_sums_correctly():
    df = pd.read_csv(FIXTURES / "train.csv")
    dist = compute_class_distribution(df, label_column="Canonicalized Taste")
    assert dist == {"sweet": 1, "bitter": 1}


def test_verify_no_smiles_leakage_passes_on_disjoint_fixtures():
    train = pd.read_csv(FIXTURES / "train.csv")
    val = pd.read_csv(FIXTURES / "val.csv")
    test = pd.read_csv(FIXTURES / "test.csv")
    # Should not raise
    verify_no_smiles_leakage(train, val, test, smiles_column="Canonicalized SMILES")


def test_verify_no_smiles_leakage_fails_on_overlap():
    train = pd.read_csv(FIXTURES / "train.csv")
    # Force overlap by duplicating train into val
    with pytest.raises(ValueError, match="SMILES leakage"):
        verify_no_smiles_leakage(train, train, train, smiles_column="Canonicalized SMILES")


def test_verify_counts_within_tolerance_passes_exact():
    observed = {"sweet": 10, "bitter": 5}
    expected = {"sweet": 10, "bitter": 5}
    verify_counts_within_tolerance(observed, expected, per_class_tol=5)


def test_verify_counts_within_tolerance_fails_large_drift():
    observed = {"sweet": 100, "bitter": 5}
    expected = {"sweet": 10, "bitter": 5}
    with pytest.raises(ValueError, match="class-count drift"):
        verify_counts_within_tolerance(observed, expected, per_class_tol=5)


def test_sha256_of_fixture_is_stable():
    h1 = sha256_file(FIXTURES / "train.csv")
    h2 = sha256_file(FIXTURES / "train.csv")
    assert h1 == h2
    assert len(h1) == 64
