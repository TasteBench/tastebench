"""Tests for scaffold-split generation."""

import pandas as pd
import pytest

from molecular.src.data.scaffold_split import (
    murcko_scaffold,
    scaffold_split,
)


def test_murcko_scaffold_benzoic_acid():
    # Benzoic acid → scaffold is benzene
    smi = "O=C(O)c1ccccc1"
    scaf = murcko_scaffold(smi)
    # Canonical scaffold for benzene
    assert scaf == "c1ccccc1"


def test_murcko_scaffold_invalid_returns_empty():
    assert murcko_scaffold("not-a-smiles") == ""


def test_scaffold_split_disjoint_scaffolds():
    # 3 methylbenzenes (same scaffold c1ccccc1), 3 methylpyridines (c1ccncc1)
    rows = [
        {"Canonicalized SMILES": "Cc1ccccc1",    "Canonicalized Taste": "sweet"},
        {"Canonicalized SMILES": "CCc1ccccc1",   "Canonicalized Taste": "sweet"},
        {"Canonicalized SMILES": "CCCc1ccccc1",  "Canonicalized Taste": "bitter"},
        {"Canonicalized SMILES": "Cc1ccncc1",    "Canonicalized Taste": "bitter"},
        {"Canonicalized SMILES": "CCc1ccncc1",   "Canonicalized Taste": "sour"},
        {"Canonicalized SMILES": "CCCc1ccncc1",  "Canonicalized Taste": "sour"},
    ]
    df = pd.DataFrame(rows)
    train, val, test = scaffold_split(df, train_frac=0.5, val_frac=0.25, seed=42)
    train_scafs = set(murcko_scaffold(s) for s in train["Canonicalized SMILES"])
    val_scafs   = set(murcko_scaffold(s) for s in val["Canonicalized SMILES"])
    test_scafs  = set(murcko_scaffold(s) for s in test["Canonicalized SMILES"])
    # No scaffold appears in more than one split
    assert not (train_scafs & val_scafs)
    assert not (train_scafs & test_scafs)
    assert not (val_scafs  & test_scafs)


def test_scaffold_split_preserves_all_rows():
    rows = [
        {"Canonicalized SMILES": f"C{'C' * i}c1ccccc1", "Canonicalized Taste": "sweet"}
        for i in range(6)
    ]
    rows += [
        {"Canonicalized SMILES": f"C{'C' * i}c1ccncc1", "Canonicalized Taste": "bitter"}
        for i in range(6)
    ]
    df = pd.DataFrame(rows)
    train, val, test = scaffold_split(df, train_frac=0.6, val_frac=0.2, seed=0)
    assert len(train) + len(val) + len(test) == len(df)
