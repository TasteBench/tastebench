"""FartDataset: tuple-yielding SMILES → label wrapper with class weights."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

LABEL_ORDER: Tuple[str, ...] = ("sweet", "bitter", "sour", "umami", "undefined")
_LABEL_TO_INT = {name: i for i, name in enumerate(LABEL_ORDER)}


def encode_label(name: str) -> int:
    return _LABEL_TO_INT[name]


def decode_label(i: int) -> str:
    return LABEL_ORDER[i]


def write_label_map(dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps({"order": list(LABEL_ORDER)}, indent=2))


class FartDataset:
    """Iterable of (smiles, label_int) tuples backed by a FartDB CSV."""

    def __init__(
        self,
        csv_path: Path,
        smiles_column: str = "Canonicalized SMILES",
        label_column: str = "Canonicalized Taste",
    ) -> None:
        self._df = pd.read_csv(csv_path)
        self._smiles_col = smiles_column
        self._label_col = label_column

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        row = self._df.iloc[idx]
        return row[self._smiles_col], encode_label(row[self._label_col])

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        for i in range(len(self)):
            yield self[i]

    @property
    def smiles(self) -> list[str]:
        return self._df[self._smiles_col].tolist()

    @property
    def labels(self) -> list[int]:
        return [encode_label(x) for x in self._df[self._label_col]]

    def class_weights(self, strategy: str = "inverse_frequency") -> np.ndarray:
        """Return a length-5 weight array aligned to LABEL_ORDER.

        Strategies
        ----------
        "none"                   : all weights = 1.0
        "inverse_frequency"      : w_c = N / (K * N_c)           (Chemprop-style)
        "sqrt_inverse_frequency" : w_c = sqrt(N / (K * N_c))     (softer, common
                                   balance between 'none' and 'inverse_frequency'
                                   -- e.g. MegaWika / HuggingFace recipes)
        Unseen classes get weight 1.0.
        """
        k = len(LABEL_ORDER)
        if strategy == "none":
            return np.ones(k, dtype=np.float32)
        counts = Counter(self.labels)
        n = sum(counts.values())
        w = np.ones(k, dtype=np.float32)
        if strategy == "inverse_frequency":
            for i in range(k):
                c = counts.get(i, 0)
                if c > 0:
                    w[i] = n / (k * c)
        elif strategy == "sqrt_inverse_frequency":
            for i in range(k):
                c = counts.get(i, 0)
                if c > 0:
                    w[i] = float(np.sqrt(n / (k * c)))
        else:
            raise ValueError(f"unknown class_weights strategy: {strategy!r}")
        return w
