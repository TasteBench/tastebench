"""Murcko-scaffold-based train/val/test split.

Uses RDKit's Bemis-Murcko scaffold. Groups rows by scaffold, sorts groups
large-to-small, then greedily assigns whole scaffolds to train/val/test so
no scaffold appears in more than one split. Standard chem-ML practice.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


def murcko_scaffold(smiles: str) -> str:
    """Canonical Murcko scaffold SMILES. Returns '' on parse failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaf, canonical=True)


def scaffold_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    smiles_column: str = "Canonicalized SMILES",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into (train, val, test) so no Murcko scaffold appears in two splits."""
    test_frac = 1.0 - train_frac - val_frac
    assert 0.0 < test_frac < 1.0, (train_frac, val_frac, test_frac)

    by_scaf: dict[str, list[int]] = defaultdict(list)
    for idx, smi in zip(df.index, df[smiles_column]):
        by_scaf[murcko_scaffold(smi)].append(idx)

    # Sort groups largest first; stabilise order among same-size groups with seeded RNG.
    groups = list(by_scaf.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda g: -len(g))

    n = len(df)
    target_val = int(n * val_frac)
    target_test = int(n * test_frac)

    val_idxs: list[int] = []
    test_idxs: list[int] = []
    train_idxs: list[int] = []
    for g in groups:
        if len(val_idxs) + len(g) <= target_val:
            val_idxs.extend(g)
        elif len(test_idxs) + len(g) <= target_test:
            test_idxs.extend(g)
        else:
            train_idxs.extend(g)

    train = df.loc[train_idxs].reset_index(drop=True)
    val   = df.loc[val_idxs].reset_index(drop=True)
    test  = df.loc[test_idxs].reset_index(drop=True)
    return train, val, test


def run(src_dir: Path, dst_dir: Path, train_frac: float, val_frac: float, seed: int) -> None:
    parts = [pd.read_csv(src_dir / f"fart_{s}.csv") for s in ("train", "val", "test")]
    df = pd.concat(parts, ignore_index=True)
    logger.info("Combined rows: %d", len(df))

    train, val, test = scaffold_split(df, train_frac, val_frac, seed)
    logger.info("Scaffold split sizes: train=%d val=%d test=%d", len(train), len(val), len(test))

    dst_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(dst_dir / "fart_train.csv", index=False)
    val.to_csv(dst_dir / "fart_val.csv", index=False)
    test.to_csv(dst_dir / "fart_test.csv", index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="molecular/data/splits", type=Path)
    ap.add_argument("--dst_dir", default="molecular/data/splits/scaffold", type=Path)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac",   type=float, default=0.15)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()
    run(args.src_dir, args.dst_dir, args.train_frac, args.val_frac, args.seed)


if __name__ == "__main__":
    main()
