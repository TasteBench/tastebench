"""Fetch fart-lab/fart github split CSVs, verify SHA-256, run sanity checks."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

GITHUB_RAW_FMT = (
    "https://raw.githubusercontent.com/fart-lab/fart/{sha}/dataset/splits/{name}"
)

SPLIT_FILES = ["fart_train.csv", "fart_val.csv", "fart_test.csv"]

EXPECTED_ROW_COUNTS = {
    "fart_train.csv": 10517,
    "fart_val.csv":    2254,
    "fart_test.csv":   2254,
}

EXPECTED_CLASS_DISTRIBUTION = {
    "fart_train.csv": {"sweet": 6612, "bitter": 1241, "sour": 1132, "undefined": 1479, "umami": 53},
    "fart_val.csv":   {"sweet": 1499, "bitter":  221, "sour":  237, "undefined":  294, "umami":  3},
    "fart_test.csv":  {"sweet": 1473, "bitter":  233, "sour":  238, "undefined":  304, "umami":  6},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_class_distribution(df: pd.DataFrame, label_column: str) -> Dict[str, int]:
    return dict(Counter(df[label_column].tolist()))


def verify_counts_within_tolerance(
    observed: Dict[str, int],
    expected: Dict[str, int],
    per_class_tol: int,
) -> None:
    """Raise ValueError if any class count differs from expected by > per_class_tol."""
    missing = set(expected) - set(observed)
    extra = set(observed) - set(expected)
    if missing or extra:
        raise ValueError(f"class set mismatch: missing={missing}, extra={extra}")
    for cls, exp in expected.items():
        obs = observed[cls]
        if abs(obs - exp) > per_class_tol:
            raise ValueError(
                f"class-count drift for {cls!r}: observed={obs}, expected={exp}, "
                f"tolerance={per_class_tol}"
            )


def verify_no_smiles_leakage(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, smiles_column: str
) -> None:
    t, v, te = set(train[smiles_column]), set(val[smiles_column]), set(test[smiles_column])
    overlaps = {
        "train/val":  t & v,
        "train/test": t & te,
        "val/test":   v & te,
    }
    for name, shared in overlaps.items():
        if shared:
            raise ValueError(f"SMILES leakage between {name}: {len(shared)} shared keys")


def download_one(sha: str, name: str, dest: Path) -> None:
    url = GITHUB_RAW_FMT.format(sha=sha, name=name)
    logger.info("Downloading %s", url)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)


def run(sha: str, dest_dir: Path, per_class_tol: int = 5) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_FILES:
        dst = dest_dir / name
        download_one(sha, name, dst)
        sha_hex = sha256_file(dst)
        logger.info("  %s sha256=%s", name, sha_hex)

    # Load and sanity-check
    train = pd.read_csv(dest_dir / "fart_train.csv")
    val   = pd.read_csv(dest_dir / "fart_val.csv")
    test  = pd.read_csv(dest_dir / "fart_test.csv")

    for name, df in [("fart_train.csv", train), ("fart_val.csv", val), ("fart_test.csv", test)]:
        if len(df) != EXPECTED_ROW_COUNTS[name]:
            raise ValueError(
                f"row-count mismatch for {name}: observed={len(df)}, expected={EXPECTED_ROW_COUNTS[name]}"
            )
        dist = compute_class_distribution(df, "Canonicalized Taste")
        verify_counts_within_tolerance(dist, EXPECTED_CLASS_DISTRIBUTION[name], per_class_tol)

    verify_no_smiles_leakage(train, val, test, smiles_column="Canonicalized SMILES")

    logger.info("All sanity checks passed.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", required=True, help="fart-lab/fart git SHA to pin")
    ap.add_argument(
        "--dest_dir",
        default="molecular/data/splits",
        type=Path,
    )
    ap.add_argument("--per_class_tol", type=int, default=5)
    args = ap.parse_args()
    run(args.sha, args.dest_dir, args.per_class_tol)


if __name__ == "__main__":
    main()
