"""Generate Kaggle competition CSVs from the FartDB splits.

Mirrors the pattern in kaggle_tastebench/generate_data. The Kaggle
test set is the exact FartDB test split (Zimmermann et al. 2024) so leaderboard
results are directly comparable to the published benchmark.

Run from the repo root:
    python kaggle_molecular_taste/generate_kaggle_datasets.py
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

TOP_DIR = Path(__file__).resolve().parents[1]
SPLITS_DIR = TOP_DIR / "molecular" / "data" / "splits"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "dataset"

VERSION = "1.0.0"  # Pinned to FartDB commit bde90e6562ce5d248e76af791fab29ffc9ae901b.
BASELINE_LABEL = "sweet"
VALID_LABELS = {"sweet", "bitter", "sour", "umami", "undefined"}

# Hashes mirrored from molecular/data/PROVENANCE.md.
SOURCE_HASHES = {
    "fart_train.csv": "35020569fb47d10d8c981d65c712cb73dc7b23512911d35d2b161665584248a3",
    "fart_val.csv":   "b8ee024f5b73324468341475e713d25d340e42aeffcb3a8b64487d8184c2c657",
    "fart_test.csv":  "74096827602864a8aff09234ae68a0637b13f3749b57d662b45fd7863185f4f1",
}
EXPECTED_ROWS = {"train": 10517, "val": 2254, "test": 2254}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source_hashes() -> None:
    """Pin upstream FART splits to their published SHA-256 hashes.

    This is the only integrity boundary that matters: the script is deterministic,
    so verifying the inputs guarantees the outputs are reproducible.
    """
    print("Verifying source FART splits against PROVENANCE hashes...")
    for filename, expected in SOURCE_HASHES.items():
        path = SPLITS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing source split: {path}\n"
                f"Run `python -m molecular.src.data.download` first."
            )
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"SHA-256 mismatch for {filename}\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}\n"
                f"Source FART splits have drifted from the pinned commit "
                f"bde90e6562ce5d248e76af791fab29ffc9ae901b."
            )
        print(f"  {filename}  OK")


def load_split(name: str) -> pd.DataFrame:
    """Load a FartDB split, returning canonicalized_smiles and taste.

    The FartDB-provided index column is dropped because train.csv has been
    reindexed by the upstream authors to 0..N-1 while val.csv and test.csv keep
    the original FartDB dataset indices, causing collisions across splits.
    """
    path = SPLITS_DIR / f"fart_{name}.csv"
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "Canonicalized SMILES": "canonicalized_smiles",
            "Canonicalized Taste": "taste",
        }
    )
    return df[["canonicalized_smiles", "taste"]].reset_index(drop=True)


def write_features(out_dir: Path) -> pd.DataFrame:
    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    # Assign fresh, contiguous, globally-unique Kaggle ids across all three splits.
    train_df.insert(0, "id", range(len(train_df)))
    val_df.insert(0, "id", range(len(train_df), len(train_df) + len(val_df)))
    test_df.insert(0, "id", range(
        len(train_df) + len(val_df),
        len(train_df) + len(val_df) + len(test_df),
    ))

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.drop(columns=["taste"]).to_csv(out_dir / "test.csv", index=False)

    print(f"  train.csv  {len(train_df):>5} rows  (labeled, ids {train_df['id'].min()}..{train_df['id'].max()})")
    print(f"  val.csv    {len(val_df):>5} rows  (labeled, ids {val_df['id'].min()}..{val_df['id'].max()})")
    print(f"  test.csv   {len(test_df):>5} rows  (ids {test_df['id'].min()}..{test_df['id'].max()}, taste withheld)")

    return test_df


def write_solution_and_sample(test_df: pd.DataFrame, out_dir: Path) -> None:
    solution = test_df[["id", "taste"]].copy()
    solution["Usage"] = "Public"
    solution.to_csv(out_dir / "solution.csv", index=False)
    print(f"  solution.csv  {len(solution):>5} rows  (100% Public leaderboard)")

    sample = test_df[["id"]].copy()
    sample["taste"] = BASELINE_LABEL
    sample.to_csv(out_dir / "sample_submission.csv", index=False)
    print(
        f"  sample_submission.csv  {len(sample):>5} rows  "
        f"(baseline: predicts {BASELINE_LABEL!r} for every molecule)"
    )


def verify_outputs(out_dir: Path) -> None:
    print("Verifying generated outputs...")
    train = pd.read_csv(out_dir / "train.csv")
    val = pd.read_csv(out_dir / "val.csv")
    test = pd.read_csv(out_dir / "test.csv")
    sample = pd.read_csv(out_dir / "sample_submission.csv")
    solution = pd.read_csv(out_dir / "solution.csv")

    assert len(train) == EXPECTED_ROWS["train"], (
        f"train.csv has {len(train)} rows, expected {EXPECTED_ROWS['train']}"
    )
    assert len(val) == EXPECTED_ROWS["val"], (
        f"val.csv has {len(val)} rows, expected {EXPECTED_ROWS['val']}"
    )
    assert len(test) == EXPECTED_ROWS["test"], (
        f"test.csv has {len(test)} rows, expected {EXPECTED_ROWS['test']}"
    )
    assert len(solution) == EXPECTED_ROWS["test"]
    assert len(sample) == EXPECTED_ROWS["test"]

    assert "taste" not in test.columns, "LEAK: test.csv contains 'taste' column"
    for name, df in [("train", train), ("val", val), ("test", test)]:
        assert "original_labels" not in df.columns, (
            f"LEAK RISK: {name}.csv contains 'original_labels' (free-text source "
            f"labels that trivially recover the target via keyword matching)"
        )

    test_ids = set(test["id"])
    assert set(sample["id"]) == test_ids, (
        "id mismatch between test.csv and sample_submission.csv"
    )
    assert set(solution["id"]) == test_ids, (
        "id mismatch between test.csv and solution.csv"
    )

    train_ids = set(train["id"])
    val_ids = set(val["id"])
    assert train_ids.isdisjoint(val_ids), "train and val share ids"
    assert train_ids.isdisjoint(test_ids), "train and test share ids"
    assert val_ids.isdisjoint(test_ids), "val and test share ids"

    for name, df in [("train", train), ("val", val), ("solution", solution)]:
        unexpected = set(df["taste"]) - VALID_LABELS
        assert not unexpected, f"unexpected labels in {name}: {unexpected}"

    print(f"  row counts, schema, id alignment, label set: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for generated CSVs (default: ./dataset).",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Generating Kaggle dataset v{VERSION}")
    print(f"Reading FART splits from {SPLITS_DIR}")
    print(f"Writing Kaggle data to    {args.output}\n")
    verify_source_hashes()
    print()
    test_df = write_features(args.output)
    write_solution_and_sample(test_df, args.output)
    print()
    verify_outputs(args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
