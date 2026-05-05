"""CLI: compute per-SMILES embeddings from a trained D-MPNN checkpoint."""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from molecular.src.models.dmpnn import (
    load_checkpoint,
    predict_embedding,
)

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


def run(checkpoint: Path, smiles_csv: Path, output_pkl: Path,
        extraction_point: str, device: str, batch_size: int,
        smiles_column: str) -> dict:
    df = pd.read_csv(smiles_csv)
    smis = df[smiles_column].tolist()

    # Filter un-parseable
    valid_idx, valid_smis = [], []
    for i, s in enumerate(smis):
        if Chem.MolFromSmiles(s) is not None:
            valid_idx.append(i); valid_smis.append(s)
    n_failed = len(smis) - len(valid_smis)
    if n_failed / max(1, len(smis)) > 0.01:
        raise RuntimeError(f"> 1% SMILES failed to parse: {n_failed}/{len(smis)}")
    logger.info("Parse: %d ok, %d failed (%.2f%%)", len(valid_smis), n_failed, 100 * n_failed / max(1, len(smis)))

    model, cfg = load_checkpoint(checkpoint, device="cpu")
    embs = predict_embedding(model, valid_smis, extraction_point=extraction_point,
                             batch_size=batch_size, device=device)

    out = {s: embs[j].astype(np.float32) for j, s in enumerate(valid_smis)}
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pickle.dump(out, f)
    logger.info("Wrote %d embeddings → %s (dim=%d)", len(out), output_pkl, embs.shape[1])
    return {"n_written": len(out), "dim": int(embs.shape[1]), "n_failed": n_failed}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--smiles_csv", type=Path, required=True)
    ap.add_argument("--output_pkl", type=Path, required=True)
    ap.add_argument("--extraction_point", choices=["encoder", "penultimate"], default="penultimate")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--smiles_column", default="Canonicalized SMILES")
    args = ap.parse_args()
    run(args.checkpoint, args.smiles_csv, args.output_pkl, args.extraction_point,
        args.device, args.batch_size, args.smiles_column)


if __name__ == "__main__":
    main()
