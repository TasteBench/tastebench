"""One-time script: compute FART embeddings for all SMILES in the cache.

Usage:
    python shared/scripts/prepare_caches/prepare_fart_embeddings.py

Reads shared/data/caches/smiles_cache.csv and generates
768-dim FART embeddings for each unique SMILES string using the
FartLabs/FART_Augmented model. Saves results to
shared/data/caches/fart_compound_embeddings.pkl.
"""

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

# Add  root to path
neurips_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(neurips_dir / "shared"))
shared_dir = neurips_dir / "shared"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TOKENIZER_NAME = "seyonec/SMILES_tokenized_PubChem_shard00_160k"
MODEL_NAME = "FartLabs/FART_Augmented"
MAX_LENGTH = 512
BATCH_SIZE = 32
EMBEDDING_DIM = 768


def main():
    smiles_cache_path = shared_dir / "data" / "caches" / "smiles_cache.csv"
    output_path = shared_dir / "data" / "caches" / "fart_compound_embeddings.pkl"

    # Load SMILES
    df = pd.read_csv(smiles_cache_path)
    df = df[df["canonical_smiles"].notna() & (df["canonical_smiles"] != "")]
    unique_smiles = list(df["canonical_smiles"].unique())
    logger.info(f"Unique SMILES to embed: {len(unique_smiles)}")

    # Load existing cache if present (for resume)
    embeddings: dict[str, np.ndarray] = {}
    if output_path.exists():
        with open(output_path, "rb") as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded existing cache: {len(embeddings)} embeddings")

    # Filter to only new SMILES
    to_embed = [s for s in unique_smiles if s not in embeddings]
    logger.info(f"New SMILES to embed: {len(to_embed)} ({len(embeddings)} cached)")

    if not to_embed:
        logger.info("All SMILES already embedded. Done.")
        return

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading FART model on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    model.to(device)
    logger.info("Model loaded.")

    # Batch inference
    for i in range(0, len(to_embed), BATCH_SIZE):
        batch = to_embed[i : i + BATCH_SIZE]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Extract [CLS] token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        for j, smiles in enumerate(batch):
            embeddings[smiles] = cls_embeddings[j].astype(np.float32)

        # Progress + incremental save
        done = i + len(batch)
        if done % 5000 < BATCH_SIZE or done == len(to_embed):
            logger.info(f"  Progress: {done}/{len(to_embed)} ({done/len(to_embed)*100:.1f}%)")
            with open(output_path, "wb") as f:
                pickle.dump(embeddings, f)

    # Final save
    with open(output_path, "wb") as f:
        pickle.dump(embeddings, f)

    logger.info(f"Done. {len(embeddings)} embeddings saved to {output_path}")

    # Verify
    sample = list(embeddings.values())[:3]
    for i, emb in enumerate(sample):
        logger.info(f"  Sample {i}: shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")


if __name__ == "__main__":
    main()
