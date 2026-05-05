"""Resolve ChEBI-only FoodAtlas compounds to SMILES via PubChem.

14,740 FoodAtlas chemical entities have ChEBI IDs but no PubChem CIDs.
PubChem can resolve ChEBI IDs to SMILES via its name/synonym search.

Appends results to the existing smiles_cache.csv.

Usage:
    python shared/scripts/prepare_caches/resolve_chebi_smiles.py
"""

import logging
import sys
import time
from ast import literal_eval
from pathlib import Path

import pandas as pd
import requests

neurips_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(neurips_dir / "shared"))
shared_dir = neurips_dir / "shared"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
REQUEST_DELAY = 0.25


def resolve_chebi_to_smiles(chebi_id: int) -> tuple:
    """Resolve a single ChEBI ID to (pubchem_cid, smiles) via PubChem."""
    url = f"{PUBCHEM_BASE}/CHEBI:{chebi_id}/property/CanonicalSMILES/JSON"
    try:
        resp = requests.get(url, timeout=15)
        time.sleep(REQUEST_DELAY)
        if resp.status_code == 404:
            return None, None
        resp.raise_for_status()
        data = resp.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        if props:
            cid = props[0].get("CID")
            smiles = (
                props[0].get("CanonicalSMILES")
                or props[0].get("ConnectivitySMILES")
                or props[0].get("SMILES")
            )
            return cid, smiles
    except Exception as e:
        logger.debug(f"ChEBI:{chebi_id} failed: {e}")
    return None, None


def main():
    food_atlas_dir = neurips_dir / "data" / "food_atlas" / "v4.0"
    cache_path = shared_dir / "data" / "caches" / "smiles_cache.csv"
    chebi_cache_path = shared_dir / "data" / "caches" / "chebi_smiles_cache.csv"

    # Load FoodAtlas entities to find ChEBI-only chemicals
    logger.info("Loading FoodAtlas entities...")
    entities = pd.read_csv(
        food_atlas_dir / "entities.tsv", sep="\t",
        converters={"external_ids": literal_eval},
    )
    chems = entities[entities["entity_type"] == "chemical"]

    chebi_only = []
    for _, row in chems.iterrows():
        ext = row["external_ids"] if isinstance(row["external_ids"], dict) else {}
        cids = ext.get("pubchem_compound", [])
        chebis = ext.get("chebi", [])
        if not cids and chebis:
            chebi_only.append((row["foodatlas_id"], chebis[0]))

    logger.info(f"ChEBI-only compounds: {len(chebi_only)}")

    # Load existing ChEBI cache for resume
    resolved = {}
    failed = set()
    if chebi_cache_path.exists():
        df = pd.read_csv(chebi_cache_path)
        for _, row in df.iterrows():
            cid = row["chebi_id"]
            smiles = row.get("canonical_smiles", "")
            pcid = row.get("pubchem_cid", "")
            if pd.notna(smiles) and str(smiles).strip():
                resolved[int(cid)] = (int(pcid) if pd.notna(pcid) else None, str(smiles))
            else:
                failed.add(int(cid))
        logger.info(f"Existing cache: {len(resolved)} resolved, {len(failed)} failed")

    # Filter to unresolved
    to_resolve = [
        (fa_id, chebi_id) for fa_id, chebi_id in chebi_only
        if chebi_id not in resolved and chebi_id not in failed
    ]
    logger.info(f"To resolve: {len(to_resolve)}")

    # Resolve
    new_resolved = 0
    for i, (fa_id, chebi_id) in enumerate(to_resolve):
        pcid, smiles = resolve_chebi_to_smiles(chebi_id)
        if smiles:
            resolved[chebi_id] = (pcid, smiles)
            new_resolved += 1
        else:
            failed.add(chebi_id)

        # Progress + save
        if (i + 1) % 500 == 0 or i == len(to_resolve) - 1:
            logger.info(
                f"  Progress: {i+1}/{len(to_resolve)} "
                f"({new_resolved} resolved, {len(failed)} failed)"
            )
            # Save incrementally
            rows = []
            for cid, (pcid, smi) in sorted(resolved.items()):
                rows.append({"chebi_id": cid, "pubchem_cid": pcid or "", "canonical_smiles": smi})
            for cid in sorted(failed):
                rows.append({"chebi_id": cid, "pubchem_cid": "", "canonical_smiles": ""})
            pd.DataFrame(rows).to_csv(chebi_cache_path, index=False)

    logger.info(f"Done. {len(resolved)} total resolved, {len(failed)} failed")
    logger.info(f"Cache saved to: {chebi_cache_path}")

    # Also append resolved CIDs to the main smiles_cache
    if resolved:
        logger.info("Appending to main SMILES cache...")
        existing = pd.read_csv(cache_path)
        existing_cids = set(existing["pubchem_cid"].astype(int))
        new_rows = []
        for chebi_id, (pcid, smiles) in resolved.items():
            if pcid and pcid not in existing_cids:
                new_rows.append({"pubchem_cid": pcid, "canonical_smiles": smiles})
        if new_rows:
            combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            combined.to_csv(cache_path, index=False)
            logger.info(f"Added {len(new_rows)} new CIDs to smiles_cache.csv")


if __name__ == "__main__":
    main()
