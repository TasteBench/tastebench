"""One-time script: resolve all FoodAtlas PubChem CIDs to SMILES.

Usage:
    python shared/scripts/prepare_caches/prepare_smiles_cache.py

This fetches canonical SMILES for all chemical compounds in FoodAtlas via
the PubChem PUG-REST API. Results are cached to
shared/data/caches/smiles_cache.csv.

At ~100 CIDs per request and 5 req/sec, ~175K CIDs takes ~6 minutes.
The script saves incrementally and is resume-safe.
"""

import logging
import sys
from pathlib import Path

# Add  root to path
neurips_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(neurips_dir / "shared"))

from compound_mapping.food_atlas import FoodAtlasMapper
from compound_mapping.smiles_resolver import SMILESResolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    # Paths
    shared_dir = neurips_dir / "shared"
    food_atlas_dir = neurips_dir / "data" / "food_atlas" / "v4.0"
    cache_path = shared_dir / "data" / "caches" / "smiles_cache.csv"

    # Load FoodAtlas to get all PubChem CIDs
    logger.info("Loading FoodAtlas...")
    mapper = FoodAtlasMapper(food_atlas_dir)
    all_cids = mapper.get_all_pubchem_cids()
    logger.info(f"Total unique PubChem CIDs in FoodAtlas: {len(all_cids)}")

    # Resolve SMILES
    resolver = SMILESResolver(cache_path)
    resolved = resolver.resolve_all(all_cids, save_interval=5000)

    # Summary
    logger.info(f"Resolved: {len(resolved)}/{len(all_cids)} CIDs have SMILES")
    logger.info(f"Cache saved to: {cache_path}")


if __name__ == "__main__":
    main()
