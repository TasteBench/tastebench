"""Resolve PubChem CIDs to canonical SMILES strings.

Uses the PubChem PUG-REST API with a CSV file cache for persistence.
Supports batch resolution and incremental saving.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid"
BATCH_SIZE = 100  # PubChem supports up to ~200, but 100 is safer
REQUEST_DELAY = 0.25  # seconds between requests (5 req/sec limit)


class SMILESResolver:
    """Resolve PubChem CIDs to canonical SMILES with CSV caching.

    Also supports ChEBI ID → SMILES resolution via a separate cache.
    """

    def __init__(self, cache_path: str | Path, cache_only: bool = False) -> None:
        self.cache_path = Path(cache_path)
        self.cache_only = cache_only  # If True, never call PubChem API
        self._cache: Dict[int, str] = {}
        self._failed: Set[int] = set()
        self._chebi_cache: Dict[int, str] = {}  # chebi_id → smiles
        self._load_cache()
        self._load_chebi_cache()

    def _load_cache(self) -> None:
        """Load existing cache from CSV."""
        if self.cache_path.exists():
            df = pd.read_csv(self.cache_path)
            for _, row in df.iterrows():
                cid = int(row["pubchem_cid"])
                smiles = row["canonical_smiles"]
                if pd.notna(smiles) and str(smiles).strip():
                    self._cache[cid] = str(smiles).strip()
                else:
                    self._failed.add(cid)
            logger.info(
                f"PubChem SMILES cache: {len(self._cache)} resolved, "
                f"{len(self._failed)} failed"
            )

    def _load_chebi_cache(self) -> None:
        """Load ChEBI → SMILES cache if available."""
        chebi_path = self.cache_path.parent / "chebi_smiles_cache.csv"
        if chebi_path.exists():
            df = pd.read_csv(chebi_path)
            for _, row in df.iterrows():
                chebi_id = int(row["chebi_id"])
                smiles = row.get("canonical_smiles", "")
                if pd.notna(smiles) and str(smiles).strip():
                    self._chebi_cache[chebi_id] = str(smiles).strip()
            logger.info(f"ChEBI SMILES cache: {len(self._chebi_cache)} resolved")

    def resolve_chebi(self, chebi_id: int) -> Optional[str]:
        """Return canonical SMILES for a ChEBI ID, or None."""
        return self._chebi_cache.get(chebi_id)

    def save_cache(self) -> None:
        """Save cache to CSV."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for cid, smiles in sorted(self._cache.items()):
            rows.append({"pubchem_cid": cid, "canonical_smiles": smiles})
        for cid in sorted(self._failed):
            rows.append({"pubchem_cid": cid, "canonical_smiles": ""})
        pd.DataFrame(rows).to_csv(self.cache_path, index=False)
        logger.info(f"SMILES cache saved: {len(self._cache)} entries to {self.cache_path}")

    def resolve(self, pubchem_cid: int) -> Optional[str]:
        """Return canonical SMILES for a CID, or None if unavailable."""
        if pubchem_cid in self._cache:
            return self._cache[pubchem_cid]
        if pubchem_cid in self._failed:
            return None
        if self.cache_only:
            return None
        # Single resolve via API
        result = self._fetch_batch([pubchem_cid])
        return result.get(pubchem_cid)

    def resolve_all(
        self,
        cids: Set[int],
        save_interval: int = 1000,
    ) -> Dict[int, str]:
        """Batch-resolve all CIDs, using cache and API.

        Args:
            cids: Set of PubChem CIDs to resolve.
            save_interval: Save cache every N new resolutions.

        Returns:
            Dict mapping CID → canonical SMILES for all resolved CIDs.
        """
        # Filter to only unresolved CIDs
        to_resolve = [
            cid for cid in cids
            if cid not in self._cache and cid not in self._failed
        ]

        if not to_resolve:
            logger.info(f"All {len(cids)} CIDs already cached.")
            return {cid: self._cache[cid] for cid in cids if cid in self._cache}

        logger.info(
            f"Resolving {len(to_resolve)} new CIDs "
            f"({len(self._cache)} cached, {len(self._failed)} failed)"
        )

        resolved_count = 0
        for i in range(0, len(to_resolve), BATCH_SIZE):
            batch = to_resolve[i : i + BATCH_SIZE]
            result = self._fetch_batch(batch)

            # Mark failures
            for cid in batch:
                if cid not in result:
                    self._failed.add(cid)

            resolved_count += len(result)

            if resolved_count % save_interval < BATCH_SIZE:
                self.save_cache()

            # Progress logging
            total_done = i + len(batch)
            if total_done % 5000 < BATCH_SIZE:
                logger.info(
                    f"  Progress: {total_done}/{len(to_resolve)} "
                    f"({total_done / len(to_resolve) * 100:.1f}%)"
                )

        self.save_cache()
        logger.info(f"Resolution complete: {resolved_count} new, {len(self._cache)} total")

        return {cid: self._cache[cid] for cid in cids if cid in self._cache}

    def _fetch_batch(self, cids: List[int]) -> Dict[int, str]:
        """Fetch SMILES for a batch of CIDs from PubChem PUG-REST."""
        if not cids:
            return {}

        cid_str = ",".join(str(c) for c in cids)
        url = f"{PUBCHEM_BASE}/{cid_str}/property/CanonicalSMILES/JSON"

        try:
            resp = requests.get(url, timeout=30)
            time.sleep(REQUEST_DELAY)

            if resp.status_code == 404:
                # None of the CIDs found
                return {}

            resp.raise_for_status()
            data = resp.json()

            result = {}
            for prop in data.get("PropertyTable", {}).get("Properties", []):
                cid = int(prop["CID"])
                # PubChem returns SMILES under varying key names depending on
                # the property requested: ConnectivitySMILES, SMILES, etc.
                smiles = (
                    prop.get("CanonicalSMILES")
                    or prop.get("ConnectivitySMILES")
                    or prop.get("SMILES")
                    or prop.get("IsomericSMILES")
                    or ""
                )
                if smiles:
                    self._cache[cid] = smiles
                    result[cid] = smiles

            return result

        except requests.RequestException as e:
            logger.warning(f"PubChem API error for batch of {len(cids)}: {e}")
            # Retry individual CIDs on batch failure
            if len(cids) > 1:
                result = {}
                for cid in cids:
                    single = self._fetch_batch([cid])
                    result.update(single)
                return result
            return {}

        except (ValueError, KeyError) as e:
            logger.warning(f"PubChem response parse error: {e}")
            return {}
