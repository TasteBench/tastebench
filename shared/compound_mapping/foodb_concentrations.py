"""FooDB concentration data cross-reference.

Loads FooDB's Content_Compounds table and provides concentration lookups
by PubChem CID, supplementing FoodAtlas's sparse concentration data.

FooDB has 718K concentration entries vs FoodAtlas's 28K.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from rdkit import Chem

logger = logging.getLogger(__name__)


class FooDBConcentrations:
    """Cross-reference FooDB concentrations by PubChem CID.

    Builds a mapping: PubChem CID → median concentration (mg/100g)
    across all foods in FooDB. This provides a "typical" concentration
    for each compound that can supplement FoodAtlas when concentrations
    are missing.
    """

    def __init__(self, foodb_dir: str | Path) -> None:
        self.foodb_dir = Path(foodb_dir)
        self._cid_to_conc: Dict[int, float] = {}
        self._smiles_to_conc: Dict[str, float] = {}
        self._food_compound_conc: Dict[Tuple[str, str], float] = {}
        self._foodb_food_names: set = set()
        self._load()

    def _load(self) -> None:
        """Load FooDB compound concentrations and build CID index."""
        compound_path = self.foodb_dir / "Compound.csv"
        # FooDB ships as Content_Compounds.csv or Content.csv depending on version
        content_path = self.foodb_dir / "Content_Compounds.csv"
        if not content_path.exists():
            content_path = self.foodb_dir / "Content.csv"

        if not compound_path.exists() or not content_path.exists():
            logger.warning(f"FooDB files not found at {self.foodb_dir}")
            return

        # Note: FooDB's CSV has shifted columns.
        # moldb_inchikey actually contains InChI strings (not InChI keys).
        # We convert InChI → canonical SMILES via RDKit.
        logger.info("Loading FooDB compounds...")
        compounds_df = pd.read_csv(
            compound_path,
            usecols=["id", "moldb_inchikey"],  # actually InChI strings
            low_memory=False,
        )

        # Build compound_id → canonical SMILES mapping
        compound_smiles: Dict[int, str] = {}
        for _, row in compounds_df.iterrows():
            cid = row["id"]
            inchi = row.get("moldb_inchikey")
            if pd.notna(inchi) and str(inchi).startswith("InChI="):
                try:
                    mol = Chem.MolFromInchi(str(inchi))
                    if mol:
                        canonical = Chem.MolToSmiles(mol, canonical=True)
                        compound_smiles[int(cid)] = canonical
                except Exception:
                    pass

        logger.info(f"FooDB compounds with SMILES (via InChI): {len(compound_smiles)}")

        # Load food names for per-food lookups
        food_path = self.foodb_dir / "Food.csv"
        food_df = pd.read_csv(food_path, usecols=["id", "name"])
        food_id_to_name: Dict[int, str] = dict(
            zip(food_df["id"].astype(int), food_df["name"].str.lower().str.strip())
        )

        # Load content data with concentrations
        logger.info("Loading FooDB concentrations...")
        content_df = pd.read_csv(
            content_path,
            usecols=["food_id", "source_id", "orig_content", "orig_unit"],
            low_memory=False,
        )

        # Filter to rows with concentration values and mg/100g units
        content_df = content_df[content_df["orig_content"].notna()]
        content_df["orig_content"] = pd.to_numeric(
            content_df["orig_content"], errors="coerce"
        )
        content_df = content_df[content_df["orig_content"] > 0]

        # Normalize units to mg/100g
        mg_mask = content_df["orig_unit"].isin(["mg/100g", "mg/100 g"])
        pct_mask = content_df["orig_unit"] == "%"
        content_df.loc[pct_mask, "orig_content"] *= 1000
        content_df = content_df[mg_mask | pct_mask].copy()

        logger.info(f"FooDB concentration entries (mg/100g): {len(content_df)}")

        # Build per-food concentration index: (food_name, smiles) → concentration
        self._food_compound_conc: Dict[Tuple[str, str], float] = {}
        for _, row in content_df.iterrows():
            food_name = food_id_to_name.get(int(row["food_id"]))
            smiles = compound_smiles.get(int(row["source_id"]))
            if food_name and smiles:
                key = (food_name, smiles)
                # Keep max concentration across measurements
                if key not in self._food_compound_conc or row["orig_content"] > self._food_compound_conc[key]:
                    self._food_compound_conc[key] = float(row["orig_content"])

        # Also build global median as fallback
        self._smiles_to_conc: Dict[str, float] = {}
        median_conc = content_df.groupby("source_id")["orig_content"].median().to_dict()
        for compound_id, conc in median_conc.items():
            smiles = compound_smiles.get(int(compound_id))
            if smiles:
                self._smiles_to_conc[smiles] = conc

        # Build food name index for fuzzy food matching
        self._foodb_food_names: set = set(food_id_to_name.values())

        n_per_food = len(set(k[0] for k in self._food_compound_conc))
        logger.info(
            f"FooDB concentration index: {len(self._smiles_to_conc)} compounds (global), "
            f"{len(self._food_compound_conc)} per-food entries across {n_per_food} foods"
        )

    # Mapping from FoodAtlas food names to FooDB food names
    _FOODATLAS_TO_FOODB = {
        "soybean": "soy bean",
        "soybean flour (defatted)": "soy bean",
        "soybean flour": "soy bean",
        "soybean milk": "soy milk",
        "soybean oil": "soybean oil",
        "common wheat kernel": "common wheat",
        "pea": "common pea",
        "oat": "oat",
        "oat flour": "oat",
        "rice": "rice",
        "coconut oil": "coconut oil",
        "coconut": "coconut meat",
        "maize (corn) food product": "corn",
        "mushroom fruitbody": "common mushroom",
        "white button mushroom": "common mushroom",
        "sunflower oil": "sunflower oil",
        "sunflower seed": "sunflower seed",
        "palm oil": "palm oil",
        "garlic bulb": "garlic",
        "onion": "common onion",
        "potato": "potato",
        "milk": "milk (cow)",
        "cow whole milk 3.5% fat": "milk (cow)",
        "sour cream": "sour cream",
        "cheddar cheese": "cheddar cheese",
        "ground beef": "cattle",
        "chickpea": "chickpea",
        "lentil": "lentil",
        "faba bean": "fava bean",
        "tapioca": "cassava",
    }

    def get_concentration(
        self, smiles: str, food_name: Optional[str] = None
    ) -> Optional[float]:
        """Get concentration (mg/100g) for a compound.

        Tries per-food lookup first (if food_name given), then global median.
        """
        if food_name:
            # Try FoodAtlas name → FooDB name mapping
            foodb_name = self._FOODATLAS_TO_FOODB.get(food_name.lower())
            if foodb_name:
                conc = self._food_compound_conc.get((foodb_name, smiles))
                if conc is not None:
                    return conc
            # Try food_name directly
            conc = self._food_compound_conc.get((food_name.lower(), smiles))
            if conc is not None:
                return conc

        # Fallback: global median
        return self._smiles_to_conc.get(smiles)

    @property
    def coverage(self) -> int:
        """Number of compounds with concentration data."""
        return len(self._smiles_to_conc)
