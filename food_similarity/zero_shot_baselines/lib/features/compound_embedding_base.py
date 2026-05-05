"""Shared base class for compound-embedding features (FART).

Contains all source-agnostic aggregation logic. Subclasses specify the
default cache filename and display name via class-level attributes.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from compound_mapping.food_atlas import FoodAtlasMapper
from compound_mapping.foodb_concentrations import FooDBConcentrations
from compound_mapping.smiles_resolver import SMILESResolver
from .base import BaseFeature

logger = logging.getLogger(__name__)


class BaseCompoundEmbeddingFeature(BaseFeature):
    """Base class for features that aggregate per-SMILES embeddings to product level.

    Subclasses must set class-level attributes:
        _default_cache_filename: str — pickle filename under shared/data/caches/
        _source_display_name: str     — short name used in log messages
        _prep_script_hint: str        — name of the prep script referenced in errors

    Aggregation pipeline:
        compound → ingredient: configurable (weighted_average | mean | max |
            log_weighted_average | top3_by_conc)
        ingredient → product:  configurable (weighted_average | average |
            sum | max)
    """

    _default_cache_filename: str = ""
    _source_display_name: str = ""
    _prep_script_hint: str = ""

    def __init__(
        self,
        config: dict,
        products_df: pd.DataFrame,
        labels_df: pd.DataFrame,
    ) -> None:
        # Set basic attributes directly; the real products schema uses
        # "Product code" (title case) while some test fixtures use
        # "product_code". We build `_product_index` defensively so either
        # schema works — the compound pipeline only needs code→ingredients
        # from labels_df anyway.
        self.config = config
        self.products_df = products_df
        self.labels_df = labels_df
        if "Product code" in products_df.columns:
            code_col = "Product code"
        elif "product_code" in products_df.columns:
            code_col = "product_code"
        else:
            code_col = None
        if code_col is not None:
            self._product_index = {
                int(row[code_col]): idx
                for idx, row in products_df.iterrows()
            }
        else:
            self._product_index = {}

        self.ingredient_agg = config.get("ingredient_agg_method", "weighted_average")
        self.product_agg = config.get("product_agg_method", "average")

        self.mapper, self.smiles_resolver, self._embeddings, self.foodb_conc = (
            self._load_caches(config)
        )

        self._code_to_ingredients, self._ingredient_weights = (
            self.parse_ingredients_with_weights(labels_df)
        )

        self._precompute()

    def _load_caches(self, config: dict):
        """Load FoodAtlas, SMILES resolver, source embeddings, and FooDB fallback."""
        unsupervised_dir = Path(__file__).resolve().parent.parent.parent
        neurips_dir = unsupervised_dir.parent.parent
        shared_dir = neurips_dir / "shared"

        def resolve(cfg_key, default):
            p = Path(config.get(cfg_key, default))
            return str(p if p.is_absolute() else unsupervised_dir / p)

        logger.info("Loading FoodAtlas mapper...")
        mapper = FoodAtlasMapper(resolve(
            "food_atlas_dir",
            str(neurips_dir / "data" / "food_atlas" / "v4.0"),
        ))

        logger.info("Loading SMILES cache...")
        smiles_resolver = SMILESResolver(
            resolve("smiles_cache", str(shared_dir / "data" / "caches" / "smiles_cache.csv")),
            cache_only=True,
        )

        logger.info(f"Loading {self._source_display_name} embeddings cache...")
        emb_path = Path(resolve(
            "embeddings_cache",
            str(shared_dir / "data" / "caches" / self._default_cache_filename),
        ))
        if not emb_path.exists():
            raise FileNotFoundError(
                f"{self._source_display_name} embeddings not found at {emb_path}. "
                f"Run scripts/{self._prep_script_hint} first."
            )
        with open(emb_path, "rb") as f:
            embeddings: Dict[str, np.ndarray] = pickle.load(f)
        logger.info(f"Loaded {len(embeddings)} {self._source_display_name} embeddings")

        foodb_conc: Optional[FooDBConcentrations] = None
        foodb_path = Path(resolve(
            "foodb_dir", str(neurips_dir / "data" / "foodb_2020_04_07_csv"),
        ))
        if foodb_path.exists():
            logger.info("Loading FooDB concentration fallback...")
            foodb_conc = FooDBConcentrations(str(foodb_path))
            logger.info(f"FooDB fallback: {foodb_conc.coverage} compounds")

        return mapper, smiles_resolver, embeddings, foodb_conc

    def _resolve_smiles(self, compound) -> Optional[str]:
        if compound.pubchem_cid is not None:
            smiles = self.smiles_resolver.resolve(compound.pubchem_cid)
            if smiles:
                return smiles
        if compound.chebi_id is not None:
            smiles = self.smiles_resolver.resolve_chebi(compound.chebi_id)
            if smiles:
                return smiles
        return None

    def _get_compound_embedding(
        self, compound
    ) -> Optional[Tuple[np.ndarray, str]]:
        smiles = self._resolve_smiles(compound)
        if smiles is None:
            return None
        emb = self._embeddings.get(smiles)
        if emb is None:
            return None
        return emb, smiles

    def _get_concentration_with_fallback(
        self, compound, smiles: Optional[str], food_entity_name: Optional[str] = None
    ) -> Optional[float]:
        if compound.concentration is not None:
            return compound.concentration
        if self.foodb_conc and smiles:
            return self.foodb_conc.get_concentration(smiles, food_name=food_entity_name)
        return None

    def _aggregate_compounds_to_ingredient(
        self,
        compound_embeddings: List[Tuple[np.ndarray, Optional[float]]],
    ) -> Optional[np.ndarray]:
        if not compound_embeddings:
            return None

        embeddings = [e for e, _ in compound_embeddings]
        concentrations = [c for _, c in compound_embeddings]

        if self.ingredient_agg == "mean":
            return np.mean(embeddings, axis=0)
        elif self.ingredient_agg == "weighted_average":
            weights = []
            for c in concentrations:
                weights.append(c if c is not None and c > 0 else 1.0)
            weights = np.array(weights)
            weights = weights / weights.sum()
            return np.average(embeddings, axis=0, weights=weights)
        elif self.ingredient_agg == "max":
            return np.max(embeddings, axis=0)
        elif self.ingredient_agg == "log_weighted_average":
            weights = []
            for c in concentrations:
                w = np.log1p(c) if c is not None and c > 0 else 0.0
                weights.append(max(w, 1e-6))
            weights = np.array(weights)
            weights = weights / weights.sum()
            return np.average(embeddings, axis=0, weights=weights)
        elif self.ingredient_agg == "top3_by_conc":
            pairs = sorted(
                zip(embeddings, concentrations),
                key=lambda p: (p[1] if p[1] is not None else -np.inf),
                reverse=True,
            )
            top = [e for e, _ in pairs[:3]]
            return np.mean(top, axis=0)
        else:
            return np.mean(embeddings, axis=0)

    def _aggregate_ingredients_to_product(
        self,
        ingredient_embeddings: List[Tuple[str, np.ndarray]],
        product_code: int,
    ) -> Optional[np.ndarray]:
        if not ingredient_embeddings:
            return None

        names = [name for name, _ in ingredient_embeddings]
        embeddings = [emb for _, emb in ingredient_embeddings]

        if self.product_agg == "weighted_average" and self._ingredient_weights:
            weights = []
            for name in names:
                w = self._ingredient_weights.get((product_code, name))
                weights.append(w if w is not None and w > 0 else 1.0)
            weights = np.array(weights)
            weights = weights / weights.sum()
            return np.average(embeddings, axis=0, weights=weights)
        elif self.product_agg == "sum":
            return np.sum(embeddings, axis=0)
        elif self.product_agg == "max":
            return np.max(embeddings, axis=0)
        else:
            return np.mean(embeddings, axis=0)

    def _compute_product_embedding(self, product_code: int) -> Optional[np.ndarray]:
        ingredient_list = self._code_to_ingredients.get(product_code, "")
        if not ingredient_list:
            return None

        mappings = self.mapper.map_product(ingredient_list)

        ingredient_embeddings: List[Tuple[str, np.ndarray]] = []
        for ing_name, mapping in mappings.items():
            if mapping is None or not mapping.compounds:
                continue

            compound_embs = []
            food_entity = mapping.matched_entity_name or ""
            food_entity_clean = food_entity.split(" (+")[0].strip()
            for compound in mapping.compounds:
                result = self._get_compound_embedding(compound)
                if result is not None:
                    emb, smiles = result
                    conc = self._get_concentration_with_fallback(
                        compound, smiles, food_entity_name=food_entity_clean
                    )
                    compound_embs.append((emb, conc))

            if not compound_embs:
                continue

            ing_emb = self._aggregate_compounds_to_ingredient(compound_embs)
            if ing_emb is not None:
                ingredient_embeddings.append((ing_name, ing_emb))

        return self._aggregate_ingredients_to_product(
            ingredient_embeddings, product_code
        )

    def _precompute(self) -> None:
        self._vectors: Dict[int, np.ndarray] = {}
        no_embedding = 0
        for code in self._code_to_ingredients:
            emb = self._compute_product_embedding(code)
            if emb is not None:
                self._vectors[code] = emb
            else:
                no_embedding += 1
        logger.info(
            f"{self._source_display_name} feature: {len(self._vectors)} products with "
            f"embeddings, {no_embedding} without"
        )

    def extract(self, product_code: int) -> Optional[np.ndarray]:
        return self._vectors.get(product_code)
