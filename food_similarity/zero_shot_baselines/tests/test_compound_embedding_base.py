"""Unit tests for BaseCompoundEmbeddingFeature.

Exercises aggregation paths using a tiny fake cache that avoids the real
FoodAtlas/SMILES/FooDB data dependencies. Uses a stub mapper that returns
a hand-built compound list per ingredient.
"""

import pickle
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

# Add predict/ and shared/ to sys.path for imports
PREDICT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PREDICT_DIR))
sys.path.insert(0, str(PREDICT_DIR.parent.parent / "shared"))

from tests.fixtures.make_fixtures import fake_cache, fake_products, fake_labels


@dataclass
class StubCompound:
    pubchem_cid: Optional[int]
    chebi_id: Optional[int]
    concentration: Optional[float]


@dataclass
class StubMapping:
    compounds: List[StubCompound]
    matched_entity_name: str = ""


class StubMapper:
    """Returns hand-built compound mappings per ingredient string."""

    def __init__(self, per_ingredient: Dict[str, StubMapping]):
        self._per_ingredient = per_ingredient

    def map_product(self, cleaned_ingredients: str):
        mappings = {}
        for ing in cleaned_ingredients.split(" | "):
            ing = ing.strip()
            if ing in self._per_ingredient:
                mappings[ing] = self._per_ingredient[ing]
        return mappings


class StubResolver:
    """Returns a SMILES given a pubchem_cid from a hand-built map."""

    def __init__(self, cid_to_smiles: Dict[int, str]):
        self._map = cid_to_smiles

    def resolve(self, pubchem_cid):
        return self._map.get(pubchem_cid)

    def resolve_chebi(self, chebi_id):
        return None


@pytest.fixture
def tmp_fixtures():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cache_path = tmp / "fake_cache.pkl"
        products_path = tmp / "products.csv"
        labels_path = tmp / "labels.csv"
        fake_cache(cache_path)
        fake_products(products_path)
        fake_labels(labels_path)
        yield {
            "cache_path": cache_path,
            "products_df": pd.read_csv(products_path),
            "labels_df": pd.read_csv(labels_path),
        }


def test_base_feature_loads_cache_and_aggregates(tmp_fixtures, monkeypatch):
    """End-to-end: base feature computes aggregated product embeddings."""
    from lib.features.compound_embedding_base import BaseCompoundEmbeddingFeature

    stub_map = StubMapper({
        "water": StubMapping(compounds=[StubCompound(pubchem_cid=11, chebi_id=None, concentration=100.0)]),
        "salt":  StubMapping(compounds=[StubCompound(pubchem_cid=22, chebi_id=None, concentration=50.0)]),
        "sugar": StubMapping(compounds=[StubCompound(pubchem_cid=33, chebi_id=None, concentration=200.0)]),
    })
    stub_resolver = StubResolver({11: "CCO", 22: "CC(=O)O", 33: "c1ccccc1"})

    def stub_load(self, config):
        with open(tmp_fixtures["cache_path"], "rb") as f:
            embeddings = pickle.load(f)
        return stub_map, stub_resolver, embeddings, None  # no foodb

    monkeypatch.setattr(BaseCompoundEmbeddingFeature, "_load_caches", stub_load)

    config = {
        "embeddings_cache": str(tmp_fixtures["cache_path"]),
        "ingredient_agg_method": "weighted_average",
        "product_agg_method": "weighted_average",
    }

    class TestFeature(BaseCompoundEmbeddingFeature):
        _default_cache_filename = "fake_cache.pkl"
        _source_display_name = "Fake"
        _prep_script_hint = "fake"

    feat = TestFeature(config, tmp_fixtures["products_df"], tmp_fixtures["labels_df"])

    # Product 1 (water | salt): water has CCO ([1,0,0,0]), salt has CC(=O)O ([0,1,0,0])
    # Compound-weighted-avg within ingredient with single compound = the embedding itself.
    # Ingredient-aggregation with inverse-rank weights: water=1/1, salt=1/2, normalized → [2/3, 1/3].
    # Product 1 embedding: 2/3 * [1,0,0,0] + 1/3 * [0,1,0,0] = [2/3, 1/3, 0, 0]
    vec1 = feat.extract(1)
    assert vec1 is not None
    np.testing.assert_allclose(vec1, [2/3, 1/3, 0, 0], atol=1e-6)

    # Product 2 (sugar): single ingredient with single compound → benzene
    vec2 = feat.extract(2)
    assert vec2 is not None
    np.testing.assert_allclose(vec2, [0, 0, 1, 0], atol=1e-6)


def test_max_pool_compound_agg(tmp_fixtures, monkeypatch):
    """Max-pool over compounds returns element-wise max of embeddings."""
    from lib.features.compound_embedding_base import BaseCompoundEmbeddingFeature

    stub_map = StubMapper({
        "water": StubMapping(compounds=[
            StubCompound(pubchem_cid=11, chebi_id=None, concentration=1.0),
            StubCompound(pubchem_cid=33, chebi_id=None, concentration=1.0),
        ]),
        "salt":  StubMapping(compounds=[StubCompound(pubchem_cid=22, chebi_id=None, concentration=1.0)]),
        "sugar": StubMapping(compounds=[StubCompound(pubchem_cid=33, chebi_id=None, concentration=1.0)]),
    })
    stub_resolver = StubResolver({11: "CCO", 22: "CC(=O)O", 33: "c1ccccc1"})

    def stub_load(self, config):
        with open(tmp_fixtures["cache_path"], "rb") as f:
            embeddings = pickle.load(f)
        return stub_map, stub_resolver, embeddings, None

    monkeypatch.setattr(BaseCompoundEmbeddingFeature, "_load_caches", stub_load)

    config = {
        "embeddings_cache": str(tmp_fixtures["cache_path"]),
        "ingredient_agg_method": "max",
        "product_agg_method": "weighted_average",
    }

    class TestFeature(BaseCompoundEmbeddingFeature):
        _default_cache_filename = "fake_cache.pkl"
        _source_display_name = "Fake"
        _prep_script_hint = "fake"

    feat = TestFeature(config, tmp_fixtures["products_df"], tmp_fixtures["labels_df"])

    # Product 1 water ingredient: max([1,0,0,0], [0,0,1,0]) = [1,0,1,0]
    # Product 1 salt ingredient: [0,1,0,0]
    # Product embedding: 2/3 * [1,0,1,0] + 1/3 * [0,1,0,0] = [2/3, 1/3, 2/3, 0]
    vec1 = feat.extract(1)
    np.testing.assert_allclose(vec1, [2/3, 1/3, 2/3, 0], atol=1e-6)



