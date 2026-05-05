"""FART taste compound embeddings (768-dim).

Thin subclass of BaseCompoundEmbeddingFeature. Aggregation pipeline:
ingredient → FoodAtlas compounds → SMILES → FART embedding → aggregate.

Requires pre-computed caches in ../../shared/data/caches/:
- smiles_cache.csv
- fart_compound_embeddings.pkl
"""

from .compound_embedding_base import BaseCompoundEmbeddingFeature


class CompoundFeature(BaseCompoundEmbeddingFeature):
    """FART taste embeddings aggregated to product level."""

    _default_cache_filename = "fart_compound_embeddings.pkl"
    _source_display_name = "FART"
    _prep_script_hint = "prepare_fart_embeddings.py"
