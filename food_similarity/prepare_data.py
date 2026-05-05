"""One-time data preparation for supervised ranking models.

Sources all data from data/ and shared/ only.
No dependency on kaggle_tastebench/ or zero_shot_baselines/.

Reads raw NECTAR data (ingredients, nutrition, sensory ratings), extracts
features (nutrition, compound, text, image), generates pairs, and saves
a self-contained dataset.

Usage:
    cd food_similarity
    python prepare_data.py

Outputs:
    data/product_features.pkl  — {(category, product_code): {nutrition, compound, text, image, ...}}
    data/pairs.csv             — all NECTAR-vs-NECTAR scored pairs
"""

import logging
import os
import pickle
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Path setup ---
SUPERVISED_DIR = Path(__file__).resolve().parent
NEURIPS_DIR = SUPERVISED_DIR.parent
SHARED_DIR = NEURIPS_DIR / "shared"
DATA_DIR = NEURIPS_DIR / "data"
DATA_OUT = SUPERVISED_DIR / "data"

# Add shared/ for compound_mapping imports
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Data paths (all under data/ or shared/) ---
INGREDIENTS_CSV = DATA_DIR / "consolidated_datasets" / "nectar_consolidated_ingredients_nutrition.csv"
SENSORY_CSV = DATA_DIR / "consolidated_datasets" / "nectar_consolidated_sensory_rating.csv"
PRODUCT_LABELS_CSV = SHARED_DIR / "data" / "nectar_product_labels.csv"
IMAGE_DIR = DATA_DIR / "product_images" / "cropped"
FOOD_ATLAS_DIR = DATA_DIR / "food_atlas" / "v4.0"
FOODB_DIR = DATA_DIR / "foodb_2020_04_07_csv"
SMILES_CACHE = SHARED_DIR / "data" / "caches" / "smiles_cache.csv"
# FART_EMBEDDINGS is overridable via EMBEDDINGS_CACHE_PATH env var so taste_gnn
# can swap in GNN-produced embeddings without touching this file (used by
# table_taste_gnn.tex pipeline).
FART_EMBEDDINGS = (
    Path(os.environ["EMBEDDINGS_CACHE_PATH"])
    if os.environ.get("EMBEDDINGS_CACHE_PATH")
    else SHARED_DIR / "data" / "caches" / "fart_compound_embeddings.pkl"
)
SENSORY_DESCRIPTIONS = SHARED_DIR / "data" / "caches" / "sensory_descriptions.csv"

DROP_CATEGORIES = ["Cold_Unbreaded_Chicken_Breast", "Tenders"]

# Image directory names that differ from NECTAR category names.
# Matches the CATEGORY_CONFIG mapping in kaggle_tastebench/generate_data/add_images_to_dataset.py.
IMAGE_CATEGORY_MAP = {
    "Butter": "Salted_Butter",
    "Cheddar_Cheese": "Cheddar_Cheese_Slices",
    "Chicken_Strips": "Unbreaded_chicken_strips_and_chunks",
    "Mozzarella": "Mozzarella_Cheese",
    "Yogurt": "Plain_Greek_Yogurt",
}

# Unified nutrition columns: union of all category-specific subsets from the zero-shot
# baselines pipeline (meat: Fat/Sodium/Protein/Fiber, dairy: Fat/Sodium,
# cheese: Fat/Sodium/Carbs, sweet dairy: Fat/Sugars). Every product gets all 6
# columns — models learn which are relevant per category.
NUTRITION_COLUMNS = [
    "Total Fat (g)",
    "Sodium (mg)",
    "Protein (g)",
    "Dietary Fiber (g)",
    "Total Carbohydrate (g)",
    "Total Sugars (g)",
]

# Category subsets (4 groups matching the kaggle_tastebench pipeline's nutrition grouping)
CATEGORY_SUBSETS = {
    "meat": [
        "Bacon", "Bratwurst", "Breakfast_Sausages", "Burgers", "Chicken_Strips",
        "Breaded_Chicken_Filet", "Unbreaded_Chicken_Breast", "Deli_Ham", "Deli_Turkey",
        "Hot_Dogs", "Meatballs", "Nuggets", "Pulled_Pork", "Steak",
    ],
    "nonsweet_dairy": [
        "Butter", "Cream_Cheese", "Sour_Cream", "Creamer", "Milk", "Barista_Milk",
    ],
    "cheese": ["Cheddar_Cheese", "Mozzarella"],
    "sweet_dairy": ["Ice_Cream_Hard_Serve", "Yogurt"],
}

# Reverse mapping: category -> subset name
_CAT_TO_SUBSET = {}
for subset, cats in CATEGORY_SUBSETS.items():
    for cat in cats:
        _CAT_TO_SUBSET[cat] = subset

# Unit conversions to grams (for nutrition normalization)
UNIT_CONVERSION = {
    "Cholesterol (mg)": 0.001,
    "Sodium (mg)": 0.001,
    "Calcium (mg)": 0.001,
    "Iron (mg)": 0.001,
    "Potassium (mg)": 0.001,
    "Vitamin D (mcg)": 0.000001,
}


# =============================================================================
# Data loading
# =============================================================================

def load_product_labels() -> pd.DataFrame:
    """Load manually curated product labels from shared data.

    Returns DataFrame with columns: category, product_code, product_type,
    has_meat, has_dairy, cleaned_ingredients, is_reference.

    Uses original NECTAR product codes (not anonymized).
    """
    return pd.read_csv(PRODUCT_LABELS_CSV)


def load_nectar_products() -> pd.DataFrame:
    """Load and filter NECTAR products.

    Filters:
        1. Drop Cold_Unbreaded_Chicken_Breast and Tenders categories
        2. Keep only products with 2025/2026 sensory ratings

    Returns DataFrame with one row per (Category, Product_Code).
    Uses original NECTAR product codes (not anonymized).
    """
    ingredients = pd.read_csv(INGREDIENTS_CSV)
    sensory = pd.read_csv(SENSORY_CSV)

    # Drop excluded categories
    ingredients = ingredients[~ingredients["Category"].isin(DROP_CATEGORIES)]

    # Products with 2025/2026 sensory data
    scored = sensory[sensory["year"].isin([2025, 2026])]
    scored_products = set(zip(scored["product_category"], scored["product_code"]))

    # Filter to scored products
    mask = ingredients.apply(
        lambda r: (r["Category"], r["Product_Code"]) in scored_products, axis=1
    )
    products = ingredients[mask].copy()

    # Deduplicate: keep latest year if a product appears in multiple years
    products = products.sort_values("Year", ascending=False)
    products = products.drop_duplicates(subset=["Category", "Product_Code"], keep="first")
    products = products.sort_values(["Category", "Product_Code"]).reset_index(drop=True)

    # Join with manually curated labels for product type and cleaned ingredients
    product_labels = load_product_labels()
    products = products.merge(
        product_labels[["category", "product_code", "product_type", "has_meat",
                        "has_dairy", "cleaned_ingredients", "is_reference"]],
        left_on=["Category", "Product_Code"],
        right_on=["category", "product_code"],
        how="left",
    )
    products.drop(columns=["category", "product_code"], inplace=True)

    # Label plant-based analogs: exclude products with actual meat in meat categories
    # and actual dairy in dairy categories
    meat_cats = {
        "Bacon", "Bratwurst", "Breakfast_Sausages", "Burgers", "Chicken_Strips",
        "Breaded_Chicken_Filet", "Unbreaded_Chicken_Breast", "Deli_Ham", "Deli_Turkey",
        "Hot_Dogs", "Meatballs", "Nuggets", "Pulled_Pork", "Steak",
    }
    dairy_cats = {
        "Butter", "Cream_Cheese", "Sour_Cream", "Creamer", "Milk",
        "Barista_Milk", "Cheddar_Cheese", "Mozzarella", "Ice_Cream_Hard_Serve", "Yogurt",
    }
    products["is_analog"] = ~(
        ((products["Category"].isin(meat_cats)) & (products["has_meat"] == True))
        | ((products["Category"].isin(dairy_cats)) & (products["has_dairy"] == True))
    )

    n_analog = products["is_analog"].sum()
    n_excluded = len(products) - n_analog
    logger.info(
        f"Loaded {len(products)} NECTAR products across "
        f"{products['Category'].nunique()} categories "
        f"({n_analog} analogs, {n_excluded} animal/hybrid excluded from ranking)"
    )
    return products


def compute_mean_similarity() -> Dict[Tuple[str, int], float]:
    """Compute per-product mean similarity from sensory ratings.

    Returns dict mapping (Category, Product_Code) -> mean_similarity.
    """
    sensory = pd.read_csv(SENSORY_CSV)
    sensory = sensory[sensory["similarity"].notna()]
    mean_sim = (
        sensory.groupby(["product_category", "product_code"])["similarity"]
        .mean()
        .reset_index()
    )
    return {
        (row["product_category"], int(row["product_code"])): row["similarity"]
        for _, row in mean_sim.iterrows()
    }


def compute_similarity_stats() -> Dict[Tuple[str, int], Tuple[float, int]]:
    """Compute per-product similarity std and panelist count.

    Used for pair weighting: pairs with small score differences relative to
    panelist variance are noisy and should be downweighted.

    Returns dict mapping (Category, Product_Code) -> (similarity_std, n_panelists).
    """
    sensory = pd.read_csv(SENSORY_CSV)
    sensory = sensory[sensory["similarity"].notna()]
    stats = (
        sensory.groupby(["product_category", "product_code"])["similarity"]
        .agg(["std", "count"])
        .reset_index()
    )
    return {
        (row["product_category"], int(row["product_code"])): (
            float(row["std"]) if not np.isnan(row["std"]) else 0.0,
            int(row["count"]),
        )
        for _, row in stats.iterrows()
    }


# =============================================================================
# Feature extraction (self-contained, no unsupervised imports)
# =============================================================================

def _parse_ingredients(cleaned_ingredients: str) -> List[str]:
    """Parse pipe-separated ingredient list (from nectar_product_labels.csv)."""
    if pd.isna(cleaned_ingredients) or not cleaned_ingredients:
        return []
    return [ing.strip() for ing in cleaned_ingredients.split(" | ") if ing.strip()]


def _inverse_rank_weights(n: int) -> np.ndarray:
    """Zipfian inverse-rank weights: 1/(i+1), normalized to sum to 1."""
    weights = np.array([1.0 / (i + 1) for i in range(n)])
    return weights / weights.sum()


def extract_category_subset(products: pd.DataFrame) -> Dict[Tuple[str, int], np.ndarray]:
    """Extract category subset indicator (4-dim one-hot).

    Encodes which of the 4 category subsets (meat, nonsweet_dairy, cheese,
    sweet_dairy) a product belongs to. This captures the broad food type
    without adding 24 sparse indicators.
    """
    subset_names = list(CATEGORY_SUBSETS.keys())  # consistent ordering
    vectors = {}
    for _, row in products.iterrows():
        cat = row["Category"]
        code = int(row["Product_Code"])
        subset = _CAT_TO_SUBSET.get(cat)
        if subset is None:
            continue
        vec = np.zeros(len(subset_names), dtype=np.float64)
        vec[subset_names.index(subset)] = 1.0
        vectors[(cat, code)] = vec
    return vectors


def extract_nutrition(products: pd.DataFrame) -> Dict[Tuple[str, int], np.ndarray]:
    """Extract unified nutrition features (6-dim) for all products.

    Every product gets the same 6 columns (union of all category-specific
    subsets from the kaggle_tastebench pipeline), normalized to per-100g and
    converted to grams. NaN values are imputed with 0.0 (the model can
    learn to ignore irrelevant columns per category).
    """
    vectors = {}
    for _, row in products.iterrows():
        cat = row["Category"]
        code = int(row["Product_Code"])

        values = []
        for col in NUTRITION_COLUMNS:
            val = row.get(col, np.nan)
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = np.nan
            values.append(val)

        vec = np.array(values, dtype=np.float64)
        if np.isnan(vec).all():
            continue

        # Normalize to per-100g
        serving_size = row.get("Serving Size (g)", np.nan)
        try:
            serving_size = float(serving_size)
        except (ValueError, TypeError):
            serving_size = np.nan
        if not np.isnan(serving_size) and serving_size > 0:
            vec = vec * (100.0 / serving_size)

        # Unit conversions (mg/mcg -> g)
        for i, col in enumerate(NUTRITION_COLUMNS):
            if col in UNIT_CONVERSION:
                vec[i] *= UNIT_CONVERSION[col]

        # Impute remaining NaN with 0
        vec = np.nan_to_num(vec, nan=0.0)

        vectors[(cat, code)] = vec

    return vectors


def extract_category_nutrition(products: pd.DataFrame) -> Dict[Tuple[str, int], np.ndarray]:
    """Extract category-specific nutrition features (matching unsupervised pipeline).

    Each category group uses different nutrition columns:
      - meat: Total Fat, Sodium, Protein, Dietary Fiber (4-dim)
      - nonsweet_dairy: Total Fat, Sodium (2-dim)
      - cheese: Total Fat, Sodium, Total Carbohydrate (3-dim)
      - sweet_dairy: Total Fat, Total Sugars (2-dim)

    Unlike the unified 6-dim nutrition, this matches the unsupervised
    pipeline's category-specific feature selection.
    """
    CAT_NUTR_COLS = {
        "meat": ["Total Fat (g)", "Sodium (mg)", "Protein (g)", "Dietary Fiber (g)"],
        "nonsweet_dairy": ["Total Fat (g)", "Sodium (mg)"],
        "cheese": ["Total Fat (g)", "Sodium (mg)", "Total Carbohydrate (g)"],
        "sweet_dairy": ["Total Fat (g)", "Total Sugars (g)"],
    }

    vectors = {}
    for _, row in products.iterrows():
        cat = row["Category"]
        code = int(row["Product_Code"])
        subset = _CAT_TO_SUBSET.get(cat)
        if subset is None:
            continue

        cols = CAT_NUTR_COLS.get(subset, CAT_NUTR_COLS["meat"])
        values = []
        for col in cols:
            val = row.get(col, np.nan)
            try:
                val = float(val)
            except (ValueError, TypeError):
                val = np.nan
            values.append(val)

        vec = np.array(values, dtype=np.float64)

        # Normalize to per-100g
        serving_size = row.get("Serving Size (g)", np.nan)
        try:
            serving_size = float(serving_size)
        except (ValueError, TypeError):
            serving_size = np.nan
        if not np.isnan(serving_size) and serving_size > 0:
            vec = vec * (100.0 / serving_size)

        # Unit conversions (mg -> g)
        for i, col in enumerate(cols):
            if col in UNIT_CONVERSION:
                vec[i] *= UNIT_CONVERSION[col]

        vec = np.nan_to_num(vec, nan=0.0)
        vectors[(cat, code)] = vec

    return vectors


def _extract_compound_generic(
    products: pd.DataFrame,
    embeddings_path: Path,
    source_name: str,
    product_agg: str,  # "top3" or "weighted_average"
    ingredient_agg: str = "weighted_average",
) -> Dict[Tuple[str, int], np.ndarray]:
    """Shared SMILES→embedding→aggregate pipeline for compound features.

    Args:
        embeddings_path: pickle with SMILES → embedding dict.
        source_name: short name for log messages (e.g. "FART").
        product_agg: "top3" (mean of first 3 ingredients by FDA order) or
                     "weighted_average" (inverse-rank weighted over all).
        ingredient_agg: "weighted_average" (concentration-weighted, default),
                        "mean", "max", "log_weighted_average", "top3_by_conc".
    """
    from compound_mapping.food_atlas import FoodAtlasMapper
    from compound_mapping.smiles_resolver import SMILESResolver
    from compound_mapping.foodb_concentrations import FooDBConcentrations

    logger.info(f"Loading compound mapping caches ({source_name})...")
    # Compound-mapping knobs (env-var overridable; defaults reproduce
    # the canonical paper run, BT+Gemini NNLS = 0.6829):
    #   FOOD_ATLAS_CONC_UNITS — conc_unit allowlist for v4.0 attestations
    #     (default "mg/100g,%"); excludes v4.0's "ions"/"molecules"
    #     units that aren't commensurate with mg/100g.
    #   FOOD_ATLAS_SOURCE_BLACKLIST — attestation sources to drop
    #     (default = the three lit2kg gpt-extracted-from-PubMed sources).
    #   FOOD_ATLAS_DISABLE_V40_NATIVE — bypass INGREDIENT_SYNONYMS_V40 +
    #     ENTRY_PAIRS_V40.
    #   FOOD_ATLAS_SYN_HYDRATE_DIR — optional path to a separately-
    #     downloaded v3.2 bundle; if set, its lookup tables are joined on
    #     entity_id to recover plural / case synonyms that v4.0 dropped.
    syn_dir = os.environ.get("FOOD_ATLAS_SYN_HYDRATE_DIR") or None
    units_env = os.environ.get("FOOD_ATLAS_CONC_UNITS", "mg/100g,%")
    conc_units = (
        {u.strip() for u in units_env.split(",") if u.strip()}
        if units_env else None
    )
    disable_v40 = os.environ.get("FOOD_ATLAS_DISABLE_V40_NATIVE", "").lower() in ("1", "true", "yes")
    flavor_filter = os.environ.get("FOOD_ATLAS_FLAVOR_FILTER", "").lower() in ("1", "true", "yes")
    include_amb = os.environ.get("FOOD_ATLAS_INCLUDE_AMBIGUOUS", "").lower() in ("1", "true", "yes")
    src_blacklist_env = os.environ.get(
        "FOOD_ATLAS_SOURCE_BLACKLIST",
        "lit2kg:gpt-5.2,lit2kg:gpt-4,lit2kg:gpt-3.5-finetuned",
    )
    src_blacklist = (
        {s.strip() for s in src_blacklist_env.split(",") if s.strip()}
        if src_blacklist_env else None
    )
    fs_min_env = os.environ.get("FOOD_ATLAS_LIT2KG_FILTER_SCORE_MIN")
    fs_min = float(fs_min_env) if fs_min_env else None
    soft_quality = os.environ.get("FOOD_ATLAS_SOFT_QUALITY_WEIGHT", "").lower() in ("1", "true", "yes")
    mapper = FoodAtlasMapper(
        str(FOOD_ATLAS_DIR),
        synonym_hydration_dir=syn_dir,
        conc_unit_allowlist=conc_units,
        disable_v40_native_dict=disable_v40,
        flavor_descriptor_filter=flavor_filter,
        include_ambiguous_attestations=include_amb,
        attestation_source_blacklist=src_blacklist,
        lit2kg_filter_score_min=fs_min,
        soft_quality_weight=soft_quality,
    )
    smiles_resolver = SMILESResolver(str(SMILES_CACHE), cache_only=True)

    with open(embeddings_path, "rb") as f:
        embeddings: Dict[str, np.ndarray] = pickle.load(f)
    logger.info(f"Loaded {len(embeddings)} {source_name} embeddings")

    foodb_conc = None
    if FOODB_DIR.exists():
        foodb_conc = FooDBConcentrations(str(FOODB_DIR))

    def resolve_smiles(compound) -> Optional[str]:
        if compound.pubchem_cid is not None:
            s = smiles_resolver.resolve(compound.pubchem_cid)
            if s:
                return s
        if compound.chebi_id is not None:
            s = smiles_resolver.resolve_chebi(compound.chebi_id)
            if s:
                return s
        return None

    def aggregate_compounds(compound_embs):
        """Aggregate a list of (embedding, concentration) into a single vec."""
        embs = np.array([e for e, _ in compound_embs])
        concs = np.array([c for _, c in compound_embs])
        if ingredient_agg == "mean":
            return np.mean(embs, axis=0)
        elif ingredient_agg == "max":
            return np.max(embs, axis=0)
        elif ingredient_agg == "log_weighted_average":
            w = np.array([np.log1p(max(c, 0.0)) + 1e-6 for c in concs])
            w = w / w.sum()
            return np.average(embs, axis=0, weights=w)
        elif ingredient_agg == "top3_by_conc":
            order = np.argsort(-concs)  # descending
            top = embs[order[:3]]
            return np.mean(top, axis=0)
        else:  # weighted_average (linear concentration)
            # Guard against all-zero concentrations (e.g. all compounds in
            # this ingredient lack a measured concentration in FoodAtlas+FooDB).
            # In that degenerate case fall back to uniform weighting rather
            # than producing 0/0 = NaN.
            total = concs.sum()
            if total > 0:
                w = concs / total
            else:
                w = np.full_like(concs, 1.0 / len(concs))
            return np.average(embs, axis=0, weights=w)

    vectors = {}
    for _, row in products.iterrows():
        cat = row["Category"]
        code = int(row["Product_Code"])
        ing_str = row.get("cleaned_ingredients", "") or ""
        ingredients = _parse_ingredients(ing_str)
        if not ingredients:
            continue

        mappings = mapper.map_product(ing_str)

        # SKIP_INGREDIENTS env var (comma-separated, lowercase) lets cheap
        # experiments drop high-frequency single-compound ingredients
        # (water, salt, sugar, citric acid, etc.) that appear in most
        # products — they contribute a near-constant compound-vector shift
        # without adding between-product discrimination.
        _skip_set = {
            s.strip().lower()
            for s in os.environ.get("SKIP_INGREDIENTS", "").split(",")
            if s.strip()
        }

        ing_embeddings: List[Tuple[str, np.ndarray]] = []
        for ing_name, mapping in mappings.items():
            if mapping is None or not mapping.compounds:
                continue
            if _skip_set and ing_name.strip().lower() in _skip_set:
                continue

            compound_embs = []
            food_entity = mapping.matched_entity_name or ""
            food_entity_clean = food_entity.split(" (+")[0].strip()

            for compound in mapping.compounds:
                smiles = resolve_smiles(compound)
                if smiles is None:
                    continue
                emb = embeddings.get(smiles)
                if emb is None:
                    continue
                conc = compound.concentration
                if conc is None and foodb_conc and smiles:
                    conc = foodb_conc.get_concentration(smiles, food_name=food_entity_clean)
                # Drop compounds without a measured concentration from
                # the concentration-weighted aggregation (default w=0.0).
                # Principled "no-info" prior: if FoodAtlas doesn't ship
                # a concentration for a (food, chemical) edge, we don't
                # fabricate one. Override via DEFAULT_CONC_WEIGHT env var.
                _default_conc_w = float(os.environ.get("DEFAULT_CONC_WEIGHT", "0.0"))
                compound_embs.append((emb, conc if conc is not None and conc > 0 else _default_conc_w))

            if not compound_embs:
                continue

            ing_emb = aggregate_compounds(compound_embs)
            ing_embeddings.append((ing_name, ing_emb))

        if not ing_embeddings:
            continue

        # Sort ingredients by their position in the original ingredient list (FDA order).
        ranked = []
        for name, emb in ing_embeddings:
            try:
                pos = ingredients.index(name)
            except ValueError:
                pos = len(ingredients)
            ranked.append((pos, emb))
        ranked.sort(key=lambda x: x[0])

        if product_agg == "top3":
            top_embs = [emb for _, emb in ranked[:3]]
            vectors[(cat, code)] = np.mean(top_embs, axis=0)
        elif product_agg == "weighted_average":
            embs = np.array([emb for _, emb in ranked])
            weights = np.array([1.0 / (pos + 1) for pos, _ in ranked])
            weights = weights / weights.sum()
            vectors[(cat, code)] = np.average(embs, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown product_agg: {product_agg}")

    return vectors


def extract_compound(products: pd.DataFrame) -> Dict[Tuple[str, int], np.ndarray]:
    """FART compound embeddings, top-3 ingredient aggregation.

    Per-ingredient aggregation via the INGREDIENT_AGG env var. Default
    "log_weighted_average"; alternatives: "weighted_average", "mean",
    "max", "top3_by_conc".
    """
    ing_agg = os.environ.get("INGREDIENT_AGG", "log_weighted_average")
    return _extract_compound_generic(
        products, FART_EMBEDDINGS, "FART", product_agg="top3",
        ingredient_agg=ing_agg,
    )


def extract_compound_weighted(products: pd.DataFrame) -> Dict[Tuple[str, int], np.ndarray]:
    """FART compound embeddings, inverse-rank weighted ingredient aggregation."""
    ing_agg = os.environ.get("INGREDIENT_AGG", "log_weighted_average")
    return _extract_compound_generic(
        products, FART_EMBEDDINGS, "FART", product_agg="weighted_average",
        ingredient_agg=ing_agg,
    )


def extract_text(products: pd.DataFrame) -> Dict[Tuple[str, int], np.ndarray]:
    """Extract ingredient text embeddings using Qwen3-Embedding.

    Embeds the full ingredient list as a single string per product.
    This preserves composition information (which ingredients appear
    together, their ordering by weight, formulation complexity) that
    is lost when averaging individual ingredient embeddings.
    """
    from sentence_transformers import SentenceTransformer

    model_name = "Qwen/Qwen3-Embedding-0.6B"
    model_revision = "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
    instruction = (
        "Instruct: Given a food ingredient list, identify products "
        "with similar taste, texture, and sensory properties\n"
        "Query:"
    )

    logger.info(f"Loading text embedding model: {model_name}")
    model = SentenceTransformer(
        model_name, trust_remote_code=True, revision=model_revision, device="cpu"
    )

    # Build per-product ingredient strings
    product_texts = {}
    for _, row in products.iterrows():
        cat = row["Category"]
        code = int(row["Product_Code"])
        ing_str = row.get("cleaned_ingredients", "") or ""
        if not ing_str or ing_str == "nan":
            continue
        # Convert pipe-separated to comma-separated for natural text
        product_texts[(cat, code)] = ing_str.replace(" | ", ", ")

    # Embed all product ingredient lists
    keys = sorted(product_texts.keys())
    texts = [product_texts[k] for k in keys]
    logger.info(f"Embedding {len(texts)} product ingredient lists...")
    embeddings = model.encode(
        texts, show_progress_bar=False, normalize_embeddings=True, prompt=instruction
    )

    vectors = {k: embeddings[i] for i, k in enumerate(keys)}
    return vectors


def extract_sensory(products: pd.DataFrame) -> Dict[Tuple[str, int], np.ndarray]:
    """Extract sensory description embeddings (128-dim) via per-ingredient pooling.

    For each product, looks up each ingredient's LLM-generated sensory
    description (texture, flavor, aroma, mouthfeel), embeds individually
    with Qwen3-Embedding, and aggregates via inverse-rank weighted mean.

    Uses Matryoshka Representation Learning (MRL) truncation to 128 dims
    to manage dimensionality at n=215.
    """
    from sentence_transformers import SentenceTransformer

    # Load sensory descriptions
    desc_df = pd.read_csv(SENSORY_DESCRIPTIONS)
    desc_map = {
        row["ingredient_name"].strip().lower(): row["description"]
        for _, row in desc_df.iterrows()
        if pd.notna(row["description"])
    }
    logger.info(f"Loaded {len(desc_map)} ingredient sensory descriptions")

    # Load embedding model with MRL truncation
    model_name = "Qwen/Qwen3-Embedding-0.6B"
    model_revision = "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
    instruction = (
        "Instruct: Represent the sensory properties of this food "
        "ingredient for similarity comparison\n"
        "Query:"
    )

    logger.info(f"Loading sensory embedding model: {model_name} (truncate_dim=128)")
    model = SentenceTransformer(
        model_name, trust_remote_code=True, revision=model_revision,
        device="cpu", truncate_dim=128,
    )

    # Pre-embed all ingredient descriptions in one batch
    desc_names = sorted(desc_map.keys())
    desc_texts = [desc_map[name] for name in desc_names]
    logger.info(f"Embedding {len(desc_texts)} ingredient descriptions...")
    desc_embeddings = model.encode(
        desc_texts, show_progress_bar=False, normalize_embeddings=True,
        prompt=instruction,
    )
    # Build lookup: ingredient_name_lower -> embedding
    emb_map = {name: desc_embeddings[i] for i, name in enumerate(desc_names)}

    # Aggregate to product level via inverse-rank weighted mean
    vectors = {}
    n_matched = []
    for _, row in products.iterrows():
        cat = row["Category"]
        code = int(row["Product_Code"])
        ing_str = row.get("cleaned_ingredients", "") or ""
        ingredients = _parse_ingredients(ing_str)
        if not ingredients:
            continue

        # Look up embeddings for each ingredient
        ing_embs = []
        for ing in ingredients:
            emb = emb_map.get(ing.strip().lower())
            if emb is not None:
                ing_embs.append(emb)

        if not ing_embs:
            continue

        n_matched.append(len(ing_embs))

        # Inverse-rank weighting: 1/(rank+1), normalized
        weights = _inverse_rank_weights(len(ing_embs))
        product_emb = np.average(np.array(ing_embs), axis=0, weights=weights)
        # L2 normalize
        norm = np.linalg.norm(product_emb)
        if norm > 0:
            product_emb = product_emb / norm
        vectors[(cat, code)] = product_emb

    if n_matched:
        logger.info(f"Sensory: matched {np.mean(n_matched):.1f} ingredients/product "
                     f"(min={min(n_matched)}, max={max(n_matched)})")
    return vectors


def extract_image(products: pd.DataFrame) -> Dict[Tuple[str, int], np.ndarray]:
    """Extract DINOv3 image embeddings from raw product images.

    Images are at data/product_images/cropped/{year}/{category}/{code}/*.jpg

    Some image directory names differ from NECTAR category names
    (e.g., Butter -> Salted_Butter). The IMAGE_CATEGORY_MAP handles
    these mismatches, matching the mapping used in the unsupervised
    pipeline's add_images_to_dataset.py.
    """
    # Check if any images exist
    if not IMAGE_DIR.exists():
        logger.warning(f"Image directory not found: {IMAGE_DIR}")
        return {}

    image_files = list(IMAGE_DIR.rglob("*.jpg")) + list(IMAGE_DIR.rglob("*.png"))
    if not image_files:
        logger.warning("No image files found — skipping image feature extraction")
        return {}

    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as e:
        logger.warning(f"Image dependencies not available: {e} — skipping")
        return {}

    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    model_revision = "ea8dc2863c51be0a264bab82070e3e8836b02d51"

    logger.info(f"Loading DINOv3 model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name, revision=model_revision)
    dino_model = AutoModel.from_pretrained(model_name, revision=model_revision)
    dino_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_model.to(device)

    vectors = {}
    for _, row in products.iterrows():
        cat = row["Category"]
        code = int(row["Product_Code"])
        year = int(row["Year"])

        # Look for images: {year}/{image_dir_name}/{code}/*.jpg
        # Some categories have different directory names than NECTAR categories
        img_cat = IMAGE_CATEGORY_MAP.get(cat, cat)
        img_dir = IMAGE_DIR / str(year) / img_cat / str(code)
        if not img_dir.exists():
            continue

        img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        if not img_files:
            continue

        # Use first available image
        img_path = img_files[0]
        try:
            img = Image.open(img_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = dino_model(**inputs)
            vec = outputs.last_hidden_state[:, 0].cpu().numpy().flatten()
            vectors[(cat, code)] = vec
        except (OSError, ValueError, RuntimeError) as e:
            logger.warning(f"Failed to extract image for {cat}/{code}: {e}")

    return vectors


# =============================================================================
# Pair generation
# =============================================================================

def generate_pairs(
    products: pd.DataFrame,
    scores: Dict[Tuple[str, int], float],
) -> pd.DataFrame:
    """Generate all scored pairs among plant-based analog products.

    For each category, create all C(n, 2) pairs among analog products that
    have sensory scores. Label each pair with which product has the higher score.
    Only includes meat-free products in meat categories and dairy-free products
    in dairy categories.

    Ties (pairs whose two products share an identical panel-mean similarity
    score) are excluded here because supervised pairwise models need an
    ordered (winner, loser) target. The evaluation metric in
    `evaluation/metrics.py` operates on the full set of within-category
    combinations and scores ties as 0.5, so these tied pairs still count
    toward the reported pairwise-accuracy denominator. This means
    `len(pairs.csv)` is the supervised-training pair count and is at most
    sum(C(n_c, 2)) (the evaluation pair count, e.g., 935 in the canonical
    NECTAR setup), with the difference equal to the number of tied pairs
    (3 in the canonical setup -> 932 training pairs, 935 evaluation pairs).
    """
    analogs = products[products["is_analog"]]
    all_pairs = []
    pair_id = 0

    for cat, group in analogs.groupby("Category"):
        codes = sorted(group["Product_Code"].unique())
        # Keep only products with scores
        scored_codes = [c for c in codes if (cat, c) in scores]

        for c1, c2 in combinations(scored_codes, 2):
            s1 = scores[(cat, c1)]
            s2 = scores[(cat, c2)]

            if abs(s1 - s2) < 1e-10:
                continue  # Skip ties

            higher = c1 if s1 > s2 else c2
            all_pairs.append({
                "pair_id": pair_id,
                "category": cat,
                "product_code_1": c1,
                "product_code_2": c2,
                "higher_rated_product": higher,
            })
            pair_id += 1

    pairs_df = pd.DataFrame(all_pairs)
    # Count what would-be pairs were skipped as ties for the log.
    n_train = len(pairs_df)
    n_eval = sum(
        len(list(combinations(
            [c for c in sorted(group["Product_Code"].unique())
             if (cat, c) in scores], 2)))
        for cat, group in analogs.groupby("Category")
    )
    n_ties = n_eval - n_train
    logger.info(
        f"Generated {n_train} scored training pairs across "
        f"{products['Category'].nunique()} categories "
        f"({n_eval} within-category evaluation pairs total; "
        f"{n_ties} excluded as ties)"
    )
    return pairs_df


# =============================================================================
# Main
# =============================================================================

def main():
    logger.info("=== Supervised Data Preparation ===")
    logger.info(f"Data source: {DATA_DIR}")
    logger.info(f"Shared caches: {SHARED_DIR}")

    # Load NECTAR products and scores
    products = load_nectar_products()
    scores = compute_mean_similarity()
    similarity_stats = compute_similarity_stats()
    logger.info(f"Products with sensory scores: {len(scores)}")
    logger.info(f"Products with similarity stats: {len(similarity_stats)}")

    # Extract features for ALL 247 NECTAR products (including references)
    # References are useful as anchor points even though they're excluded from pairs
    logger.info(f"Extracting features for all {len(products)} products...")

    logger.info("--- Extracting category subset features ---")
    category_subset = extract_category_subset(products)
    logger.info(f"Category subset: {len(category_subset)} products (4-dim one-hot)")

    logger.info("--- Extracting nutrition features ---")
    nutrition = extract_nutrition(products)
    logger.info(f"Nutrition: {len(nutrition)} products")

    logger.info("--- Extracting category-specific nutrition features ---")
    category_nutrition = extract_category_nutrition(products)
    logger.info(f"Category nutrition: {len(category_nutrition)} products")

    logger.info("--- Extracting compound features ---")
    compound = extract_compound(products)
    logger.info(f"Compound: {len(compound)} products")

    logger.info("--- Extracting compound features (weighted avg) ---")
    compound_weighted = extract_compound_weighted(products)
    logger.info(f"Compound (weighted): {len(compound_weighted)} products")

    # Heavy embeddings (text, sensory, image) don't depend on FoodAtlas.
    # REUSE_FEATURES_FROM_PKL skips ~10 min of Qwen + DINOv3 inference by
    # pulling those three vectors from a previously built pkl; only
    # compound and compound_weighted are recomputed.
    reuse_path = os.environ.get("REUSE_FEATURES_FROM_PKL")
    if reuse_path:
        logger.info(f"--- Reusing text/sensory/image from {reuse_path} ---")
        with open(reuse_path, "rb") as f:
            reused = pickle.load(f)
        text = {k: v["text"] for k, v in reused.items() if v.get("text") is not None}
        sensory = {k: v["sensory"] for k, v in reused.items() if v.get("sensory") is not None}
        image = {k: v["image"] for k, v in reused.items() if v.get("image") is not None}
        logger.info(f"Reused: text={len(text)}, sensory={len(sensory)}, image={len(image)}")
    else:
        logger.info("--- Extracting text features ---")
        text = extract_text(products)
        logger.info(f"Text: {len(text)} products")

        logger.info("--- Extracting sensory features ---")
        sensory = extract_sensory(products)
        logger.info(f"Sensory: {len(sensory)} products")

        logger.info("--- Extracting image features ---")
        image = extract_image(products)
        logger.info(f"Image: {len(image)} products")

    # Build per-product feature dict keyed by (Category, Product_Code)
    # Includes all 247 products; is_analog flag marks which are used for ranking
    product_features = {}
    for _, row in products.iterrows():
        cat = row["Category"]
        code = int(row["Product_Code"])
        key = (cat, code)
        product_features[key] = {
            "category": cat,
            "product_code": code,
            "mean_similarity": scores.get(key, np.nan),
            "is_analog": bool(row["is_analog"]),
            "is_reference": bool(row.get("is_reference", False)),
            "product_type": row.get("product_type", "unknown"),
            "category_subset": category_subset.get(key),
            "nutrition": nutrition.get(key),
            "category_nutrition": category_nutrition.get(key),
            "compound": compound.get(key),
            "compound_weighted": compound_weighted.get(key),
            "text": text.get(key),
            "sensory": sensory.get(key),
            "image": image.get(key),
            "similarity_std": similarity_stats.get(key, (0.0, 0))[0],
            "n_panelists": similarity_stats.get(key, (0.0, 0))[1],
        }

    # Log feature dimensions
    for feat_name in ["category_subset", "nutrition", "category_nutrition", "compound", "compound_weighted", "text", "sensory", "image"]:
        dims = [
            pf[feat_name].shape[0]
            for pf in product_features.values()
            if pf[feat_name] is not None
        ]
        if dims:
            logger.info(f"Feature '{feat_name}': {len(dims)} products, {dims[0]}-dim")
        else:
            logger.info(f"Feature '{feat_name}': 0 products (not available)")

    # Save features. Default (FART) writes product_features.pkl; runs with
    # EMBEDDINGS_CACHE_PATH (e.g. taste_gnn) get a suffixed filename to avoid
    # colliding with the canonical FART pickle that the paper pipeline consumes.
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    # Allow env-var override so parallel v4 variants can each write to a
    # distinct pkl without clobbering each other.
    if os.environ.get("PRODUCT_FEATURES_OUT"):
        features_path = Path(os.environ["PRODUCT_FEATURES_OUT"])
    elif os.environ.get("EMBEDDINGS_CACHE_PATH"):
        _suffix = FART_EMBEDDINGS.stem.replace("_compound_embeddings", "")
        features_path = DATA_OUT / f"product_features_{_suffix}.pkl"
    else:
        features_path = DATA_OUT / "product_features.pkl"
    with open(features_path, "wb") as f:
        pickle.dump(product_features, f)
    logger.info(f"Saved product features to {features_path}")

    # Generate and save pairs
    pairs = generate_pairs(products, scores)
    pairs_path = DATA_OUT / "pairs.csv"
    pairs.to_csv(pairs_path, index=False)
    logger.info(f"Saved pairs to {pairs_path}")

    # Summary
    logger.info("=== Summary ===")
    logger.info(f"Products: {len(product_features)}")
    logger.info(f"Training pairs: {len(pairs)}  (saved to pairs.csv; ties excluded)")
    logger.info(f"Categories: {products['Category'].nunique()}")
    n_complete = sum(
        1 for p in product_features.values()
        if all(p[k] is not None for k in ["nutrition", "compound", "text", "image"])
    )
    logger.info(f"Products with all 4 features: {n_complete}/{len(product_features)}")


if __name__ == "__main__":
    main()
