#!/usr/bin/env python3
"""Generate sensory-functional descriptions for food ingredients.

Calls an LLM via OpenRouter to generate 120-180 word descriptions covering
7 sensory dimensions for each ingredient. Descriptions are optimized for
text embedding models to capture taste, texture, aroma, and functional
similarity between ingredients.

Usage (from ):
    # From product_labels_manually_cleaned.csv (Kaggle challenge)
    python shared/scripts/generate_sensory_descriptions.py \
        --labels unsupervised/predict/data/product_labels_manually_cleaned.csv \
        --output shared/data/caches/sensory_descriptions.csv

    # From any CSV with an ingredient column
    python shared/scripts/generate_sensory_descriptions.py \
        --ingredients-csv path/to/ingredients.csv \
        --ingredient-column "ingredient_name" \
        --output path/to/output.csv

    # Quick test with 5 ingredients
    python shared/scripts/generate_sensory_descriptions.py \
        --labels unsupervised/data/product_labels_manually_cleaned.csv \
        --output /tmp/test_descriptions.csv \
        --sample 5

Requires OPENROUTER_API_KEY in .env or environment.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM prompt for generating sensory-functional ingredient descriptions
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a food scientist specializing in sensory science and product \
formulation. Generate concise, consistent sensory-functional descriptions \
of food ingredients used in food products, including both animal-based and \
plant-based products. These descriptions will be embedded using text embedding \
models to compute sensory similarity between products.

RULES:
1. Write exactly ONE paragraph of 120-180 words.
2. Always cover these dimensions IN THIS ORDER: (a) functional role in the \
product matrix, (b) texture contribution, (c) flavor/taste impact, \
(d) aroma properties, (e) visual appearance contribution, (f) mouthfeel \
characteristics, (g) behavior during cooking/processing, (h) overall \
contribution to the sensory experience of the final product.
3. Use precise sensory vocabulary (e.g., "fibrous," "umami," "glossy," \
"viscous," "Maillard browning").
4. If the ingredient has minimal impact on a dimension, state this briefly \
rather than omitting it.
5. Describe the ingredient's sensory contribution objectively, whether it \
appears in animal-based or plant-based products.
6. Use present tense, third person, declarative sentences throughout.
7. Do not include citations, chemical formulas, or regulatory information.

EXAMPLES:

Methylcellulose:
Methylcellulose acts as a thermogelation binding agent, uniquely gelling upon \
heating rather than cooling, enabling the product to firm up during cooking \
similarly to how animal muscle proteins denature and set. It contributes \
structural cohesion and a slightly elastic, springy texture that prevents \
crumbling during handling and cooking. It is entirely flavor-neutral and \
taste-neutral. Visually transparent, it does not alter color. In the mouth, it \
provides subtle gel-like body that smooths overall chew and enhances succulence \
perception by trapping moisture within the matrix. During cooking, thermal \
gelation triggers between 50-70C, progressively firming the product as internal \
temperature rises. At excessive concentrations it imparts slimy or gummy \
mouthfeel. It provides heat-set binding and shape-retention, contributing \
structural integrity to the final product.

Whole Milk:
Whole milk serves as the primary liquid and fat-carrying medium, providing a \
homogeneous emulsion of butterfat globules, casein proteins, lactose, and \
minerals that defines dairy product character. It contributes smooth, creamy \
texture through its natural fat content and protein network. Its flavor profile \
delivers mild sweetness from lactose with subtle cooked-milk notes when heated, \
and clean dairy richness. It carries volatile compounds that produce the \
characteristic fresh dairy aroma. Visually, it provides opaque white \
appearance from light-scattering fat globules and casein micelles. In the \
mouth, it delivers rich body, fat-coating lubricity, and clean finish without \
astringency. During heating, casein proteins denature and Maillard reactions \
between lactose and proteins produce browning and caramelized flavors. It \
establishes the baseline sensory profile of dairy products including body, \
sweetness, clean flavor, and creamy mouthfeel."""

USER_PROMPT_TEMPLATE = """\
Generate a sensory-functional description for each of the following ingredients. \
Return a JSON object where each key is the ingredient name and the value is the \
description paragraph.

Ingredients:
{ingredients}"""

# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3.1-pro-preview"


def call_openrouter(
    ingredients: list[str],
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_retries: int = 5,
) -> dict[str, str]:
    """Call OpenRouter API to generate descriptions for a batch of ingredients.

    Returns a dict mapping ingredient_name → description.
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(
        ingredients="\n".join(f"- {ing}" for ing in ingredients)
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries):
        try:
            # Shorter timeout on retries to avoid long hangs
            read_timeout = 120 if attempt == 0 else 90
            resp = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=(10, read_timeout),
            )

            if resp.status_code == 429:
                wait = min(2 ** (attempt + 2), 60)
                logger.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            content = data["choices"][0]["message"]["content"]
            if content is None:
                raise ValueError("API returned null content")
            # Strip markdown code fences if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            result = json.loads(content)
            return {str(k): str(v) for k, v in result.items()}

        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            wait = min(2 ** (attempt + 1), 30)
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    logger.error(f"Failed after {max_retries} retries for batch: {ingredients[:3]}... Skipping.")
    return {}


# ---------------------------------------------------------------------------
# Ingredient extraction
# ---------------------------------------------------------------------------

def extract_ingredients_from_labels(labels_path: str) -> list[str]:
    """Extract unique ingredient names from a product_labels_manually_cleaned.csv file."""
    df = pd.read_csv(labels_path)
    all_ings = set()
    for ing_list in df["cleaned_ingredients"].dropna():
        for ing in ing_list.split(" | "):
            ing = ing.strip()
            if ing:
                all_ings.add(ing)
    return sorted(all_ings)


def extract_ingredients_from_csv(csv_path: str, column: str) -> list[str]:
    """Extract unique ingredient names from a generic CSV column."""
    df = pd.read_csv(csv_path)
    return sorted(df[column].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate sensory descriptions for food ingredients via LLM."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--labels",
        help="Path to product_labels_manually_cleaned.csv (extracts from cleaned_ingredients column)",
    )
    input_group.add_argument(
        "--ingredients-csv",
        help="Path to a CSV with an ingredient name column",
    )
    parser.add_argument(
        "--ingredient-column",
        default="ingredient_name",
        help="Column name in --ingredients-csv (default: ingredient_name)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output CSV path (ingredient_name, description)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Ingredients per API call (default: 10)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Only process N ingredients (for testing)",
    )
    args = parser.parse_args()

    # Load API key
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment or .env file")
        sys.exit(1)

    # Extract ingredients
    if args.labels:
        ingredients = extract_ingredients_from_labels(args.labels)
    else:
        ingredients = extract_ingredients_from_csv(args.ingredients_csv, args.ingredient_column)

    logger.info(f"Found {len(ingredients)} unique ingredients")

    # Load existing cache (for resumability)
    output_path = Path(args.output)
    existing: dict[str, str] = {}
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        for _, row in existing_df.iterrows():
            name = str(row["ingredient_name"]).strip()
            desc = str(row["description"]).strip()
            if name and desc and desc != "nan":
                existing[name] = desc
        logger.info(f"Found {len(existing)} existing descriptions in cache")

    # Filter out already-described ingredients
    remaining = [ing for ing in ingredients if ing not in existing]

    if args.sample:
        remaining = remaining[:args.sample]

    logger.info(f"{len(remaining)} ingredients to generate ({len(existing)} cached)")

    if not remaining:
        logger.info("All ingredients already have descriptions. Nothing to do.")
        return

    # Generate in batches
    results = dict(existing)
    batch_size = args.batch_size

    for i in range(0, len(remaining), batch_size):
        batch = remaining[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(remaining) + batch_size - 1) // batch_size

        logger.info(
            f"Batch {batch_num}/{total_batches}: {len(batch)} ingredients "
            f"({batch[0]!r} ... {batch[-1]!r})"
        )

        batch_results = call_openrouter(batch, api_key, model=args.model)

        # Match results back to original ingredient names
        # (LLM may return slightly different casing)
        batch_lower = {name.lower(): name for name in batch}
        for key, desc in batch_results.items():
            original_name = batch_lower.get(key.lower(), key)
            results[original_name] = desc

        # Save after each batch (incremental)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df = pd.DataFrame(
            [{"ingredient_name": k, "description": v} for k, v in sorted(results.items())]
        )
        out_df.to_csv(output_path, index=False)

        generated_this_batch = len(batch_results)
        missing_this_batch = len(batch) - generated_this_batch
        if missing_this_batch > 0:
            missing_names = [b for b in batch if b.lower() not in {k.lower() for k in batch_results}]
            logger.warning(f"  Missing {missing_this_batch} descriptions: {missing_names}")

        # Rate limit courtesy
        if i + batch_size < len(remaining):
            time.sleep(1)

    total_described = sum(1 for ing in ingredients if ing in results)
    logger.info(
        f"Done. {total_described}/{len(ingredients)} ingredients with descriptions. "
        f"Saved to {output_path}"
    )


if __name__ == "__main__":
    main()
