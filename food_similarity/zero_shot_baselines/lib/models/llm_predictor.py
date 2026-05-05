"""LLM pairwise ranking model via OpenRouter API.

Queries an LLM to predict which of two plant-based products is more similar
to its animal-based counterpart. Supports ablation over prompt features
(ingredients, nutrition, image) and multimodal models.

Saves results via run.py into:
  results/llm/{model}/submissions/{features}.csv — submission
  results/llm/{model}/logs/{features}.csv        — full response log
  results/llm/{model}/configs/{features}.yaml    — config copy
"""

import base64
import hashlib
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from ..features.base import BaseFeature
from .base import BaseModel

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# All 17 nutrition columns in products.csv
NUTRITION_COLUMNS = [
    "Calories", "Total Fat (g)", "Saturated Fat (g)", "Trans Fat (g)",
    "Polyunsaturated Fat (g)", "Monounsaturated Fat (g)", "Cholesterol (mg)",
    "Sodium (mg)", "Total Carbohydrate (g)", "Dietary Fiber (g)",
    "Total Sugars (g)", "Added Sugars (g)", "Protein (g)",
    "Vitamin D (mcg)", "Calcium (mg)", "Iron (mg)", "Potassium (mg)",
]


# ============================================================================
# Prompt templates
# ============================================================================

_SYSTEM_PROMPT = """<role>
You are an expert food scientist. Your task is to predict which of two plant-based products would be perceived as more similar to its animal-based counterpart in a blind taste test.
</role>

<output_format>
Respond with ONLY the number 1 or 2. No other text.
</output_format>"""

_INGREDIENTS_BLOCK = """<ingredients order="descending_by_weight">
{ingredient_list}
</ingredients>"""

_NUTRITION_BLOCK = """<nutrition unit="per_100g">
{nutrition_facts}
</nutrition>"""

_IMAGE_BLOCKS = {
    "both": "<images>\nThe first image shows Product 1 and the second image shows Product 2.\n</images>",
    "product_1_only": "<images>\nAn image is provided for Product 1 only. No image is available for Product 2.\n</images>",
    "product_2_only": "<images>\nAn image is provided for Product 2 only. No image is available for Product 1.\n</images>",
}


# ============================================================================
# Helpers
# ============================================================================

def _deterministic_swap(code1: int, code2: int, seed: int) -> bool:
    """Deterministic swap decision based on product codes and seed.

    Uses MD5 hash so the swap is independent of random state,
    code changes, or pair processing order.
    """
    h = hashlib.md5(f"{code1}:{code2}:{seed}".encode()).hexdigest()
    return int(h, 16) % 2 == 0


def _deterministic_fallback(code1: int, code2: int, seed: int) -> str:
    """Deterministic fallback prediction when LLM fails.

    Uses a salted MD5 hash (independent of the swap hash) so that
    failures produce a reproducible coin flip rather than always "1".
    """
    h = hashlib.md5(f"fallback:{code1}:{code2}:{seed}".encode()).hexdigest()
    return "2" if int(h, 16) % 2 == 0 else "1"


def _parse_response(text: str) -> Optional[str]:
    """Extract '1' or '2' from LLM output.

    Checks the last non-empty line first (where the prompt instructs the
    model to place the answer), then falls back to the first line.
    Scans each line from the end to find the last '1' or '2', avoiding
    stray digits earlier in a reasoning sentence.
    """
    if not text or not text.strip():
        return None
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return None
    # Check last line first (expected answer location), then first line
    for line in [lines[-1], lines[0]]:
        for char in reversed(line):
            if char in ("1", "2"):
                return char
    return None


def _backoff_delay(attempt: int, is_rate_limit: bool = False) -> float:
    """Exponential backoff with jitter."""
    delay = min(1.0 * (2 ** attempt), 60.0)
    delay += random.uniform(0, 0.3 * delay)
    if is_rate_limit:
        delay = max(delay, 10.0)
    return delay


def _encode_image(path: Path) -> str:
    """Encode image to base64 data URL."""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode('utf-8')}"


def _format_nutrition(row: pd.Series) -> str:
    """Format nutrition values as a human-readable per-100g string.

    Returns empty string if serving size is missing (cannot normalize).
    """
    serving = row.get("Serving Size (g)", np.nan)
    try:
        serving = float(serving)
    except (ValueError, TypeError):
        serving = np.nan

    if np.isnan(serving) or serving <= 0:
        return ""

    parts = []
    for col in NUTRITION_COLUMNS:
        try:
            val = float(row.get(col, np.nan))
        except (ValueError, TypeError):
            continue
        if pd.isna(val) or val == 0:
            continue
        val = val * (100.0 / serving)
        if "(" in col:
            name, unit = col.split("(")[0].strip(), col.split("(")[1].rstrip(")")
            parts.append(f"{name}: {val:.1f}{unit}")
        else:
            parts.append(f"{col}: {val:.1f}")
    return ", ".join(parts)


# ============================================================================
# Model
# ============================================================================

class LLMPredictor(BaseModel):
    """LLM-based pairwise ranking model.

    Queries an LLM via OpenRouter to compare products pairwise.
    Supports ablation over prompt features: ingredients, nutrition, image.
    Logs raw responses for reproducibility audit.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.model_name: str = config.get("model_name", "qwen/qwen3.5-397b-a17b")
        self.prompt_features: frozenset = frozenset(config.get("prompt_features", ["ingredients", "nutrition"]))
        self.temperature: float = config.get("temperature", 0.0)
        self.reasoning_enabled: bool = config.get("reasoning_enabled", False)
        self.max_retries: int = config.get("max_retries", 5)
        self.timeout_seconds: int = config.get("timeout_seconds", 30)
        self.timeout_reasoning_seconds: int = config.get("timeout_reasoning_seconds", 120)
        self.random_seed: int = config.get("random_seed", 42)
        self.max_workers: int = config.get("max_workers", 8)

        self._api_key = os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Set it in a .env file or export it in your shell."
            )

        logger.info(
            f"LLMPredictor: model={self.model_name}, "
            f"features={sorted(self.prompt_features)}, "
            f"reasoning={self.reasoning_enabled}"
        )

        # Populated during fit
        self._product_info: Dict[int, dict] = {}
        self._image_base_dir: Optional[Path] = None

        # Response log — populated during predict_pairs, saved by run.py
        self.response_log: List[dict] = []

    def fit(
        self,
        products_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        features: Dict[str, BaseFeature],
    ) -> "LLMPredictor":
        """Pre-build product info lookup from competition data."""
        unsupervised_dir = Path(__file__).resolve().parent.parent.parent
        self._image_base_dir = unsupervised_dir / "data" / "competition"

        code_to_category = dict(
            zip(labels_df["product_code"].astype(int), labels_df["category"])
        )
        code_to_ingredients = dict(
            zip(labels_df["product_code"].astype(int),
                labels_df["cleaned_ingredients"].fillna(""))
        )

        for _, row in products_df.iterrows():
            code = int(row["Product code"])
            image_path_str = row.get("image_path", "")
            image_path = None
            if pd.notna(image_path_str) and image_path_str:
                full_path = self._image_base_dir / image_path_str
                if full_path.exists():
                    image_path = full_path

            self._product_info[code] = {
                "category": code_to_category.get(code, "Unknown"),
                "ingredients": code_to_ingredients.get(code, ""),
                "nutrition": _format_nutrition(row),
                "image_path": image_path,
            }

        logger.info(f"LLMPredictor fitted: {len(self._product_info)} products indexed")
        return self

    def _process_pair(self, row: pd.Series) -> Tuple[dict, dict]:
        """Process a single pair. Returns (result_row, log_entry).

        Thread-safe: reads from self._product_info (immutable after fit)
        and calls self._query_pair (stateless HTTP requests).
        """
        test_id = int(row["test_id"])
        code1 = int(row["product_code_1"])
        code2 = int(row["product_code_2"])
        category = row.get("product_category", "")

        info1 = self._product_info.get(code1)
        info2 = self._product_info.get(code2)

        if info1 is None or info2 is None:
            logger.warning(f"Missing product info for pair {test_id}")
            fallback = _deterministic_fallback(code1, code2, self.random_seed)
            winner = code2 if fallback == "2" else code1
            return (
                {"test_id": test_id, "higher_rated_product": winner},
                {
                    "test_id": test_id, "category": category,
                    "product_code_1": code1, "product_code_2": code2,
                    "prediction": fallback, "raw_response": "", "reasoning": "",
                    "status": "MISSING_PRODUCT", "swapped": False,
                    "elapsed_seconds": 0.0, "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
            )

        # Check if the prompt would contain any actual data for the model.
        # If not (e.g., image-only config but neither product has an image),
        # use a deterministic fallback instead of wasting an API call.
        has_data = False
        if "ingredients" in self.prompt_features:
            has_data |= bool(info1["ingredients"] or info2["ingredients"])
        if "nutrition" in self.prompt_features:
            has_data |= bool(info1["nutrition"] or info2["nutrition"])
        if "image" in self.prompt_features:
            has_data |= bool(info1.get("image_path") or info2.get("image_path"))

        if not has_data:
            fallback = _deterministic_fallback(code1, code2, self.random_seed)
            winner = code2 if fallback == "2" else code1
            return (
                {"test_id": test_id, "higher_rated_product": winner},
                {
                    "test_id": test_id, "category": category,
                    "product_code_1": code1, "product_code_2": code2,
                    "prediction": fallback, "raw_response": "", "reasoning": "",
                    "status": "NO_DATA", "swapped": False,
                    "elapsed_seconds": 0.0, "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
            )

        # Anti-bias: deterministic swap based on product codes + seed
        swapped = _deterministic_swap(code1, code2, self.random_seed)
        if swapped:
            prediction, raw_response, reasoning, status, elapsed, usage = (
                self._query_pair(info2, info1, code1, code2)
            )
            if prediction == "1":
                prediction = "2"
            elif prediction == "2":
                prediction = "1"
        else:
            prediction, raw_response, reasoning, status, elapsed, usage = (
                self._query_pair(info1, info2, code1, code2)
            )

        winner = code2 if prediction == "2" else code1
        return (
            {"test_id": test_id, "higher_rated_product": winner},
            {
                "test_id": test_id, "category": category,
                "product_code_1": code1, "product_code_2": code2,
                "prediction": prediction, "raw_response": raw_response,
                "reasoning": reasoning, "status": status, "swapped": swapped,
                "elapsed_seconds": round(elapsed, 2),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
            },
        )

    def predict_pairs(
        self, pairs_df: pd.DataFrame, on_checkpoint=None,
    ) -> pd.DataFrame:
        """Query LLM for each pair. Populates self.response_log for audit.

        Args:
            pairs_df: Test pairs to predict.
            on_checkpoint: Optional callback(results, log_entries) called every
                100 completed pairs to allow periodic saving. Both lists may
                contain None entries for pairs not yet completed.
        """
        total = len(pairs_df)
        results = [None] * total
        log_entries = [None] * total
        completed = 0

        rows = [(i, row) for i, (_, row) in enumerate(pairs_df.iterrows())]

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._process_pair, row): idx
                for idx, row in rows
            }
            for future in as_completed(futures):
                idx = futures[future]
                result_row, log_entry = future.result()
                results[idx] = result_row
                log_entries[idx] = log_entry
                completed += 1

                if completed % 100 == 0 or completed == total:
                    done = [e for e in log_entries if e is not None]
                    total_prompt = sum(e["prompt_tokens"] for e in done)
                    total_completion = sum(e["completion_tokens"] for e in done)
                    ok_count = sum(1 for e in done if e["status"] == "OK")
                    logger.info(
                        f"Progress: {completed}/{total} pairs | "
                        f"{ok_count} OK | "
                        f"{total_prompt:,} prompt + {total_completion:,} completion tokens"
                    )
                    if on_checkpoint:
                        self.response_log = [e for e in log_entries if e is not None]
                        on_checkpoint(
                            [r for r in results if r is not None],
                            self.response_log,
                        )

        self.response_log = log_entries
        return pd.DataFrame(results)

    def _query_pair(
        self, info1: dict, info2: dict, code1: int, code2: int,
    ) -> tuple:
        """Query LLM for one pair.

        Returns:
            (prediction, raw_response, reasoning, status, elapsed, usage) where:
            - prediction: '1' or '2'
            - raw_response: full content text from LLM (or error message)
            - reasoning: reasoning text from LLM (empty if not available)
            - status: 'OK', 'PARSE_FAIL', 'API_FAIL', or 'TIMEOUT'
            - elapsed: wall-clock seconds for all attempts
            - usage: token usage dict from the API (empty dict on failure)
        """
        fallback = _deterministic_fallback(code1, code2, self.random_seed)

        # Resolve images
        image_urls, image_status = [], "none"
        if "image" in self.prompt_features:
            img1, img2 = info1.get("image_path"), info2.get("image_path")
            if img1 and img2:
                image_urls = [_encode_image(img1), _encode_image(img2)]
                image_status = "both"
            elif img1:
                image_urls = [_encode_image(img1)]
                image_status = "product_1_only"
            elif img2:
                image_urls = [_encode_image(img2)]
                image_status = "product_2_only"

        system_prompt, user_prompt = self._build_prompt(info1, info2, image_status)

        # Retry loop
        last_response = ""
        last_reasoning = ""
        last_status = "API_FAIL"
        last_usage = {}
        status_code = None
        t0 = time.time()

        for attempt in range(self.max_retries + 1):
            try:
                response = self._call_api(system_prompt, user_prompt, image_urls)

                if "choices" not in response or not response["choices"]:
                    logger.warning(f"Invalid API response (attempt {attempt + 1})")
                    last_status = "API_FAIL"
                    if attempt < self.max_retries:
                        time.sleep(_backoff_delay(attempt))
                        continue
                    elapsed = time.time() - t0
                    return fallback, last_response, last_reasoning, last_status, elapsed, last_usage

                message = response["choices"][0]["message"]
                content = message.get("content") or ""
                reasoning = message.get("reasoning", "") or ""
                last_response = content
                last_reasoning = reasoning
                last_usage = response.get("usage", {})

                # Try parsing from content first, fall back to reasoning
                parsed = _parse_response(content)
                if parsed is None and reasoning:
                    parsed = _parse_response(reasoning)

                if parsed in ("1", "2"):
                    elapsed = time.time() - t0
                    return parsed, content, reasoning, "OK", elapsed, last_usage

                logger.warning(f"Unparseable response (attempt {attempt + 1}): content='{content}', reasoning='{reasoning[:100]}'")
                last_status = "PARSE_FAIL"

            except requests.exceptions.Timeout:
                logger.error(f"Timeout (attempt {attempt + 1})")
                last_response = "TIMEOUT"
                last_status = "TIMEOUT"

            except requests.exceptions.RequestException as e:
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code == 404:
                    logger.error(f"Model '{self.model_name}' not found (404)")
                    elapsed = time.time() - t0
                    return fallback, f"404: {e}", "", "API_FAIL", elapsed, {}
                logger.error(f"API error (attempt {attempt + 1}): {e}")
                last_response = str(e)
                last_status = "API_FAIL"

            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
                last_response = str(e)
                last_status = "API_FAIL"

            if attempt < self.max_retries:
                time.sleep(_backoff_delay(attempt, is_rate_limit=(status_code == 429)))

        logger.error(f"All retries exhausted, using deterministic fallback: {fallback}")
        elapsed = time.time() - t0
        return fallback, last_response, last_reasoning, last_status, elapsed, last_usage

    # ------------------------------------------------------------------
    # Prompt builder
    # ------------------------------------------------------------------

    def _build_prompt(self, info1: dict, info2: dict, image_status: str) -> Tuple[str, str]:
        """Compose system and user prompts from feature blocks.

        Returns:
            (system_prompt, user_prompt) tuple.
        """
        category = info1["category"].replace("_", " ")

        names = []
        if "ingredients" in self.prompt_features:
            names.append("the ingredient lists" if len(self.prompt_features) == 1 else "ingredients")
        if "nutrition" in self.prompt_features:
            names.append("nutrition facts")
        if "image" in self.prompt_features and image_status != "none":
            names.append("visual appearance" if image_status == "both" else "visual appearance (when available)")

        if not names:
            consideration = "all available information"
        elif len(names) <= 2:
            consideration = " and ".join(names)
        else:
            consideration = f"{', '.join(names[:-1])}, and {names[-1]}"

        # Build per-product data blocks (omit <data> entirely for image-only)
        product_blocks = []
        for pid, info in [("1", info1), ("2", info2)]:
            inner_parts = []
            if "ingredients" in self.prompt_features:
                inner_parts.append(_INGREDIENTS_BLOCK.format(
                    ingredient_list=info["ingredients"],
                ))
            if "nutrition" in self.prompt_features and info["nutrition"]:
                inner_parts.append(_NUTRITION_BLOCK.format(
                    nutrition_facts=info["nutrition"],
                ))
            if inner_parts:
                product_blocks.append(
                    f'<product id="{pid}">\n' + "\n".join(inner_parts) + "\n</product>"
                )

        parts = [
            f"<context>\nTwo plant-based {category} products are compared for "
            f"similarity to their animal-based {category} counterpart. "
            f"Consider {consideration}.\n</context>",
        ]
        if product_blocks:
            parts.append("<data>\n" + "\n\n".join(product_blocks) + "\n</data>")

        if "image" in self.prompt_features and image_status in _IMAGE_BLOCKS:
            parts.append(_IMAGE_BLOCKS[image_status])

        parts.append(
            f"<task>\nIn a blind taste test, which product would omnivores rank "
            f"as more similar to the animal-based {category}?\n</task>"
        )

        return _SYSTEM_PROMPT, "\n\n".join(parts)

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _call_api(self, system_prompt: str, user_prompt: str, image_urls: List[str]) -> dict:
        """Make HTTP POST to OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
        }

        if image_urls:
            content = [{"type": "text", "text": user_prompt}]
            for url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": url}})
        else:
            content = user_prompt

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "seed": self.random_seed,
        }
        if self.reasoning_enabled:
            payload["reasoning"] = {"effort": "high", "exclude": False}

        timeout = self.timeout_reasoning_seconds if self.reasoning_enabled else self.timeout_seconds
        if image_urls:
            timeout *= 2

        resp = requests.post(OPENROUTER_API_URL, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
