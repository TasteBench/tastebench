"""Model registry.

Adding a new model type:
1. Create a module in this package
2. Implement a class extending BaseModel
3. Register it in MODEL_REGISTRY below
"""

from typing import Dict, Type

from .base import BaseModel
from .distance_predictor import DistancePredictor


def _get_llm_predictor():
    from .llm_predictor import LLMPredictor
    return LLMPredictor


MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "distance_predictor": DistancePredictor,
    "llm_predictor": _get_llm_predictor,  # lazy to avoid requests import
}


def get_model(model_type: str, config: dict) -> BaseModel:
    """Instantiate a model by type name."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    cls = MODEL_REGISTRY[model_type]
    if not isinstance(cls, type):
        cls = cls()
    return cls(config)
