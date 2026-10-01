from typing import Type, Dict

from src.embeddings.base_embedding_engine import BaseEmbeddingsEngine


class EmbeddingsEngineRegistry:
    """Maps model name patterns to their concrete engine classes"""

    _registry: Dict[str, Type[BaseEmbeddingsEngine]] = {}

    @classmethod
    def register(cls, key: str, engine_class: Type[BaseEmbeddingsEngine]) -> None:
        cls._registry[key.lower()] = engine_class

    @classmethod
    def resolve(cls, model_name: str) -> BaseEmbeddingsEngine:
        model_lower = model_name.lower()

        # Check for specialized overrides (like "gemma" or "minilm")
        for key, engine_class in cls._registry.items():
            if key in model_lower:
                return engine_class(model_name)

        #  If no specific match, drop down to the generic fallback engine
        if "default" in cls._registry:
            return cls._registry["default"](model_name)

        raise ValueError(
            f"No engine found for model '{model_name}'. "
            f"Registered: {list(cls._registry.keys())}"
        )
