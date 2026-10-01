from abc import ABC, abstractmethod
from typing import Literal

import torch


# Type alias for task options
TaskType = Literal["query", "document", "symmetric"]


class BaseEmbeddingsEngine(ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @staticmethod
    def determine_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @abstractmethod
    def prepare(self, text: list[str], task_type: TaskType) -> list[str]:
        """Handles any model-specific prompt styling or prefix injections"""
        pass

    @abstractmethod
    def embed(self, formatted_inputs: list[str]) -> list[list[float]]:
        """Handles the embedding process"""
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Returns the absolute token count for a string using this model's logic."""
        pass

    @abstractmethod
    def calculate_payload_tokens(self, text: str, task_type: TaskType) -> int:
        """Calculates the total tokens a string consumes, including its prefix and special tokens"""
        pass
