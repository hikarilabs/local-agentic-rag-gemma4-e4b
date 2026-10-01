import os
from sentence_transformers import SentenceTransformer

from src.embeddings.base_embedding_engine import BaseEmbeddingsEngine, TaskType
from src.embeddings.engine_registry import EmbeddingsEngineRegistry


class StandardTransformerEngine(BaseEmbeddingsEngine):
    def __init__(self, embedding_model: str):
        super().__init__(embedding_model)
        device = self.determine_device()

        if device == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        self.client = SentenceTransformer(embedding_model).to(device)

    def prepare(self, text: list[str], task_type: TaskType) -> list[str]:
        return text  # Traditional models get raw text

    def embed(self, formatted_inputs: list[str]) -> list[list[float]]:
        return self.client.encode(
            formatted_inputs, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def count_tokens(self, text: str) -> int:
        tokens = self.client.tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)

    def calculate_payload_tokens(self, text: str, task_type: TaskType) -> int:
        full_tokens = self.client.tokenizer.encode(text, add_special_tokens=True)
        return len(full_tokens)


EmbeddingsEngineRegistry.register("default", StandardTransformerEngine)
