import os

from sentence_transformers import SentenceTransformer
from huggingface_hub.errors import LocalEntryNotFoundError

from src.embeddings.base_embedding_engine import BaseEmbeddingsEngine, TaskType
from src.embeddings.engine_registry import EmbeddingsEngineRegistry


class GemmaEngine(BaseEmbeddingsEngine):
    PREFIX_MAP: dict[str, str] = {
        "query": "query: ",
        "document": "document: ",
        "symmetric": "search_query: ",
    }

    def __init__(self, embedding_model: str):
        super().__init__(embedding_model)
        device = self.determine_device()

        if device == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        try:
            self.client = SentenceTransformer(
                embedding_model, local_files_only=True
            ).to(device)
        except (LocalEntryNotFoundError, ValueError, OSError):
            self.client = SentenceTransformer(
                embedding_model,
                local_files_only=False,
            ).to(device)

    def prepare(self, text: list[str], task_type: TaskType) -> list[str]:
        # Keeps Gemma prefix quirks isolated here
        prefix = self.PREFIX_MAP.get(task_type, "")
        return [f"{prefix}{v}" for v in text]

    def embed(self, formatted_inputs: list[str]) -> list[list[float]]:
        return self.client.encode(
            formatted_inputs,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        ).tolist()

    def count_tokens(self, text: str) -> int:
        return len(self.client.tokenizer.encode(text, add_special_tokens=False))

    def calculate_payload_tokens(self, text: str, task_type: TaskType) -> int:
        formatted_text = self.prepare([text], task_type)[0]

        # Encode with special tokens. It mirrors exactly what is actually sent to the LLM
        full_tokens = self.client.tokenizer.encode(
            formatted_text, add_special_tokens=True
        )

        return len(full_tokens)


EmbeddingsEngineRegistry.register("gemma", GemmaEngine)
