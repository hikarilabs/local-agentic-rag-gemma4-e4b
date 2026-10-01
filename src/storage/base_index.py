from abc import ABC, abstractmethod
from pathlib import Path

from src.local_agent_harness.harness_config import LocalModelConfig
from src.local_agent_harness.local_config import LocalEmbeddingModel


class BaseIndex(ABC):
    """
    Abstract vector index contract.
    Any backend (FAISS, Chroma, Pinecone, etc.) must implement all three
    capability groups: build, persistence, and search.

    Search results must always be a list of dicts with the shape:
        {
            "score":    float,
            "text":     str,
            "metadata": dict  # must contain at least "source" and "pages"
        }
    """

    # -- build ----------------------------------------------------------------- #

    @classmethod
    @abstractmethod
    async def build(
        cls,
        pdf_path: Path,
        embedding_model: LocalEmbeddingModel,
        max_tokens: int = 480,
        overlap_sentences: int = 1,
    ) -> "BaseIndex":
        """Builds the index from a PDF file using the given embedding model."""
        ...

    # -- persistence ----------------------------------------------------------- #

    @abstractmethod
    def save(self, store: str) -> None:
        """Persists the index and its metadata to the given directory path."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, store: str, model_config: LocalModelConfig) -> "BaseIndex":
        """Loads a previously saved index from the given directory path."""
        ...

    # -- search ---------------------------------------------------------------- #

    @abstractmethod
    async def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Searches the index for the top-k most relevant chunks.

        Returns a list of dicts with keys: "score", "text", "metadata".
        """
        ...
