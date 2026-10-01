import importlib
import json
import numpy as np
from pathlib import Path

from src.local_agent_harness.harness_config import LocalModelConfig
from src.local_agent_harness.local_config import LocalEmbeddingModel
from src.processors.pdf.corpus import load_pdf_corpus
from src.processors.pdf.pdf_processor import chunk_continuous_corpus
from src.storage.base_index import BaseIndex


def _require(module_name: str, package_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        raise ImportError(
            f"The package '{module_name}' is required to use this storage index. "
            f"Please run: uv add {package_name}"
        )


class FaissIndex(BaseIndex):
    def __init__(self, embedding_model: LocalEmbeddingModel, index, chunks: list[dict]):
        # Inject our agnostic, decoupled embedding wrapper layer
        self.embedding_model = embedding_model
        self.index = index
        self.chunks = chunks

    # -- build ------------------------------------------------------------- #
    @classmethod
    async def build(
        cls,
        pdf_path: Path,
        embedding_model: LocalEmbeddingModel,
        max_tokens: int = 480,
        overlap_sentences: int = 1,
    ):
        """Builds a FAISS index from a streaming PDF parsing pipeline."""
        faiss = _require("faiss", "faiss-cpu")

        print(f"Reading and mapping stream context from {pdf_path} ...")
        # 1. Load PDF using the continuous offset-aware character mapping strategy
        corpus = load_pdf_corpus(pdf_path)

        # 2. Chunk text using your exact sentence-boundary rules + model token counting
        chunked_documents = chunk_continuous_corpus(
            corpus=corpus,
            model_harness=embedding_model,
            task_type="document",
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences,
        )
        print(f"  Compiled {len(chunked_documents)} token-safe semantic chunks")

        # 3. Transform our structured Document objects into JSON-serializable chunk dictionaries
        chunks_metadata = []
        texts_to_embed = []
        for doc in chunked_documents:
            texts_to_embed.append(doc.page_content)
            chunks_metadata.append({"text": doc.page_content, "metadata": doc.metadata})

        # 4. Generate embeddings using our async injection framework (handles task_type="document")
        print("Generating vector matrix via local framework engine...")
        result = await embedding_model.do_embed_async(
            texts_to_embed, task_type="document"
        )

        # Convert our nested Python float matrix into a structured NumPy array for FAISS
        emb = np.array(result.embeddings, dtype="float32")

        # 5. Populate the FAISS Vector Database Layer
        index = faiss.IndexFlatIP(
            emb.shape[1]
        )  # Inner Product == Cosine Similarity (vectors are normalized)
        index.add(np.ascontiguousarray(emb))
        print(f"  Indexed {index.ntotal} vectors of dim {emb.shape[1]}")

        return cls(embedding_model, index, chunks_metadata)

    # -- persistence ------------------------------------------------------- #
    def save(self, store: str):
        faiss = _require("faiss", "faiss-cpu")
        d = Path(store)
        d.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(d / "index.faiss"))
        (d / "meta.json").write_text(
            json.dumps(
                {
                    "model_name": self.embedding_model.embedding_model,
                    "chunks": self.chunks,
                },
                ensure_ascii=False,
            )
        )
        print(f"Saved index + multi-page metadata to {d}/")

    @classmethod
    def load(cls, store: str, model_config: LocalModelConfig):
        """Loads index and metadata, reconstructing the framework runtime via ModelConfig injection."""
        faiss = _require("faiss", "faiss-cpu")
        d = Path(store)

        meta = json.loads((d / "meta.json").read_text())
        index = faiss.read_index(str(d / "index.faiss"))

        # Re-initialize a clean, validated embedding driver framework context
        # (This leverages the same model id but configures it using current system environment rules)
        embedding_model = model_config.get_local_embedding_model()

        return cls(embedding_model, index, meta["chunks"])

    # -- search ------------------------------------------------------------ #
    async def search(self, query: str, k: int = 5) -> list[dict]:
        """Asynchronously converts query text and performs vector similarity search."""
        # 1. Run inference using task_type="query" to trigger the proper prefix engine strategy
        result = await self.embedding_model.do_embed_async([query], task_type="query")

        # Format the query vector precisely into a 2D matrix shape array for FAISS
        q_emb = np.array(result.embeddings, dtype="float32").reshape(1, -1)

        # 2. Query the vector matrix index space
        scores, idx = self.index.search(np.ascontiguousarray(q_emb), k)

        results = []
        for score, i in zip(scores[0], idx[0]):
            if i == -1:
                continue
            c = self.chunks[i]
            results.append(
                {
                    "score": float(score),
                    "text": c["text"],
                    "metadata": c[
                        "metadata"
                    ],  # Retains our multi-page lists, source string, etc.
                }
            )
        return results
