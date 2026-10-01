from typing import Literal
from pydantic import BaseModel

from src.local_agent_harness.local_config import LocalEmbeddingModel
from src.processors import split_into_sentences_with_offsets
from src.processors.pdf.corpus import Corpus, resolve_pages_for_chunk


class Document(BaseModel):
    page_content: str
    metadata: dict[str, object]


def chunk_continuous_corpus(
    corpus: Corpus,
    model_harness: LocalEmbeddingModel,
    task_type: Literal["query", "document", "symmetric"] = "document",
    max_tokens: int = 512,
    overlap_sentences: int = 1,
) -> list[Document]:
    """
    Groups custom cleaned sentences into token-safe spans.
    Resolves crossings cleanly by looking up character indices.
    """
    sentences_data = split_into_sentences_with_offsets(corpus.full_text)

    chunked_documents = []
    current_sentences = []

    def _commit_chunk(sentences_to_commit: list[dict]) -> None:
        """Helper to build the Document object and resolve overlapping pages."""
        if not sentences_to_commit:
            return

        chunk_text = " ".join([s["text"] for s in sentences_to_commit])
        c_start = sentences_to_commit[0]["start"]
        c_end = sentences_to_commit[-1]["end"]

        # Resolve exactly which physical pages this chunk intersects with
        pages = resolve_pages_for_chunk(c_start, c_end, corpus.page_maps)

        chunked_documents.append(
            Document(
                page_content=chunk_text,
                metadata={
                    "source": corpus.source_name,
                    "pages": pages,
                    "document_type": "cat_health_guideline",
                },
            )
        )

    for s_data in sentences_data:
        test_chunk_text = " ".join(
            [s["text"] for s in current_sentences] + [s_data["text"]]
        )
        total_tokens = model_harness.calculate_payload_tokens(
            test_chunk_text, task_type=task_type
        )

        if total_tokens <= max_tokens:
            current_sentences.append(s_data)
        else:
            # Commit the full active chunk buffer
            _commit_chunk(current_sentences)

            # Apply rolling window overlap tracking logic
            if overlap_sentences > 0 and len(current_sentences) >= overlap_sentences:
                current_sentences = current_sentences[-overlap_sentences:]
            else:
                current_sentences = []

            current_sentences.append(s_data)

    # Commit any lingering trailing items left in the buffer
    _commit_chunk(current_sentences)

    return chunked_documents
