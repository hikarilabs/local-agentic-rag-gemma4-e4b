from src.storage.base_index import BaseIndex
from src.local_agent_harness.local_config import LocalLLM, LocalLLMResponse


def build_rag_context(search_results: list[dict]) -> str:
    """Formats vector search results into an injectable context block for the LLM."""
    context_parts = []
    for i, result in enumerate(search_results, start=1):
        pages = result["metadata"].get("pages", [])
        source = result["metadata"].get("source", "unknown")
        context_parts.append(
            f"[Source {i} | {source} | pages {pages}]\n{result['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


async def answer_with_rag(
    query: str,
    index: BaseIndex,  # ← accepts ANY index backend
    llm: LocalLLM,
    k: int = 5,
) -> LocalLLMResponse:
    """Full RAG pipeline: retrieve → format → generate."""
    search_results = await index.search(query, k=k)

    if not search_results:
        raise ValueError("No relevant context found in the index for this query.")

    context = build_rag_context(search_results)

    system_prompt = (
        "You are a helpful assistant that answers questions strictly based on "
        "the provided context excerpts. If the answer cannot be found in the "
        "context, say so clearly — do not speculate or use outside knowledge.\n\n"
        "CONTEXT:\n"
        f"{context}"
    )

    return llm.do_generate(system_prompt=system_prompt, prompt=query)
