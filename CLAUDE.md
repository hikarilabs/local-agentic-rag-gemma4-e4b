# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Fully local RAG over a PDF: embeddinggemma-300m embeddings + FAISS for retrieval, Gemma 4 E4B via Ollama for generation.

## Commands

```bash
uv sync                                  # install deps (Python 3.13, see .python-version)
uv run main.py build                     # build index from data/cat_health_guidelines.pdf → storage/cache/faiss_cache/ (also the default with no args)
uv run main.py query "your question"     # answer a question from the saved index (needs Ollama running)
uv run ruff check .                      # lint (ruff is the only dev dependency)
uv run ruff format .                     # format
ollama pull gemma4:e4b && ollama serve   # required before running queries
```

There is no test suite. Run from the repo root: paths are relative, and `.env.local` is read from the current directory.

## Configuration

`LocalModelConfig` (`src/local_agent_harness/harness_config.py`) is a pydantic-settings class that reads `.env.local` (`PROVIDER`, `MODEL`, `API_KEY`, `API_URL`, `EMBEDDING_MODEL`). It is the factory for both the LLM (`get_local_llm`) and the embedding model (`get_local_embedding_model`). `.env.hf` holds an optional `HF_TOKEN` for the embedding model.

## Architecture

Imports are absolute from the repo root (`from src.… import …`); there is no installed package.

Data flow: `load_pdf_corpus` → `chunk_continuous_corpus` → `LocalEmbeddingModel.do_embed_async` → `FaissIndex` → `answer_with_rag` → `LocalLLM.do_generate`.

- **Embedding engines** (`src/embeddings/`): `BaseEmbeddingsEngine` defines `prepare` (model-specific prompt prefixes), `embed`, `count_tokens`, `calculate_payload_tokens`. Concrete engines register themselves in `EmbeddingsEngineRegistry` **at import time** (`register(...)` at the bottom of each module); registration happens because `local_config.py` does `import src.embeddings.models`. `resolve()` picks the first registered key that is a substring of the model name (e.g. `"gemma"`), otherwise falls back to `"default"` (`StandardTransformerEngine`). A new engine must be registered and imported from `src/embeddings/models/__init__.py`.
- **Task types** (`"query" | "document" | "symmetric"`) matter: `GemmaEngine` prepends `query: ` / `document: ` / `search_query: `. Indexing uses `"document"` and search uses `"query"`; keep these consistent when changing either side.
- **Chunking** (`src/processors/`): `load_pdf_corpus` joins all pages into one string and records `(start, end, page)` char ranges. `split_into_sentences_with_offsets` produces sentences with char offsets into that string, and `chunk_continuous_corpus` greedily packs sentences until `calculate_payload_tokens` (prefix and special tokens included) would exceed `max_tokens` (480 in `main.py`), with sentence overlap between chunks. Page numbers for a chunk come from overlapping its offsets with the page map, so a chunk can span several pages. Chunk metadata has `document_type` hardcoded to `"cat_health_guideline"`.
- **Storage** (`src/storage/`): `BaseIndex` is the backend contract (`build`, `save`, `load`, `search`). Search results must be `{"score", "text", "metadata"}` with `metadata` containing `source` and `pages`, which `build_rag_context` relies on. `FaissIndex` uses `IndexFlatIP` over normalized embeddings (cosine similarity) and persists `index.faiss` + `meta.json`. `load()` rebuilds the embedder from the **current** `EMBEDDING_MODEL` and ignores `model_name` in `meta.json`, so rebuild the index after changing the embedding model or the source PDF.
- **Generation** (`src/processors/rag.py`, `LocalLLM`): retrieved chunks are formatted as `[Source i | source | pages …]` blocks in the system prompt, which tells the model to answer only from that context. `LocalLLM.do_generate` is synchronous and creates a new `openai.OpenAI` client against `API_URL` on each call.
