# local-agentic-rag-gemma4-e4b

A fully local agentic RAG (Retrieval-Augmented Generation) pipeline powered by **Gemma 4 E4B**
(via Ollama) and **Google Embedding Gemma 300M** — no cloud inference required.

The pipeline ingests a PDF document, chunks it into token-safe semantic segments, generates
embeddings locally, stores them in a FAISS vector index, and answers natural language queries
using the locally running LLM.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | >= 3.13 |
| [uv](https://docs.astral.sh/uv/) | latest |
| [Ollama](https://ollama.com/) | latest |

---

## Prerequisites

### 1. Install Ollama and pull the model

Install Ollama from https://ollama.com, then pull the required model:

    ollama pull gemma4:e4b

Make sure Ollama is running before executing any queries:

    ollama serve

### 2. Hugging Face Token (optional)

The embedding model (google/embeddinggemma-300m) may require a Hugging Face token if it
becomes gated. Create a .env.hf file in the project root:

    HF_TOKEN=your_hf_token_here

Generate a token at https://huggingface.co/settings/tokens.
If the model is publicly accessible, this file is not required.

---

## Installation

### 1. Clone the repository

    git clone <repository-url>
    cd local-agentic-rag-gemma4-e4b

### 2. Create the virtual environment and sync dependencies

    uv venv
    uv sync

uv will automatically use Python 3.13 as defined in .python-version.

---

## Configuration

The local runtime is configured via .env.local in the project root:

    PROVIDER=ollama
    MODEL=gemma4:e4b
    API_KEY=ollama
    API_URL=http://localhost:11434/v1
    EMBEDDING_MODEL=google/embeddinggemma-300m

No changes are needed here unless you want to swap the model or point to a different Ollama instance.

---

## Usage

### Step 1 — Build the vector index

Run main.py once to process the PDF and persist the FAISS index to disk:

    uv run main.py

Expected output:

    Loading weights: 100%|████████| 314/314 [00:00<00:00, 7573.51it/s]
    Building local vector space index layers...
    Reading and mapping stream context from data/cat_health_guidelines.pdf ...
      Compiled 60 token-safe semantic chunks
    Generating vector matrix via local framework engine...
      Indexed 60 vectors of dim 768
    Saved index + multi-page metadata to storage/cache/faiss_cache/
    Pipeline complete! Storage indexed successfully inside './storage/cache/faiss_cache'

The index is saved to storage/cache/faiss_cache/ and only needs to be built once,
or whenever the source PDF changes.

### Step 2 — Run a query

Open main.py, comment out the run_pipeline() call and uncomment the query(...) call
with your question, then run:

    uv run main.py

---

## Project Structure

    local-agentic-rag-gemma4-e4b/
    ├── data/                          # Source PDF documents
    │   └── cat_health_guidelines.pdf
    ├── src/
    │   ├── embeddings/                # Embedding engine abstraction & registry
    │   ├── local_agent_harness/       # Model config & runtime harness
    │   ├── processors/                # PDF ingestion, text chunking, RAG pipeline
    │   └── storage/                   # FAISS index abstraction & persistence
    ├── storage/
    │   └── cache/faiss_cache/         # Persisted FAISS index (auto-generated)
    ├── main.py                        # Entry point — build index or run queries
    ├── .env.local                     # Local LLM & embedding configuration
    ├── .env.hf                        # Hugging Face token (not committed)
    ├── pyproject.toml
    └── uv.lock

---

## License

See LICENSE.
