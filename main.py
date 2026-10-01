import argparse
import asyncio
from pathlib import Path

from src.local_agent_harness.harness_config import LocalModelConfig
from src.processors.rag import answer_with_rag
from src.storage.faiss.faiss_index import FaissIndex


async def run_pipeline():
    # 1. Boot system configurations and establish your decoupled runtime harness
    config = LocalModelConfig()
    model_harness = config.get_local_embedding_model()

    # 2. Generate vectors and build the FAISS Vector Index
    pdf_path = Path("./data/cat_health_guidelines.pdf")
    print("Building local vector space index layers...")
    faiss_index = await FaissIndex.build(
        pdf_path=pdf_path,
        embedding_model=model_harness,
        max_tokens=480,
        overlap_sentences=1,
    )

    # 3. Persist the index and structured metadata payloads to disk
    storage_dir = "./storage/cache/faiss_cache"
    faiss_index.save(storage_dir)
    print(f"Pipeline complete! Storage indexed successfully inside '{storage_dir}'")


async def query(user_query: str) -> None:
    config = LocalModelConfig()
    llm = config.get_local_llm()

    storage_dir = "./storage/cache/faiss_cache"
    faiss_index = FaissIndex.load(storage_dir, config)

    print(f"\nQuery: {user_query}")
    response = await answer_with_rag(query=user_query, index=faiss_index, llm=llm, k=5)
    print(f"\nAnswer:\n{response.response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local agentic RAG with Gemma")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("build", help="Build and persist the FAISS index (default)")
    query_parser = subparsers.add_parser("query", help="Answer a question from the index")
    query_parser.add_argument("question", help="Natural language question to answer")
    args = parser.parse_args()

    if args.command == "query":
        asyncio.run(query(args.question))
    else:
        asyncio.run(run_pipeline())
