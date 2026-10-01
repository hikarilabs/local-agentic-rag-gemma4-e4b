from typing import Optional

from openai import OpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.local_agent_harness.local_config import LocalEmbeddingModel, LocalLLM


class LocalModelConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env.local", env_file_encoding="utf-8")

    provider: str = Field(default=...)
    model: str = Field(default=...)
    api_key: str = Field(default=...)
    api_url: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_api_model: Optional[str] = None

    def get_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.api_url)

    def get_local_llm(self) -> LocalLLM:
        return LocalLLM(
            model=self.model,
            api_key=self.api_key,
            api_url=self.api_url or "",
        )

    def get_local_embedding_model(self) -> LocalEmbeddingModel:
        if not self.embedding_model:
            raise ValueError(
                f"EMBEDDING_MODEL is not set in .env.local — "
                f"embeddings are not available for provider '{self.provider}'"
            )
        return LocalEmbeddingModel(embedding_model=self.embedding_model)
