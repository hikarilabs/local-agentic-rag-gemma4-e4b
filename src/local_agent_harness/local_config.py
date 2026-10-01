import asyncio
from typing import Optional

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel, Field, PrivateAttr

from src.embeddings.base_embedding_engine import BaseEmbeddingsEngine, TaskType
from src.embeddings.engine_registry import EmbeddingsEngineRegistry
import src.embeddings.models


class LocalEmbeddingResult(BaseModel):
    text: list[str]
    model_formatted_text: list[str]
    embeddings: list[list[float]]


class LocalLLMResponse(BaseModel):
    response: str
    raw_response: dict


class LocalEmbeddingModel(BaseModel):
    embedding_model: str = Field(default=...)

    _engine: BaseEmbeddingsEngine | None = PrivateAttr(default=None)

    def model_post_init(self, __context: object) -> None:
        self._engine = EmbeddingsEngineRegistry.resolve(self.embedding_model)

    @classmethod
    def _validate_inputs(cls, text: list[str]) -> None:
        if not text:
            raise ValueError("No text provided for embedding")
        if any(not word.strip() for word in text):
            raise ValueError("Empty text provided for embedding")

    def do_embed(
        self, text: list[str], task_type: TaskType = "symmetric"
    ) -> LocalEmbeddingResult:
        """Synchronous pipeline entrypoint"""
        self._validate_inputs(text)
        if self._engine is None:
            raise RuntimeError("Embedding engine was not initialised")

        model_formatted_text = self._engine.prepare(text, task_type)
        local_embeddings = self._engine.embed(model_formatted_text)

        return LocalEmbeddingResult(
            text=text,
            model_formatted_text=model_formatted_text,
            embeddings=local_embeddings,
        )

    async def do_embed_async(self, text: list[str], task_type: TaskType = "symmetric"):
        self._validate_inputs(text)
        if self._engine is None:
            raise RuntimeError("Embedding engine was not initialised")

        model_formatted_text = self._engine.prepare(text, task_type)
        local_embeddings = await asyncio.to_thread(
            self._engine.embed, model_formatted_text
        )

        return LocalEmbeddingResult(
            text=text,
            model_formatted_text=model_formatted_text,
            embeddings=local_embeddings,
        )

    def count_tokens(self, text: str) -> int:
        """Exposes token validation capabilities directly to the local harness."""
        if self._engine is None:
            raise RuntimeError("Embedding engine was not initialised")
        return self._engine.count_tokens(text)

    def calculate_payload_tokens(
        self, text: str, task_type: TaskType = "document"
    ) -> int:
        if self._engine is None:
            raise RuntimeError("Embedding engine was not initialised")
        return self._engine.calculate_payload_tokens(text, task_type)


class LocalLLM(BaseModel):
    model: str = Field(default=...)
    api_key: str = Field(default=...)
    api_url: str = Field(default=...)

    def do_generate(
        self,
        *,
        system_prompt: str,
        prompt: str,
        history: Optional[list[ChatCompletionMessageParam]] = None,
    ) -> LocalLLMResponse:
        if not system_prompt or not system_prompt.strip():
            raise ValueError("No system prompt provided")
        if not prompt or not prompt.strip():
            raise ValueError("No prompt provided")

        messages: list[ChatCompletionMessageParam] = [
            ChatCompletionSystemMessageParam(role="system", content=system_prompt),
        ]

        if history:
            messages.extend(history)

        messages.append(ChatCompletionUserMessageParam(role="user", content=prompt))

        client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return LocalLLMResponse(
            response=response.choices[0].message.content or "",
            raw_response=response.model_dump(),
        )
