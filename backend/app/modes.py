"""The two ways a document can be processed, chosen per upload in the UI.

private  everything runs on this machine: bge-small embeddings and Ollama models (the default).
cloud    document text and questions are sent through OpenRouter to Google (Gemini embeddings and chunk
         contexts) and DeepSeek (answers and the pre-judge).

Both modes use the same MongoDB, BM25 and reranker. Embeddings of different models can't be compared,
so each mode stores its vectors in its own chunk field with its own vector index, and a document is
answered in the mode it was uploaded in.
"""
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from . import cloud, embeddings, llm
from .config import settings

PRIVATE, CLOUD = "private", "cloud"


@dataclass(frozen=True)
class Mode:
    name: str
    label: str
    description: str
    local: bool               # True: models run in this machine's Ollama, which loads and unloads them
    embed_model: str
    context_model: str
    llm_model: str
    embedding_field: str      # chunk field holding this mode's vectors
    vector_index: str
    embed_dim: int
    vector_min_score: float
    context_num_ctx: int      # tokens of the document the context model reads per chunk

    def missing_keys(self) -> list[str]:
        return [] if self.local else cloud.missing_keys()

    def info(self) -> dict:
        """What the UI shows about the mode (ModeInfo in frontend/lib/api.ts)."""
        return {
            "id": self.name,
            "label": self.label,
            "description": self.description,
            "embed_model": self.embed_model,
            "context_model": self.context_model if settings.contextual_embedding else None,
            "llm_model": self.llm_model,
            "missing_keys": self.missing_keys(),
        }

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return embeddings.embed_passages(texts) if self.local else cloud.embed(texts, "document")

    def embed_query(self, query: str) -> np.ndarray:
        return embeddings.embed_query(query) if self.local else cloud.embed([query], "query")[0]

    def write_context(self, prompt: str) -> str:
        """The context model's reply to a Contextual Retrieval prompt."""
        if self.local:
            return llm.complete(settings.context_model, prompt, settings.context_num_ctx,
                                read_timeout=settings.llm_load_timeout, num_predict=120)
        return cloud.complete(settings.cloud_context_model, prompt, max_tokens=200)

    def judge_passages(self, prompt: str) -> str:
        """The answering model's short reply to the pre-judge prompt."""
        if self.local:
            # num_ctx must match answer generation (llm.stream_chat), or Ollama reloads the model
            return llm.complete(settings.llm_model, prompt, settings.llm_num_ctx,
                                read_timeout=settings.llm_timeout, num_predict=3)
        return cloud.complete(settings.cloud_llm_model, prompt, max_tokens=3)

    def stream_answer(self, messages: list[dict]) -> Iterator[str]:
        return llm.stream_chat(messages) if self.local else cloud.stream_chat(messages)


MODES = {
    PRIVATE: Mode(
        name=PRIVATE,
        label="Private mode",
        description="Runs on this machine. Nothing leaves it.",
        local=True,
        embed_model=settings.embed_model,
        context_model=settings.context_model,
        llm_model=settings.llm_model,
        embedding_field="embedding",
        vector_index=settings.vector_index,
        embed_dim=settings.embed_dim,
        vector_min_score=settings.vector_min_score,
        context_num_ctx=settings.context_num_ctx,
    ),
    CLOUD: Mode(
        name=CLOUD,
        label="Cloud mode",
        description="Runs through OpenRouter (Gemini, DeepSeek): the document text and your questions leave this machine.",
        local=False,
        embed_model=settings.cloud_embed_model,
        context_model=settings.cloud_context_model,
        llm_model=settings.cloud_llm_model,
        embedding_field="embedding_cloud",
        vector_index=f"{settings.vector_index}_cloud",
        embed_dim=settings.cloud_embed_dim,
        vector_min_score=settings.cloud_vector_min_score,
        context_num_ctx=settings.context_num_ctx,  # same windows as private mode, so the two stay comparable
    ),
}


def get(name: str | None) -> Mode:
    """A document's mode; documents from before modes existed are private."""
    return MODES[name or PRIVATE]
