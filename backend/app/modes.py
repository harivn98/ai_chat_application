"""The ways a document can be processed, chosen per upload in the UI.

private          everything runs on this machine: bge-small embeddings, Ollama models, the local reranker and
                 pre-judge as RERANKER_ENABLED / PREJUDGE_ENABLED say (the default).
cloud-rerank     document text and questions go through OpenRouter to Google (Gemini embeddings and chunk
                 contexts) and DeepSeek (answers); the local cross-encoder reranks all the fused candidates.
cloud-prejudge   the same cloud models without the reranker; instead Gemini Flash-Lite pre-judges whether the
                 top passages hold all, part or none of the information the question asks for.

All modes use the same MongoDB and BM25. Embeddings of different models can't be compared, so the cloud modes
store their vectors in their own chunk field with their own vector index, and a document is answered in the mode
it was uploaded in.
"""
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from . import cloud, embeddings, llm
from .config import settings

PRIVATE, CLOUD_RERANK, CLOUD_PREJUDGE = "private", "cloud-rerank", "cloud-prejudge"


@dataclass(frozen=True)
class Mode:
    name: str
    label: str
    description: str
    local: bool               # True: models run in this machine's Ollama, which loads and unloads them
    embed_model: str
    context_model: str
    llm_model: str
    reranker: bool            # the local cross-encoder reranks the fused candidates down to TOP_K
    prejudge: bool            # a pre-judge checks the passages before the answer is generated
    prejudge_model: str
    top_k: int                # passages sent to the pre-judge and the answering model
    bm25_candidates: int      # chunks BM25 contributes to the fusion
    vector_candidates: int    # max chunks vector search contributes (before its score cutoff, if any)
    embedding_field: str      # chunk field holding this mode's vectors
    vector_index: str
    embed_dim: int
    vector_min_score: float | None  # minimum cosine similarity for vector hits; None = no cutoff
    context_num_ctx: int      # tokens of the document the context model reads per chunk

    @property
    def candidate_pool(self) -> int:
        """Most chunks the fusion can return; the reranker scores all of them."""
        return self.bm25_candidates + self.vector_candidates

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
            "reranker_model": settings.reranker_model if self.reranker else None,
            "prejudge_model": self.prejudge_model if self.prejudge else None,
            "passages": self.top_k,
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
        return cloud.complete(settings.cloud_context_model, prompt, max_tokens=400,
                              reasoning=cloud.MINIMAL_REASONING)

    def judge_passages(self, prompt: str) -> str:
        """The pre-judge model's one-word reply to a pre-judge prompt."""
        if self.local:
            # the answering model; num_ctx must match answer generation (llm.stream_chat), or Ollama reloads it
            return llm.complete(settings.llm_model, prompt, settings.llm_num_ctx,
                                read_timeout=settings.llm_timeout, num_predict=3)
        return cloud.complete(self.prejudge_model, prompt, max_tokens=50, reasoning=cloud.MINIMAL_REASONING)

    def stream_answer(self, messages: list[dict]) -> Iterator[str]:
        return llm.stream_chat(messages) if self.local else cloud.stream_chat(messages)


_CLOUD = dict(
    local=False,
    embed_model=settings.cloud_embed_model,
    context_model=settings.cloud_context_model,
    llm_model=settings.cloud_llm_model,
    prejudge_model=settings.cloud_prejudge_model,
    embedding_field="embedding_cloud",  # both cloud modes share the embeddings and their index
    vector_index=f"{settings.vector_index}_cloud",
    embed_dim=settings.cloud_embed_dim,
    vector_min_score=settings.cloud_vector_min_score,
    context_num_ctx=settings.context_num_ctx,  # same windows as private mode, so the modes stay comparable
)

MODES = {
    PRIVATE: Mode(
        name=PRIVATE,
        label="Private mode",
        description="Runs on this machine. Nothing leaves it.",
        local=True,
        embed_model=settings.embed_model,
        context_model=settings.context_model,
        llm_model=settings.llm_model,
        reranker=settings.reranker_enabled,
        prejudge=settings.prejudge_enabled,
        prejudge_model=settings.llm_model,
        top_k=settings.top_k,
        bm25_candidates=settings.bm25_candidates,
        vector_candidates=settings.vector_candidates,
        embedding_field="embedding",
        vector_index=settings.vector_index,
        embed_dim=settings.embed_dim,
        vector_min_score=None,  # top vector_candidates by rank only; the reranker judges relevance
        context_num_ctx=settings.context_num_ctx,
    ),
    CLOUD_RERANK: Mode(
        name=CLOUD_RERANK,
        label="Cloud · reranker",
        description="Cloud models via OpenRouter; this machine's cross-encoder reranks the retrieved passages. "
                    "The document text and your questions leave this machine.",
        reranker=True,
        prejudge=False,
        top_k=settings.top_k,  # the local reranker picks as many passages as in private mode
        bm25_candidates=settings.bm25_candidates,
        vector_candidates=settings.vector_candidates,
        **_CLOUD,
    ),
    CLOUD_PREJUDGE: Mode(
        name=CLOUD_PREJUDGE,
        label="Cloud · pre-judge",
        description="Cloud models via OpenRouter, no reranker; before answering, a pre-judge checks whether the "
                    "passages hold all, part or none of the answer. The document text and your questions leave "
                    "this machine.",
        reranker=False,
        prejudge=True,
        top_k=settings.cloud_prejudge_top_k,  # no reranker: Flash-Lite and DeepSeek read more passages instead
        bm25_candidates=settings.cloud_prejudge_bm25_candidates,
        vector_candidates=settings.cloud_prejudge_vector_candidates,
        **_CLOUD,
    ),
}

_RENAMED = {"cloud": CLOUD_RERANK}  # cloud documents uploaded before the two cloud modes existed were reranked


def get(name: str | None) -> Mode:
    """A document's mode; documents from before modes existed are private."""
    name = name or PRIVATE
    return MODES[_RENAMED.get(name, name)]
