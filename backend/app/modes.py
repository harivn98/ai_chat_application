"""The ways a document can be processed, chosen per upload in the UI.

private   everything runs on this machine: bge-small embeddings, Ollama models, the local reranker and
          pre-judge as RERANKER_ENABLED / PREJUDGE_ENABLED say (the default).
cloud     document text and questions go through OpenRouter to Google (Gemini embeddings and chunk contexts) and
          DeepSeek (answers). Each upload switches on the reranker, the pre-judge or both:
            reranker   the local cross-encoder reranks all the fused candidates down to TOP_K
            pre-judge  Gemini Flash-Lite checks whether the TOP_K passages hold all, part or none of the answer
          (both: rerank first, then pre-judge the reranked passages).

All modes use the same MongoDB and BM25. Embeddings of different models can't be compared, so cloud mode stores
its vectors in their own chunk field with their own vector index, and a document is answered in the mode (and
with the switches) it was uploaded with.
"""
from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np

from . import embeddings, ollama, openrouter
from .config import settings

PRIVATE, CLOUD = "private", "cloud"

# Cloud variants by name (reranker, pre-judge): the evaluation's --mode / --compare, and the old per-variant modes
CLOUD_VARIANTS = {
    "cloud-rerank": (True, False),
    "cloud-prejudge": (False, True),
    "cloud-rerank-prejudge": (True, True),
}


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
    def variant(self) -> str:
        """private, or cloud-rerank / cloud-prejudge / cloud-rerank-prejudge."""
        if self.local:
            return PRIVATE
        return "-".join([CLOUD] + ["rerank"] * self.reranker + ["prejudge"] * self.prejudge)

    @property
    def candidate_pool(self) -> int:
        """Most chunks the fusion can return; the reranker scores all of them."""
        return self.bm25_candidates + self.vector_candidates

    def missing_keys(self) -> list[str]:
        return [] if self.local else openrouter.missing_keys()

    def missing_keys_error(self) -> str | None:
        """Why the mode can't be used yet (the API keys it still needs), or None when it can."""
        missing = self.missing_keys()
        return f"{self.label} needs {' and '.join(missing)} (see README)." if missing else None

    def info(self) -> dict:
        """What the upload page shows about the mode (ModeInfo in frontend/lib/api.ts)."""
        return {
            "id": self.name,
            "label": self.label,
            "description": self.description,
            "missing_keys": self.missing_keys(),
        }

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return embeddings.embed_passages(texts) if self.local else openrouter.embed(texts, "document")

    def embed_query(self, query: str) -> np.ndarray:
        return embeddings.embed_query(query) if self.local else openrouter.embed([query], "query")[0]

    def write_context(self, prompt: str) -> str:
        """The context model's reply to a Contextual Retrieval prompt."""
        if self.local:
            return ollama.complete(settings.context_model, prompt, settings.context_num_ctx,
                                   read_timeout=settings.llm_load_timeout, num_predict=120)
        return openrouter.complete(settings.cloud_context_model, prompt, max_tokens=400,
                                   reasoning=openrouter.MINIMAL_REASONING)

    def rewrite_question(self, prompt: str) -> str:
        """The answering model's standalone version of a follow-up question."""
        if self.local:
            # num_ctx must match answer generation (ollama.stream_chat), or Ollama reloads it
            return ollama.complete(settings.llm_model, prompt, settings.llm_num_ctx,
                                   read_timeout=settings.llm_timeout, num_predict=150)
        return openrouter.complete(settings.cloud_llm_model, prompt, max_tokens=150, reasoning={"enabled": False})

    def judge_passages(self, prompt: str) -> str:
        """The pre-judge model's one-word reply to a pre-judge prompt."""
        if self.local:
            # the answering model; num_ctx must match answer generation (ollama.stream_chat), or Ollama reloads it
            return ollama.complete(settings.llm_model, prompt, settings.llm_num_ctx,
                                   read_timeout=settings.llm_timeout, num_predict=3)
        return openrouter.complete(self.prejudge_model, prompt, max_tokens=50,
                                   reasoning=openrouter.MINIMAL_REASONING)

    def stream_answer(self, messages: list[dict]) -> Iterator[str]:
        return ollama.stream_chat(messages) if self.local else openrouter.stream_chat(messages)


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
    CLOUD: Mode(
        name=CLOUD,
        label="Cloud mode",
        description="Cloud models via OpenRouter. The document text and your questions leave this machine.",
        local=False,
        embed_model=settings.cloud_embed_model,
        context_model=settings.cloud_context_model,
        llm_model=settings.cloud_llm_model,
        reranker=True,  # the switches are chosen per upload (cloud_mode)
        prejudge=True,
        prejudge_model=settings.cloud_prejudge_model,
        top_k=settings.top_k,
        bm25_candidates=settings.cloud_bm25_candidates,
        vector_candidates=settings.cloud_vector_candidates,
        embedding_field="embedding_cloud",  # every cloud variant shares the embeddings and their index
        vector_index=f"{settings.vector_index}_cloud",
        embed_dim=settings.cloud_embed_dim,
        vector_min_score=settings.cloud_vector_min_score,
        context_num_ctx=settings.context_num_ctx,  # same windows as private mode, so the modes stay comparable
    ),
}

_CLOUD_LABELS = {(True, False): "Cloud · reranker", (False, True): "Cloud · pre-judge",
                 (True, True): "Cloud · reranker + pre-judge"}


def cloud_mode(reranker: bool, prejudge: bool) -> Mode:
    """Cloud mode with the reranker and/or the pre-judge switched on (at least one)."""
    if not (reranker or prejudge):
        raise ValueError("Cloud mode needs the reranker, the pre-judge or both.")
    return replace(MODES[CLOUD], reranker=reranker, prejudge=prejudge, label=_CLOUD_LABELS[reranker, prejudge])


def by_variant(name: str) -> Mode:
    """private, or a cloud variant from CLOUD_VARIANTS."""
    return MODES[PRIVATE] if name == PRIVATE else cloud_mode(*CLOUD_VARIANTS[name])


def for_upload(name: str, reranker: bool, prejudge: bool) -> Mode:
    """The mode picked for an upload. The switches apply to cloud mode only; private mode follows .env.
    Raises ValueError for an unknown mode, or for cloud mode with both switches off."""
    if name not in MODES:
        raise ValueError(f"Unknown mode {name!r}.")
    return cloud_mode(reranker, prejudge) if name == CLOUD else MODES[name]


def for_document(doc: dict) -> Mode:
    """The mode a document was uploaded in. Documents from before modes existed are private; cloud-rerank and
    cloud-prejudge documents come from before the switches; plain cloud documents without switches from before
    the two cloud modes, when cloud mode always reranked."""
    name = doc.get("mode") or PRIVATE
    if name == PRIVATE:
        return MODES[PRIVATE]
    if name in CLOUD_VARIANTS:
        return by_variant(name)
    return cloud_mode(doc.get("reranker", True), doc.get("prejudge", False))
