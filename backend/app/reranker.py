"""Cross-encoder reranking of the fused BM25 + vector candidates (RERANKER_MODEL, default BAAI/bge-reranker-base).

The retrievers score the question and each chunk separately; a cross-encoder reads the question and a
chunk together, which ranks them much more precisely. Hybrid search hands over its top RERANK_CANDIDATES
chunks and the reranker keeps the best TOP_K. It runs on the CPU, so the GPU stays free for the LLM.
Both supported models are baked into the image (Dockerfile ARG RERANKER_MODELS); RERANKER_MODEL picks one:
BAAI/bge-reranker-base (278M parameters, more accurate) or cross-encoder/ms-marco-MiniLM-L6-v2 (23M, fast).
"""
import os
import threading
from functools import lru_cache

from .config import settings

_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import CrossEncoder

    # The container runs offline, so only models baked into the image (Dockerfile ARG RERANKER_MODELS) load
    baked = os.environ.get("RERANKER_MODELS", "").split()
    if baked and settings.reranker_model not in baked:
        raise RuntimeError(
            f"RERANKER_MODEL={settings.reranker_model} is not in the image. Choose one of: {', '.join(baked)} "
            "(or add it to RERANKER_MODELS in backend/Dockerfile and rebuild)."
        )
    return CrossEncoder(settings.reranker_model, device="cpu")


def rerank(query: str, candidates: list[dict], k: int) -> list[dict]:
    """Return the k best candidates by cross-encoder score, each with rerank_score and rerank_rank added.

    Candidates are scored on their indexed content (context + heading path + chunk text), the same text
    BM25 and the embeddings see.
    """
    if not candidates:
        return []
    pairs = [(query, c.get("content") or c["text"]) for c in candidates]
    with _lock:  # the model is shared across request threads
        scores = get_model().predict(pairs, batch_size=32, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda cs: float(cs[1]), reverse=True)[:k]
    return [{**c, "rerank_score": round(float(s), 4), "rerank_rank": i}
            for i, (c, s) in enumerate(ranked, start=1)]
