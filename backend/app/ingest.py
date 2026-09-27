"""Ingestion pipeline: file -> Markdown -> chunks -> (contexts) -> embeddings -> MongoDB -> vector index sync."""
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from pymongo.errors import OperationFailure

from . import contextual, db, llm, retrieval
from .chunker import chunk_markdown
from .config import MARKDOWN_DIR, settings
from .contextual import ContextError, contextualize, indexed_content
from .converter import ConversionError, to_markdown
from .embeddings import embed_passages

log = logging.getLogger("ingest")


def _status(doc_id: str, status: str, progress: int, **extra):
    db.documents().update_one(
        {"_id": doc_id},
        {"$set": {"status": status, "progress": progress, "updated_at": datetime.now(timezone.utc), **extra}},
    )


def _wait_until_searchable(doc_id: str, expected: int, timeout_s: int = 120) -> bool:
    """mongot syncs asynchronously; only report 'ready' once the vector index sees every chunk."""
    if not retrieval.vector_index_ready:
        return True
    probe = [0.0] * settings.embed_dim
    probe[0] = 1.0
    pipeline = [
        {
            "$vectorSearch": {
                "index": settings.vector_index,
                "path": "embedding",
                "queryVector": probe,
                "exact": True,
                "limit": expected,
                "filter": {"doc_id": doc_id},
            }
        },
        {"$count": "n"},
    ]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            res = list(db.chunks().aggregate(pipeline))
            if res and res[0]["n"] >= expected:
                return True
        except OperationFailure as e:
            log.warning("waiting for vector index: %s", e)
        time.sleep(1)
    return False


def _free_gpu_for_chat() -> None:
    """As soon as contextualizing ends: unload the context model and start loading the answering model
    (which also does the pre-judge) in the background, so it is ready before the first question."""
    threading.Thread(target=_prepare_for_chat, daemon=True).start()


def _prepare_for_chat() -> None:
    if contextual.hand_over() is None:
        return  # another upload is still contextualizing; it hands over when it finishes
    try:
        started = time.perf_counter()
        llm.warm_up()
        log.info("LLM %s loaded after contextualizing in %.0fs", settings.llm_model, time.perf_counter() - started)
    except Exception as e:  # noqa: BLE001
        log.warning("Could not preload %s: %s", settings.llm_model, e)


def ingest(doc_id: str, path: Path, original_name: str) -> None:
    if settings.contextual_embedding:
        # contextualizing is the next model step: load the context model while converting and chunking
        contextual.warm_up_in_background("for a new upload")
    stage = "converting"
    try:
        _status(doc_id, "converting", 10)
        markdown = to_markdown(path, original_name)
        MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
        (MARKDOWN_DIR / f"{doc_id}.md").write_text(markdown, encoding="utf-8")
        db.documents().update_one({"_id": doc_id}, {"$set": {"markdown": markdown}})

        stage = "chunking"
        _status(doc_id, stage, 20)
        chunks = chunk_markdown(markdown, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise ConversionError("No text chunks could be produced from this document.")

        contexts = [""] * len(chunks)
        if settings.contextual_embedding:
            stage = "contextualizing"
            _status(doc_id, stage, 25, num_chunks=len(chunks), context_done=0)
            try:
                contexts = contextualize(
                    markdown, chunks,
                    lambda done, total: _status(doc_id, "contextualizing", 25 + int(35 * done / total),
                                                context_done=done),
                )
            finally:
                _free_gpu_for_chat()
        contents = [indexed_content(c, ctx) for c, ctx in zip(chunks, contexts)]

        stage = "embedding"
        _status(doc_id, stage, 60, num_chunks=len(chunks))
        vectors = []
        batch = 64
        for i in range(0, len(chunks), batch):
            vectors.extend(embed_passages(contents[i:i + batch]))
            _status(doc_id, "embedding", 60 + int(20 * min(i + batch, len(chunks)) / len(chunks)))

        stage = "storing"
        _status(doc_id, stage, 85)
        db.chunks().delete_many({"doc_id": doc_id})
        db.chunks().insert_many(
            [
                {
                    "doc_id": doc_id,
                    "index": c.index,
                    "section": c.section,
                    "text": c.text,
                    "context": ctx,
                    "content": content,
                    "embedding": v.tolist(),
                }
                for c, ctx, content, v in zip(chunks, contexts, contents, vectors)
            ]
        )

        stage = "indexing"
        _status(doc_id, stage, 92)
        if not _wait_until_searchable(doc_id, len(chunks)):
            log.warning("vector index did not catch up for %s; local fallback will cover it", doc_id)
        retrieval.bm25_cache.drop(doc_id)
        retrieval.bm25_cache.get(doc_id)  # warm the BM25 index

        _status(doc_id, "ready", 100)
        log.info("Ingested %s (%d chunks)", original_name, len(chunks))
    except (ConversionError, ContextError) as e:
        _status(doc_id, "failed", 100, error=str(e), failed_stage=stage)
    except Exception as e:  # noqa: BLE001
        log.exception("Ingestion failed for %s", doc_id)
        _status(doc_id, "failed", 100, error=f"Ingestion failed: {e}", failed_stage=stage)
