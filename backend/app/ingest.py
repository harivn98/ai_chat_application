"""Ingestion pipeline: file -> Markdown -> chunks -> (contexts) -> embeddings -> MongoDB -> vector index sync."""
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import contextual, db, llm, retrieval
from .chunker import Chunk, chunk_markdown
from .config import settings
from .contextual import ContextError, contextualize, indexed_content
from .converter import ConversionError, to_markdown
from .embeddings import embed_passages

log = logging.getLogger("ingest")

EMBED_BATCH = 64  # chunks embedded between two progress updates


def store_chunks(
    doc_id: str, chunks: list[Chunk], contexts: list[str], contents: list[str], vectors: Iterable[np.ndarray]
) -> None:
    """Replace the document's chunks in MongoDB (the evaluation stores its papers the same way)."""
    db.chunks().delete_many({"doc_id": doc_id})
    db.chunks().insert_many(
        [
            {
                "doc_id": doc_id,
                "index": c.index,
                "section": c.section,
                "text": c.text,
                "start": c.start,
                "end": c.end,
                "context": ctx,
                "content": content,
                "embedding": v.tolist(),
            }
            for c, ctx, content, v in zip(chunks, contexts, contents, vectors)
        ]
    )


def _status(doc_id: str, status: str, progress: int, **extra):
    db.documents().update_one(
        {"_id": doc_id},
        {"$set": {"status": status, "progress": progress, "updated_at": datetime.now(timezone.utc), **extra}},
    )


def _switch_to_answering_model() -> None:
    """Right after contextualizing: unload the context model and start loading the answering model (which also
    does the pre-judge), so it is ready before the first question. While another upload is still contextualizing,
    nothing happens here; that upload switches when it finishes."""
    if contextual.hand_over() is not None:
        llm.load_in_background(settings.llm_model, settings.llm_num_ctx, "after contextualizing")


def ingest(doc_id: str, path: Path, original_name: str) -> None:
    if settings.contextual_embedding:
        # contextualizing is the next model step: load the context model while converting and chunking
        llm.load_in_background(settings.context_model, settings.context_num_ctx, "for a new upload")
    stage = "converting"
    try:
        _status(doc_id, stage, 10)
        markdown = to_markdown(path, original_name)
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
                _switch_to_answering_model()
        contents = [indexed_content(c, ctx) for c, ctx in zip(chunks, contexts)]

        stage = "embedding"
        _status(doc_id, stage, 60, num_chunks=len(chunks))
        vectors = []
        for i in range(0, len(chunks), EMBED_BATCH):
            vectors.extend(embed_passages(contents[i:i + EMBED_BATCH]))
            _status(doc_id, stage, 60 + int(20 * min(i + EMBED_BATCH, len(chunks)) / len(chunks)))

        stage = "storing"
        _status(doc_id, stage, 85)
        store_chunks(doc_id, chunks, contexts, contents, vectors)

        stage = "indexing"
        _status(doc_id, stage, 92)
        if not db.wait_until_searchable(doc_id, len(chunks)):
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
