"""Ingestion pipeline: file -> Markdown -> chunks -> embeddings -> MongoDB -> vector index sync."""
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from pymongo.errors import OperationFailure

from . import db, retrieval
from .chunker import chunk_markdown
from .config import MARKDOWN_DIR, settings
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


def ingest(doc_id: str, path: Path, original_name: str) -> None:
    stage = "converting"
    try:
        _status(doc_id, "converting", 10)
        markdown = to_markdown(path, original_name)
        MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
        (MARKDOWN_DIR / f"{doc_id}.md").write_text(markdown, encoding="utf-8")
        db.documents().update_one({"_id": doc_id}, {"$set": {"markdown": markdown}})

        stage = "chunking"
        _status(doc_id, stage, 25)
        chunks = chunk_markdown(markdown, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise ConversionError("No text chunks could be produced from this document.")

        stage = "embedding"
        _status(doc_id, stage, 40, num_chunks=len(chunks))
        vectors = []
        batch = 64
        for i in range(0, len(chunks), batch):
            part = chunks[i:i + batch]
            vectors.extend(embed_passages([c.content for c in part]))
            _status(doc_id, "embedding", 40 + int(40 * min(i + batch, len(chunks)) / len(chunks)))

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
                    "content": c.content,
                    "embedding": v.tolist(),
                }
                for c, v in zip(chunks, vectors)
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
    except ConversionError as e:
        _status(doc_id, "failed", 100, error=str(e), failed_stage=stage)
    except Exception as e:  # noqa: BLE001
        log.exception("Ingestion failed for %s", doc_id)
        _status(doc_id, "failed", 100, error=f"Ingestion failed: {e}", failed_stage=stage)
