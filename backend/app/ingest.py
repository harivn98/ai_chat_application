"""Ingestion pipeline: file -> Markdown -> chunks -> (contexts) -> embeddings -> MongoDB -> vector index sync."""
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import db, ollama, retrieval
from .chunker import chunk_markdown
from .config import settings
from .contextual import ContextError, contextualize, indexed_content, release_context_model
from .converter import ConversionError, to_markdown
from .modes import Mode

log = logging.getLogger("ingest")

EMBED_BATCH = 64  # chunks embedded between two progress updates


class DocumentDeleted(Exception):
    """The document was deleted while it was being ingested."""


def _set_status(doc_id: str, status: str, progress: int, **extra) -> None:
    """Raises DocumentDeleted if the document no longer exists, so its ingestion stops."""
    if not db.update_document(doc_id, status=status, progress=progress, updated_at=datetime.now(timezone.utc), **extra):
        raise DocumentDeleted


def _set_failed(doc_id: str, error: str, stage: str) -> None:
    """Mark the document failed (nothing to do if it was deleted meanwhile)."""
    db.update_document(doc_id, status="failed", progress=100, error=error, failed_stage=stage,
                       updated_at=datetime.now(timezone.utc))


def _switch_to_answering_model() -> None:
    """Right after contextualizing: unload the context model and start loading the answering model (which also
    does the pre-judge), so it is ready before the first question. While another upload is still contextualizing,
    nothing happens here; that upload switches when it finishes."""
    if release_context_model() is not None:
        ollama.load_in_background(settings.llm_model, settings.llm_num_ctx, "after contextualizing")


def ingest(doc_id: str, path: Path, original_name: str, mode: Mode) -> None:
    if settings.contextual_embedding and mode.local:
        # contextualizing is the next model step: load the context model while converting and chunking
        ollama.load_in_background(settings.context_model, settings.context_num_ctx, "for a new upload")
    stage = "converting"
    try:
        _set_status(doc_id, stage, 10)
        markdown = to_markdown(path, original_name)
        if not db.update_document(doc_id, markdown=markdown):
            raise DocumentDeleted

        stage = "chunking"
        _set_status(doc_id, stage, 20)
        chunks = chunk_markdown(markdown, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise ConversionError("No text chunks could be produced from this document.")

        contexts = [""] * len(chunks)
        if settings.contextual_embedding:
            stage = "contextualizing"
            _set_status(doc_id, stage, 25, num_chunks=len(chunks), context_done=0)
            try:
                contexts = contextualize(
                    markdown, chunks, mode,
                    lambda done, total: _set_status(doc_id, "contextualizing", 25 + int(35 * done / total),
                                                    context_done=done),
                )
            finally:
                if mode.local:
                    _switch_to_answering_model()
        contents = [indexed_content(c, ctx) for c, ctx in zip(chunks, contexts)]

        stage = "embedding"
        _set_status(doc_id, stage, 60, num_chunks=len(chunks))
        vectors = []
        for i in range(0, len(chunks), EMBED_BATCH):
            vectors.extend(mode.embed_passages(contents[i:i + EMBED_BATCH]))
            _set_status(doc_id, stage, 60 + int(20 * min(i + EMBED_BATCH, len(chunks)) / len(chunks)))

        stage = "storing"
        _set_status(doc_id, stage, 85)
        db.replace_chunks(doc_id, chunks, contexts, contents, vectors, mode)

        stage = "indexing"
        _set_status(doc_id, stage, 92)
        if not db.wait_until_searchable(doc_id, len(chunks), mode):
            log.warning("vector index did not catch up for %s; local fallback will cover it", doc_id)
        retrieval.preload_document(doc_id)

        _set_status(doc_id, "ready", 100)
        log.info("Ingested %s (%d chunks)", original_name, len(chunks))
    except DocumentDeleted:
        # deleted mid-way (New session, or deleted from the documents list): drop anything stored since
        log.info("%s was deleted while it was being ingested; stopped at %s", original_name, stage)
        db.delete_chunks(doc_id)
        retrieval.drop_document(doc_id)
    except (ConversionError, ContextError) as e:
        _set_failed(doc_id, str(e), stage)
    except Exception as e:  # noqa: BLE001
        log.exception("Ingestion failed for %s", doc_id)
        _set_failed(doc_id, f"Ingestion failed: {e}", stage)
