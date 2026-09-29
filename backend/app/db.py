"""MongoDB: the only module that reads or writes it.

Two collections: `documents` (one record per upload: status, progress, Markdown) and `chunks` (one record per
chunk, with its vectors in the embedding field of the mode the document was uploaded in). Each mode has an Atlas
Vector Search index over its embedding field.
"""
import logging
import time
from collections.abc import Iterable
from functools import lru_cache

import numpy as np
from pymongo import ASCENDING, MongoClient
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from .chunker import Chunk
from .config import settings
from .modes import MODES, Mode

log = logging.getLogger("db")

# Set by ensure_indexes(). False: the deployment has no search support, so vector search runs in-process.
vector_index_ready = False


@lru_cache(maxsize=1)
def client() -> MongoClient:
    return MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000, tz_aware=True)


def _database():
    return client()[settings.mongo_db]


def _documents():
    return _database()["documents"]


def _chunks():
    return _database()["chunks"]


def ping() -> None:
    """Raises if MongoDB can't be reached."""
    client().admin.command("ping")


# ------------------------------------------------------------------ documents
def insert_document(doc: dict) -> None:
    _documents().insert_one(doc)


def get_document(doc_id: str) -> dict | None:
    """The document's record without its Markdown (which can be large)."""
    return _documents().find_one({"_id": doc_id}, {"markdown": 0})


def get_markdown(doc_id: str) -> str | None:
    doc = _documents().find_one({"_id": doc_id}, {"markdown": 1})
    return doc.get("markdown") if doc else None


def update_document(doc_id: str, **fields) -> None:
    _documents().update_one({"_id": doc_id}, {"$set": fields})


def delete_document(doc_id: str) -> None:
    """The document's record and all its chunks."""
    delete_chunks(doc_id)
    _documents().delete_one({"_id": doc_id})


# ------------------------------------------------------------------ chunks
def replace_chunks(
    doc_id: str, chunks: list[Chunk], contexts: list[str], contents: list[str], vectors: Iterable[np.ndarray],
    mode: Mode,
) -> None:
    """Replace the document's chunks, with the vectors in the mode's embedding field (the evaluation stores its
    papers the same way)."""
    delete_chunks(doc_id)
    _chunks().insert_many(
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
                mode.embedding_field: v.tolist(),
            }
            for c, ctx, content, v in zip(chunks, contexts, contents, vectors)
        ]
    )


def load_chunks(doc_id: str) -> list[dict]:
    """The document's chunks in document order, without their vectors."""
    no_vectors = {mode.embedding_field: 0 for mode in MODES.values()}
    return list(_chunks().find({"doc_id": doc_id}, no_vectors).sort("index", 1))


def chunk_vectors(doc_id: str, mode: Mode) -> list[tuple[int, list[float]]]:
    """(chunk index, vector) for every chunk of the document that has a vector in the mode's embedding field."""
    field = mode.embedding_field
    rows = _chunks().find({"doc_id": doc_id, field: {"$exists": True}}, {"index": 1, field: 1})
    return [(r["index"], r[field]) for r in rows]


def delete_chunks(doc_id: str) -> None:
    _chunks().delete_many({"doc_id": doc_id})


# ------------------------------------------------------------------ vector search
def _index_definition(mode: Mode) -> dict:
    return {
        "fields": [
            {"type": "vector", "path": mode.embedding_field, "numDimensions": mode.embed_dim, "similarity": "cosine"},
            {"type": "filter", "path": "doc_id"},
        ]
    }


def _find_vector_index(name: str) -> dict | None:
    return next(iter(_chunks().list_search_indexes(name)), None)


def ensure_indexes(timeout_s: int = 180) -> bool:
    """Create the chunks collection and its indexes if missing: (doc_id, index), and each mode's vector index.
    Then wait until the vector indexes are queryable.

    Returns False (and retrieval falls back to in-process cosine search) when the
    MongoDB deployment has no search support, e.g. a plain `mongo` image.
    """
    global vector_index_ready
    if "chunks" not in _database().list_collection_names():
        _database().create_collection("chunks")
    _chunks().create_index([("doc_id", ASCENDING), ("index", ASCENDING)])

    deadline = time.time() + timeout_s
    while True:
        try:
            queryable = True
            for mode in {m.vector_index: m for m in MODES.values()}.values():  # modes can share an index
                if _find_vector_index(mode.vector_index) is None:
                    _chunks().create_search_index(
                        SearchIndexModel(definition=_index_definition(mode), name=mode.vector_index, type="vectorSearch")
                    )
                    log.info("Created vector search index %s", mode.vector_index)
                index = _find_vector_index(mode.vector_index)
                queryable = queryable and bool(index and index.get("queryable"))
            if queryable:
                log.info("Vector search indexes are queryable")
                vector_index_ready = True
                return True
        except OperationFailure as e:
            # mongot can take a few seconds to come up after mongod
            log.warning("Vector index not ready yet: %s", e)
        if time.time() > deadline:
            log.error("Vector search index unavailable; using in-process cosine fallback")
            vector_index_ready = False
            return False
        time.sleep(3)


def vector_search(doc_id: str, query_vector: np.ndarray, k: int, mode: Mode) -> list[tuple[int, float]]:
    """(chunk index, cosine similarity) of the document's k chunks closest to the query, via the mode's vector index.
    Empty when there is no vector index or the search failed; the caller then searches in-process."""
    if not vector_index_ready:
        return []
    pipeline = [
        {
            "$vectorSearch": {
                "index": mode.vector_index,
                "path": mode.embedding_field,
                "queryVector": query_vector.tolist(),
                "numCandidates": max(k * 10, 100),
                "limit": k,
                "filter": {"doc_id": doc_id},
            }
        },
        {"$project": {"_id": 0, "index": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    try:
        # Atlas reports cosine as (1 + cos) / 2; convert back so scores match the in-process search
        return [(r["index"], 2.0 * float(r["score"]) - 1.0) for r in _chunks().aggregate(pipeline)]
    except OperationFailure as e:
        log.warning("$vectorSearch failed, using local fallback: %s", e)
        return []


def wait_until_searchable(doc_id: str, expected: int, mode: Mode, timeout_s: int = 120) -> bool:
    """mongot syncs asynchronously: wait until the mode's vector index sees all `expected` chunks of the document."""
    if not vector_index_ready:
        return True
    probe = [0.0] * mode.embed_dim
    probe[0] = 1.0
    pipeline = [
        {
            "$vectorSearch": {
                "index": mode.vector_index,
                "path": mode.embedding_field,
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
            res = list(_chunks().aggregate(pipeline))
            if res and res[0]["n"] >= expected:
                return True
        except OperationFailure as e:
            log.warning("waiting for vector index: %s", e)
        time.sleep(1)
    return False
