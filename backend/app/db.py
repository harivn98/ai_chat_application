"""MongoDB access and the Atlas Vector Search indexes over the chunk embeddings (one per mode)."""
import logging
import time
from functools import lru_cache

from pymongo import ASCENDING, MongoClient
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from .config import settings
from .modes import MODES, Mode

log = logging.getLogger("db")

# Set by ensure_vector_index(). False: the deployment has no search support, so vector search runs in-process.
vector_index_ready = False


@lru_cache(maxsize=1)
def client() -> MongoClient:
    return MongoClient(settings.mongo_uri, serverSelectionTimeoutMS=5000, tz_aware=True)


def db():
    return client()[settings.mongo_db]


def documents():
    return db()["documents"]


def chunks():
    return db()["chunks"]


def _index_definition(mode: Mode) -> dict:
    return {
        "fields": [
            {"type": "vector", "path": mode.embedding_field, "numDimensions": mode.embed_dim, "similarity": "cosine"},
            {"type": "filter", "path": "doc_id"},
        ]
    }


def _find_vector_index(name: str) -> dict | None:
    return next(iter(chunks().list_search_indexes(name)), None)


def ensure_vector_index(timeout_s: int = 180) -> bool:
    """Create each mode's vector index if missing and wait until all of them are queryable.

    Returns False (and retrieval falls back to in-process cosine search) when the
    MongoDB deployment has no search support, e.g. a plain `mongo` image.
    """
    global vector_index_ready
    if "chunks" not in db().list_collection_names():
        db().create_collection("chunks")
    chunks().create_index([("doc_id", ASCENDING), ("index", ASCENDING)])

    deadline = time.time() + timeout_s
    while True:
        try:
            queryable = True
            for mode in {m.vector_index: m for m in MODES.values()}.values():  # modes can share an index
                if _find_vector_index(mode.vector_index) is None:
                    chunks().create_search_index(
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
            res = list(chunks().aggregate(pipeline))
            if res and res[0]["n"] >= expected:
                return True
        except OperationFailure as e:
            log.warning("waiting for vector index: %s", e)
        time.sleep(1)
    return False
