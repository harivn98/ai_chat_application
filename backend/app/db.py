"""MongoDB access and the Atlas Vector Search index over the chunk embeddings."""
import logging
import time
from functools import lru_cache

from pymongo import ASCENDING, MongoClient
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from .config import settings

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


VECTOR_INDEX_DEFINITION = {
    "fields": [
        {"type": "vector", "path": "embedding", "numDimensions": settings.embed_dim, "similarity": "cosine"},
        {"type": "filter", "path": "doc_id"},
    ]
}


def _find_vector_index() -> dict | None:
    return next(iter(chunks().list_search_indexes(settings.vector_index)), None)


def ensure_vector_index(timeout_s: int = 180) -> bool:
    """Create the vector index if missing and wait until it is queryable.

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
            if _find_vector_index() is None:
                chunks().create_search_index(
                    SearchIndexModel(
                        definition=VECTOR_INDEX_DEFINITION, name=settings.vector_index, type="vectorSearch"
                    )
                )
                log.info("Created vector search index %s", settings.vector_index)
            index = _find_vector_index()
            if index and index.get("queryable"):
                log.info("Vector search index is queryable")
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


def wait_until_searchable(doc_id: str, expected: int, timeout_s: int = 120) -> bool:
    """mongot syncs asynchronously: wait until the vector index sees all `expected` chunks of the document."""
    if not vector_index_ready:
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
            res = list(chunks().aggregate(pipeline))
            if res and res[0]["n"] >= expected:
                return True
        except OperationFailure as e:
            log.warning("waiting for vector index: %s", e)
        time.sleep(1)
    return False
