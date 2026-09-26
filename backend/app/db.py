"""MongoDB access + Atlas Vector Search index management."""
import logging
import time
from functools import lru_cache

from pymongo import ASCENDING, MongoClient
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from .config import settings

log = logging.getLogger("db")


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


def _find_index():
    for idx in chunks().list_search_indexes(settings.vector_index):
        return idx
    return None


def ensure_vector_index(timeout_s: int = 180) -> bool:
    """Create the vector index if missing and wait until it is queryable.

    Returns False (and the app falls back to in-process cosine search) when the
    MongoDB deployment has no search support, e.g. a plain `mongo` image.
    """
    if "chunks" not in db().list_collection_names():
        db().create_collection("chunks")
    chunks().create_index([("doc_id", ASCENDING), ("index", ASCENDING)])
    documents().create_index([("created_at", ASCENDING)])

    deadline = time.time() + timeout_s
    while True:
        try:
            if _find_index() is None:
                chunks().create_search_index(
                    SearchIndexModel(
                        definition=VECTOR_INDEX_DEFINITION, name=settings.vector_index, type="vectorSearch"
                    )
                )
                log.info("Created vector search index %s", settings.vector_index)
            idx = _find_index()
            if idx and idx.get("queryable"):
                log.info("Vector search index is queryable")
                return True
        except OperationFailure as e:
            # mongot can take a few seconds to come up after mongod
            log.warning("Vector index not ready yet: %s", e)
        if time.time() > deadline:
            log.error("Vector search index unavailable; using in-process cosine fallback")
            return False
        time.sleep(3)
