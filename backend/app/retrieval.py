"""Hybrid retrieval: BM25 (lexical) + bge-small (dense), fused with Reciprocal Rank Fusion."""
import logging
import re
import threading
from collections import OrderedDict

import numpy as np
import snowballstemmer
from pymongo.errors import OperationFailure
from rank_bm25 import BM25Plus

from . import db
from .config import settings
from .embeddings import embed_query

log = logging.getLogger("retrieval")

STOPWORDS = set(
    """a an and are as at be but by for from has have he her his i if in into is it its
    me my of on or our she so than that the their them then there these they this to
    was we were what when where which while who whom why will with you your do does did
    can could should would about how""".split()
)
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-_][a-z0-9]+)*")
_stemmer = snowballstemmer.stemmer("english")
_stem_lock = threading.Lock()


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stopwords, Snowball-stem (so 'replacing' matches 'replacement')."""
    words = [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
    with _stem_lock:  # the stemmer object is not thread-safe
        return _stemmer.stemWords(words)


# ---------------------------------------------------------------- BM25 cache
class _BM25Cache:
    """One BM25 index per document, built from Mongo on first use (LRU)."""

    def __init__(self, maxsize: int = 16):
        self._data: OrderedDict[str, tuple[BM25Plus, list[dict]]] = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def get(self, doc_id: str) -> tuple[BM25Plus | None, list[dict]]:
        with self._lock:
            if doc_id in self._data:
                self._data.move_to_end(doc_id)
                return self._data[doc_id]
        rows = list(
            db.chunks().find({"doc_id": doc_id}, {"embedding": 0}).sort("index", 1)
        )
        if not rows:
            return None, []
        # BM25+ keeps IDF positive even for short documents with only a few chunks,
        # where classic Okapi IDF collapses to zero/negative values.
        for r in rows:
            r["_tokens"] = set(tokenize(r["content"]))
        bm25 = BM25Plus([tokenize(r["content"]) or ["_"] for r in rows])
        with self._lock:
            self._data[doc_id] = (bm25, rows)
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)
        return bm25, rows

    def drop(self, doc_id: str):
        with self._lock:
            self._data.pop(doc_id, None)


bm25_cache = _BM25Cache()
vector_index_ready = False  # set by main.py at startup


def bm25_search(doc_id: str, query: str, k: int) -> list[tuple[int, float]]:
    bm25, rows = bm25_cache.get(doc_id)
    q = tokenize(query)
    if bm25 is None or not q:
        return []
    scores = bm25.get_scores(q)
    qset = set(q)
    order = np.argsort(scores)[::-1]
    # BM25+ gives every chunk a small baseline score; only keep chunks sharing a query term
    hits = [(rows[i]["index"], float(scores[i])) for i in order if rows[i]["_tokens"] & qset]
    return hits[:k]


def _vector_search_mongo(doc_id: str, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
    pipeline = [
        {
            "$vectorSearch": {
                "index": settings.vector_index,
                "path": "embedding",
                "queryVector": qvec.tolist(),
                "numCandidates": max(k * 10, 100),
                "limit": k,
                "filter": {"doc_id": doc_id},
            }
        },
        {"$project": {"_id": 0, "index": 1, "score": {"$meta": "vectorSearchScore"}}},
    ]
    return [(r["index"], float(r["score"])) for r in db.chunks().aggregate(pipeline)]


def _vector_search_local(doc_id: str, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
    rows = list(db.chunks().find({"doc_id": doc_id}, {"index": 1, "embedding": 1}))
    if not rows:
        return []
    mat = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
    sims = mat @ qvec
    order = np.argsort(sims)[::-1][:k]
    return [(rows[i]["index"], float(sims[i])) for i in order]


def vector_search(doc_id: str, query: str, k: int) -> list[tuple[int, float]]:
    qvec = embed_query(query)
    if vector_index_ready:
        try:
            hits = _vector_search_mongo(doc_id, qvec, k)
            if hits:
                return hits
        except OperationFailure as e:
            log.warning("$vectorSearch failed, using local fallback: %s", e)
    return _vector_search_local(doc_id, qvec, k)


def rrf_fuse(rankings: dict[str, list[tuple[int, float]]], k: int, rrf_k: int) -> list[dict]:
    fused: dict[int, dict] = {}
    for name, hits in rankings.items():
        for rank, (idx, score) in enumerate(hits, start=1):
            entry = fused.setdefault(idx, {"index": idx, "rrf": 0.0})
            entry["rrf"] += 1.0 / (rrf_k + rank)
            entry[f"{name}_rank"] = rank
            entry[f"{name}_score"] = round(score, 4)
    return sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:k]


def hybrid_search(doc_id: str, query: str) -> list[dict]:
    n = settings.candidates_per_retriever
    rankings = {
        "bm25": bm25_search(doc_id, query, n),
        "vector": vector_search(doc_id, query, n),
    }
    fused = rrf_fuse(rankings, settings.top_k, settings.rrf_k)
    _, rows = bm25_cache.get(doc_id)
    by_index = {r["index"]: r for r in rows}
    results = []
    for f in fused:
        row = by_index.get(f["index"])
        if row:
            results.append({**f, "section": row.get("section", ""), "text": row["text"]})
    return results
