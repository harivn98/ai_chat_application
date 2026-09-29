"""Hybrid retrieval: BM25 (lexical) + vector search (dense), fused with Reciprocal Rank Fusion, then (if the mode
reranks) re-ordered by the cross-encoder.

search() is what the chat uses. It runs three stages, which the evaluation runs one by one to time them:
candidate_rankings() (both searches), fuse() (RRF, plus each chunk's text) and reranker.rerank().
"""
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import snowballstemmer
from rank_bm25 import BM25Plus

from . import db, reranker
from .config import settings
from .modes import Mode

Ranking = list[tuple[int, float]]  # (chunk index, score), best first

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


# ---------------------------------------------------------------- per-document chunk cache
@dataclass
class _DocumentChunks:
    """A document's chunks (without vectors) and the BM25 index over their indexed content."""
    chunks: list[dict]      # in document order
    terms: list[set[str]]   # each chunk's stemmed terms
    bm25: BM25Plus


class _ChunkCache:
    """One _DocumentChunks per document, loaded from MongoDB on first use (LRU)."""

    def __init__(self, maxsize: int = 16):
        self._data: OrderedDict[str, _DocumentChunks] = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def get(self, doc_id: str) -> _DocumentChunks | None:
        with self._lock:
            if doc_id in self._data:
                self._data.move_to_end(doc_id)
                return self._data[doc_id]
        chunks = db.load_chunks(doc_id)
        if not chunks:
            return None
        tokens = [tokenize(c["content"]) for c in chunks]
        # BM25+ keeps IDF positive even for short documents with only a few chunks,
        # where classic Okapi IDF collapses to zero/negative values.
        doc = _DocumentChunks(chunks, [set(t) for t in tokens], BM25Plus([t or ["_"] for t in tokens]))
        with self._lock:
            self._data[doc_id] = doc
            if len(self._data) > self._maxsize:
                self._data.popitem(last=False)
        return doc

    def drop(self, doc_id: str) -> None:
        with self._lock:
            self._data.pop(doc_id, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_cache = _ChunkCache()


def drop_document(doc_id: str) -> None:
    """Forget a document's cached chunks and BM25 index (after its chunks are deleted)."""
    _cache.drop(doc_id)


def drop_all_documents() -> None:
    """Forget every cached document (after all documents are deleted)."""
    _cache.clear()


def preload_document(doc_id: str) -> None:
    """(Re)build a document's cached chunks and BM25 index now, so its first question doesn't wait for them."""
    _cache.drop(doc_id)
    _cache.get(doc_id)


# ---------------------------------------------------------------- the two searches
def bm25_search(doc_id: str, query: str, k: int) -> Ranking:
    doc = _cache.get(doc_id)
    query_terms = tokenize(query)
    if doc is None or not query_terms:
        return []
    scores = doc.bm25.get_scores(query_terms)
    query_set = set(query_terms)
    order = np.argsort(scores)[::-1]
    # BM25+ gives every chunk a small baseline score; only keep chunks sharing a query term
    hits = [(doc.chunks[i]["index"], float(scores[i])) for i in order if doc.terms[i] & query_set]
    return hits[:k]


def vector_search(doc_id: str, query: str, k: int, mode: Mode) -> Ranking:
    """The k chunks closest to the query by cosine similarity, at or above the mode's minimum score if it has one."""
    query_vector = mode.embed_query(query)
    hits = db.vector_search(doc_id, query_vector, k, mode) or _cosine_search(doc_id, query_vector, k, mode)
    if mode.vector_min_score is None:
        return hits
    return [(idx, score) for idx, score in hits if score >= mode.vector_min_score]


def _cosine_search(doc_id: str, query_vector: np.ndarray, k: int, mode: Mode) -> Ranking:
    """In-process vector search, for deployments without a vector index (or when it fails)."""
    vectors = db.chunk_vectors(doc_id, mode)
    if not vectors:
        return []
    matrix = np.asarray([vector for _, vector in vectors], dtype=np.float32)
    similarities = matrix @ query_vector
    order = np.argsort(similarities)[::-1][:k]
    return [(vectors[i][0], float(similarities[i])) for i in order]


# ---------------------------------------------------------------- stages
def candidate_rankings(doc_id: str, query: str, mode: Mode) -> dict[str, Ranking]:
    """Each search's hits before fusion: the BM25 top bm25_candidates, and the vector top vector_candidates
    (at or above vector_min_score, if the mode has one)."""
    return {
        "bm25": bm25_search(doc_id, query, mode.bm25_candidates),
        "vector": vector_search(doc_id, query, mode.vector_candidates, mode),
    }


def rrf_fuse(rankings: dict[str, Ranking], k: int, rrf_k: int) -> list[dict]:
    fused: dict[int, dict] = {}
    for name, hits in rankings.items():
        for rank, (idx, score) in enumerate(hits, start=1):
            entry = fused.setdefault(idx, {"index": idx, "rrf": 0.0})
            entry["rrf"] += 1.0 / (rrf_k + rank)
            entry[f"{name}_rank"] = rank
            entry[f"{name}_score"] = round(score, 4)
    return sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:k]


def fuse(doc_id: str, rankings: dict[str, Ranking], k: int) -> list[dict]:
    """The top k chunks by Reciprocal Rank Fusion of the rankings, as passages: each search's rank and score, the
    fused rank, and the chunk's section, text and position."""
    fused = rrf_fuse(rankings, k, settings.rrf_k)
    doc = _cache.get(doc_id)
    by_index = {c["index"]: c for c in doc.chunks} if doc else {}
    passages = []
    for rank, hit in enumerate(fused, start=1):
        chunk = by_index.get(hit["index"])
        if chunk:
            passages.append({
                **hit,
                "fused_rank": rank,
                "section": chunk.get("section", ""),
                "text": chunk["text"],
                "content": chunk["content"],  # the indexed text (context + heading path + chunk text)
                "start": chunk.get("start"),  # position in the Markdown; None for uploads from before it was stored
                "end": chunk.get("end"),
            })
    return passages


def search(doc_id: str, query: str, mode: Mode) -> list[dict]:
    """The passages sent to the LLM: the fused top_k, or, if the mode reranks, the top_k the cross-encoder picks
    from all the fused chunks."""
    rankings = candidate_rankings(doc_id, query, mode)
    if not mode.reranker:
        return fuse(doc_id, rankings, mode.top_k)
    candidates = fuse(doc_id, rankings, mode.candidate_pool)
    return reranker.rerank(query, candidates, mode.top_k)
