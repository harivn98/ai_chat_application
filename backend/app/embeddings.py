"""bge-small-en-v1.5 embeddings (384-d, L2-normalised -> cosine == dot product)."""
import threading
from functools import lru_cache

import numpy as np

from .config import settings

_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embed_model, device="cpu")
    dim = model.get_sentence_embedding_dimension()
    if dim != settings.embed_dim:
        raise RuntimeError(f"Embedding dim mismatch: model={dim}, EMBED_DIM={settings.embed_dim}")
    return model


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    with _lock:  # the model is shared across request threads
        vecs = get_model().encode(
            texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
        )
    return np.asarray(vecs, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    with _lock:
        vec = get_model().encode(
            [settings.query_instruction + query], normalize_embeddings=True, show_progress_bar=False
        )[0]
    return np.asarray(vec, dtype=np.float32)
