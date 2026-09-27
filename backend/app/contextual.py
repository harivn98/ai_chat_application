"""Contextual Retrieval: a small local LLM writes a short context for every chunk.

For each chunk, the context model reads the document (or, for long documents, the window of it
that contains the chunk) and writes one or two sentences situating the chunk, e.g. which section
and topic it belongs to. That context is prepended to the text that is embedded and BM25-indexed;
the chunk text shown to the answering LLM and in the UI is unchanged.

The document is always sent first and the chunk last, so consecutive requests share a long
prompt prefix and Ollama can reuse its KV cache instead of re-reading the document every time.
"""
import logging
from collections.abc import Callable

import httpx

from .chunker import Chunk
from .config import settings

log = logging.getLogger("contextual")

# Prompt from Anthropic's "Introducing Contextual Retrieval"
PROMPT = """<document>
{document}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

CHARS_PER_TOKEN = 3        # conservative estimate, so windows never overflow num_ctx
RESERVED_TOKENS = 1024     # prompt wrapper + chunk + the generated context
HEAD_CHARS = 1500          # document start (title, abstract) kept in every window of a long document


class ContextError(RuntimeError):
    pass


def _windows(chunks: list[Chunk], full_text: str) -> list[tuple[str, list[Chunk]]]:
    """Group chunks under the document text each group should be situated in.

    Short documents: one window, the whole document. Long documents: consecutive chunks are
    grouped until the window budget is used, and each window starts with the document head.
    """
    budget = (settings.context_num_ctx - RESERVED_TOKENS) * CHARS_PER_TOKEN
    if len(full_text) <= budget:
        return [(full_text, chunks)]

    head = full_text[:HEAD_CHARS]
    windows, group, size = [], [], len(head)
    for c in chunks:
        piece = len(c.content) + 2
        if group and size + piece > budget:
            windows.append((head + "\n\n[...]\n\n" + "\n\n".join(g.content for g in group), group))
            group, size = [], len(head)
        group.append(c)
        size += piece
    if group:
        windows.append((head + "\n\n[...]\n\n" + "\n\n".join(g.content for g in group), group))
    return windows


def _generate(document: str, chunk: str) -> str:
    payload = {
        "model": settings.context_model,
        "messages": [{"role": "user", "content": PROMPT.format(document=document, chunk=chunk)}],
        "stream": False,
        "think": False,
        "keep_alive": settings.llm_keep_alive,
        "options": {"temperature": 0, "num_ctx": settings.context_num_ctx, "num_predict": 120},
    }
    timeout = httpx.Timeout(connect=10, read=settings.llm_load_timeout, write=60, pool=10)
    try:
        r = httpx.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=timeout)
    except httpx.HTTPError as e:
        raise ContextError(f"Could not reach Ollama for {settings.context_model}: {e}") from e
    if r.status_code == 404:
        raise ContextError(
            f"Context model {settings.context_model} is not installed. Pull it with "
            f"`docker compose exec ollama ollama pull {settings.context_model}` "
            f"or set CONTEXTUAL_EMBEDDING=false."
        )
    if r.status_code != 200:
        raise ContextError(f"Ollama error {r.status_code}: {r.text[:300]}")
    return " ".join(r.json()["message"]["content"].split())


def contextualize(
    full_text: str, chunks: list[Chunk], on_progress: Callable[[int, int], None] | None = None
) -> list[str]:
    """Return one context sentence per chunk (same order as `chunks`)."""
    contexts: dict[int, str] = {}
    done = 0
    for document, group in _windows(chunks, full_text):
        for c in group:
            contexts[c.index] = _generate(document, c.content)
            done += 1
            if on_progress:
                on_progress(done, len(chunks))
    return [contexts[c.index] for c in chunks]


def indexed_content(chunk: Chunk, context: str) -> str:
    """Text that gets embedded and BM25-indexed: generated context + heading path + chunk text."""
    return f"{context}\n\n{chunk.content}" if context else chunk.content
