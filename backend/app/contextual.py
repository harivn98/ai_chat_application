"""Contextual Retrieval: a small LLM (the mode's context model) writes a short context for every chunk.

For each chunk, the context model reads the document (or, for long documents, the window of it
that contains the chunk) and writes one or two sentences situating the chunk, e.g. which section
and topic it belongs to. That context is prepended to the text that is embedded and BM25-indexed;
the chunk text shown to the answering LLM and in the UI is unchanged.

The document is always sent first and the chunk last, so consecutive requests share a long
prompt prefix and Ollama can reuse its KV cache instead of re-reading the document every time
(and Gemini its implicit prompt cache).
"""
import logging
import threading
from collections.abc import Callable

import httpx

from . import ollama
from .chunker import Chunk
from .config import settings
from .modes import Mode
from .openrouter import OpenRouterError

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


def _windows(chunks: list[Chunk], full_text: str, num_ctx: int) -> list[tuple[str, list[Chunk]]]:
    """Group chunks under the document text each group should be situated in.

    Short documents: one window, the whole document. Long documents: consecutive chunks are
    grouped until the window budget is used, and each window starts with the document head.
    """
    budget = (num_ctx - RESERVED_TOKENS) * CHARS_PER_TOKEN
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


def _generate(document: str, chunk: str, mode: Mode) -> str:
    try:
        reply = mode.write_context(PROMPT.format(document=document, chunk=chunk))
    except httpx.HTTPError as e:
        raise ContextError(f"Could not reach the context model {mode.context_model}: {e}") from e
    except OpenRouterError as e:
        raise ContextError(f"Context model {mode.context_model} failed: {e}") from e
    except ollama.OllamaError as e:
        if e.status == 404:
            raise ContextError(
                f"Context model {settings.context_model} is not installed. Pull it with "
                f"`docker compose exec ollama ollama pull {settings.context_model}` "
                f"or set CONTEXTUAL_EMBEDDING=false."
            ) from e
        raise ContextError(str(e)) from e
    return " ".join(reply.split())


_active = 0                  # local contextualize() calls in progress (uploads run in parallel threads)
_active_lock = threading.Lock()


def contextualize(
    full_text: str, chunks: list[Chunk], mode: Mode, on_progress: Callable[[int, int], None] | None = None
) -> list[str]:
    """Return one context sentence per chunk (same order as `chunks`)."""
    global _active
    with _active_lock:
        _active += mode.local
    try:
        contexts: dict[int, str] = {}
        done = 0
        for document, group in _windows(chunks, full_text, mode.context_num_ctx):
            for c in group:
                contexts[c.index] = _generate(document, c.content, mode)
                done += 1
                if on_progress:
                    on_progress(done, len(chunks))
        return [contexts[c.index] for c in chunks]
    finally:
        with _active_lock:
            _active -= mode.local


def release_context_model() -> str | None:
    """After contextualizing locally, unload the context model so the GPU is free for the answering model.

    Nothing else uses the context model (the answering model also does the pre-judge). The caller then
    loads the answering model right away, so it is ready before the first question arrives.
    Returns what was done, or None if another upload is still contextualizing (it releases the model when it ends).
    """
    with _active_lock:
        if _active:
            return None
        if settings.context_model == settings.llm_model:
            return f"kept {settings.context_model} (it is also the answering model)"
        try:
            ollama.unload(settings.context_model)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not unload %s: %s", settings.context_model, e)
            return None
    action = f"unloaded {settings.context_model}"
    log.info("After contextualizing: %s", action)
    return action


def indexed_content(chunk: Chunk, context: str) -> str:
    """Text that gets embedded and BM25-indexed: generated context + heading path + chunk text."""
    return f"{context}\n\n{chunk.content}" if context else chunk.content
