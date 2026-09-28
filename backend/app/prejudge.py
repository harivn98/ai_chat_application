"""Pre-judge: decide whether the retrieved passages can answer the question before generating an answer.

The verdict is ALL (the passages hold the answer), PARTIAL (only part of it: the answer says what is missing)
or NONE (answer generation is skipped and the user gets NOT_ENOUGH_CONTENT instead of a guess).

Private mode: the answering model (LLM_MODEL) replies YES or NO (ALL or NONE). It uses the same model and the
same num_ctx as answer generation, so the model that is already on the GPU does both steps and Ollama never has
to swap or reload a model between them.
Cloud pre-judge mode: CLOUD_PREJUDGE_MODEL (Gemini Flash-Lite) replies ALL, PARTIAL or NONE.
"""
import logging
import re

from .modes import Mode

log = logging.getLogger("prejudge")

ALL, PARTIAL, NONE = "all", "partial", "none"
NOT_ENOUGH_CONTENT = "There isn't enough content in the document to answer this question."

# Strict: the passages must contain the specific information asked for, not just be on the same topic,
# so on-topic questions the document doesn't actually answer are caught. Yes/no questions and answers that
# follow directly from stated facts still count, so answerable questions aren't blocked for their wording.
YES_NO_PROMPT = """You check whether passages retrieved from a document contain the answer to a question.

Passages:
{passages}

Question: {question}

Reply YES if the passages contain the specific information the question asks for, either stated directly or following clearly from what they state (for a yes/no question, facts that settle the yes or no count).
Reply NO if the passages are only about the same topic but do not contain that specific information, or if answering would require guessing or knowledge from outside the passages.
Reply with exactly one word: YES or NO."""

LEVELS_PROMPT = """You check how much of the answer to a question is contained in passages retrieved from a document.

Passages:
{passages}

Question: {question}

Reply ALL if the passages contain all the specific information the question asks for, either stated directly or following clearly from what they state (for a yes/no question, facts that settle the yes or no count).
Reply PARTIAL if they contain some of the specific information asked for, but not all of it (for example, one of several requested items, or a related detail without the main fact).
Reply NONE if they contain none of the specific information asked for: passages that are only about the same topic count as NONE, as does anything that would require guessing or knowledge from outside the passages.
Reply with exactly one word: ALL, PARTIAL or NONE."""


def verdict(question: str, passages: list[dict], mode: Mode) -> str:
    """ALL, PARTIAL or NONE. Fails open (ALL) if the model gives no verdict."""
    if not passages:
        return NONE
    text = "\n\n".join(f"[{i}] {p['text']}" for i, p in enumerate(passages, start=1))
    if mode.local:
        reply = mode.judge_passages(YES_NO_PROMPT.format(passages=text, question=question)).strip().upper()
        m = re.match(r"\W*(YES|NO)\b", reply)
        found = m and {"YES": ALL, "NO": NONE}[m.group(1)]
    else:
        reply = mode.judge_passages(LEVELS_PROMPT.format(passages=text, question=question)).strip().upper()
        m = re.match(r"\W*(ALL|PARTIAL|NONE)\b", reply)
        found = m and m.group(1).lower()
    if not found:
        log.warning("Pre-judge gave no verdict (%r); answering anyway", reply[:50])
        return ALL
    return found
