"""Follow-up rewriting: turn a follow-up question into one that can be understood without the chat history.

Retrieval (BM25, vector search, the reranker) and the pre-judge only see the question, so a follow-up such as
"What about its limitations?" would be searched as written. Before retrieval, the mode's answering model
(LLM_MODEL in private mode, CLOUD_LLM_MODEL in cloud mode) rewrites it with the recent chat turns, e.g. into
"What are the limitations of the proposed attention mechanism?". In private mode that is the model already on the
GPU, so no model is swapped. When answering, the model still gets the original question together with the history.

The first question of a chat has no history and is used as typed. If the rewrite fails, the original question is
used (fails open, like the pre-judge).
"""
import logging

from .config import settings
from .modes import Mode

log = logging.getLogger("rewrite")

PROMPT = """Here is a conversation about a document, followed by a follow-up question.
<conversation>
{conversation}
</conversation>
<question>
{question}
</question>
Rewrite the follow-up question as a standalone question that can be understood without the conversation: replace words such as "it", "they", "that" or "the second one" with what they refer to in the conversation. Keep the meaning and the wording of the question otherwise; do not answer it. If the question is already standalone, repeat it unchanged. Answer only with the question and nothing else."""

TURN_CHARS = 1500   # characters of each earlier message the rewrite reads (answers can be long)
MAX_CHARS = 1000    # a longer reply is not a question: use the original


def standalone_question(question: str, history: list[dict], mode: Mode) -> str:
    """The question rewritten so it stands on its own; the question itself when there is no history."""
    turns = [h for h in history[-settings.history_turns:] if h.get("role") in {"user", "assistant"} and h.get("content")]
    if not settings.query_rewrite_enabled or not turns:
        return question
    conversation = "\n\n".join(f"{h['role'].capitalize()}: {h['content'][:TURN_CHARS]}" for h in turns)
    try:
        reply = mode.rewrite_question(PROMPT.format(conversation=conversation, question=question))
    except Exception:  # noqa: BLE001
        log.exception("Rewriting the follow-up failed; searching with the question as asked")
        return question
    rewritten = " ".join(reply.split()).strip("\"'")
    if not rewritten or len(rewritten) > MAX_CHARS:
        log.warning("Rewrite gave no usable question (%r); searching with the question as asked", reply[:80])
        return question
    if rewritten != question:
        log.info("Follow-up %r rewritten as %r", question, rewritten)
    return rewritten
