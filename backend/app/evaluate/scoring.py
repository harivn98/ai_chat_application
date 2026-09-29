"""How one answer is scored against a question's annotator references (qasper.references): the official QASPER
answer and evidence F1, retrieval recall, and the LLM judge's prompt and verdict."""
import re

from .qasper import paragraph_f1_score, paragraphs_in, token_f1_score

CITE_RE = re.compile(r"\[(\d+)\]")
NOT_FOUND_RE = re.compile(
    r"could(?: not|n't) find|not (?:mentioned|found|provided|specified|stated|available|included)"
    r"|does(?: not|n't) (?:mention|say|specify|state|provide|contain|include)|no information",
    re.IGNORECASE,
)

JUDGE_PROMPT = """You are grading an answer to a question about a research paper.

Question:
{question}

Reference answers (written by different annotators; any one of them is acceptable):
{references}

Candidate answer:
{response}

Is the candidate answer correct, i.e. does it convey the same key information as at least one reference answer?
- Extra correct detail is fine. Missing the key information or contradicting it is not.
- If a reference says the paper does not answer the question, the candidate is correct only if it also says the answer is not in the paper.
- Ignore citation markers such as [1].

Reply with exactly one word: CORRECT or INCORRECT."""


def cited_passages(response: str, count: int) -> list[int]:
    """The passage numbers the response cites as [n], sorted; numbers outside 1..count are ignored."""
    return sorted({int(n) for n in CITE_RE.findall(response) if 1 <= int(n) <= count})


def prediction(response: str, answered: bool, cited: list[int], passages: list[dict],
               paragraphs: list[str]) -> tuple[str, list[str]]:
    """What QASPER scores: the answer without citation markers, and as evidence the paper paragraphs inside the
    cited passages. "Unanswerable" with no evidence when the pre-judge skipped the answer, or when the answer says
    it found nothing and cites no passage."""
    if not answered or (NOT_FOUND_RE.search(response) and not cited):
        return "Unanswerable", []
    return CITE_RE.sub("", response), paragraphs_in([passages[n - 1]["text"] for n in cited], paragraphs)


def answer_f1(predicted_answer: str, refs: list[dict]) -> tuple[float, str]:
    """Token F1 against the best-matching reference, and that reference's answer type."""
    return max(((token_f1_score(predicted_answer, r["answer"]), r["type"]) for r in refs), key=lambda x: x[0])


def evidence_f1(predicted_evidence: list[str], refs: list[dict]) -> float:
    return max(paragraph_f1_score(predicted_evidence, r["evidence"]) for r in refs)


def evidence_recall(texts: list[str], paragraphs: list[str], refs: list[dict]) -> float | None:
    """Share of a reference's evidence paragraphs that appear in the texts, for the best reference; None when no
    reference has text evidence (e.g. the question is unanswerable)."""
    found = set(paragraphs_in(texts, paragraphs))
    recalls = [len(found & set(r["evidence"])) / len(r["evidence"]) for r in refs if r["evidence"]]
    return max(recalls) if recalls else None


def judge_prompt(question: str, refs: list[dict], response: str) -> str:
    references = "\n".join(
        "- The paper does not answer this question." if r["type"] == "none" else f"- {r['answer']}" for r in refs
    )
    return JUDGE_PROMPT.format(question=question, references=references, response=response)


def parse_judge_verdict(text: str) -> bool | None:
    """True for CORRECT, False for INCORRECT (the judge's last such word), None if it gave neither."""
    m = re.findall(r"\b(INCORRECT|CORRECT)\b", text.upper())
    return m[-1] == "CORRECT" if m else None
