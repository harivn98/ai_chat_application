"""The QASPER dataset (allenai/qasper): download, papers as Markdown, and the official scoring functions."""
import io
import json
import re
import string
import tarfile
from collections import Counter

import httpx

from .config import settings

S3 = "https://qasper-dataset.s3.us-west-2.amazonaws.com"
SPLITS = {
    "test": (f"{S3}/qasper-test-and-evaluator-v0.3.tgz", "qasper-test-v0.3.json"),
    "validation": (f"{S3}/qasper-train-dev-v0.3.tgz", "qasper-dev-v0.3.json"),
}
ANSWER_TYPES = ("extractive", "abstractive", "boolean", "none")


def load_split(split: str) -> dict:
    """Paper id -> paper, downloaded on first use and cached under DATA_DIR/eval."""
    url, member = SPLITS[split]
    cache = settings.data_dir / "eval" / member
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading the QASPER {split} split (first run only)…", flush=True)
        r = httpx.get(url, follow_redirects=True, timeout=300)
        r.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
            cache.write_bytes(tar.extractfile(member).read())
    return json.loads(cache.read_text(encoding="utf-8"))


def one_line(text: str) -> str:
    return " ".join((text or "").split())


def paper_markdown(paper: dict) -> tuple[str, list[str]]:
    """Paper -> Markdown (nested 'A ::: B' section names become nested headings) + its paragraphs."""
    abstract = one_line(paper["abstract"])
    lines = [f"# {one_line(paper['title'])}", "", "## Abstract", "", abstract, ""]
    paragraphs = [abstract] if abstract else []
    prev: list[str] = []
    for sec in paper["full_text"]:
        parts = [p.strip() for p in (sec["section_name"] or "").split(":::") if p.strip()]
        same = 0
        while same < min(len(parts), len(prev)) and parts[same] == prev[same]:
            same += 1
        for depth in range(same, len(parts)):
            lines += [f"{'#' * min(depth + 2, 6)} {parts[depth]}", ""]
        prev = parts
        for para in map(one_line, sec["paragraphs"]):
            if para:
                lines += [para, ""]
                paragraphs.append(para)
    return "\n".join(lines), paragraphs


def paragraphs_in(texts: list[str], paragraphs: list[str]) -> list[str]:
    """Paper paragraphs that (fully or partly) appear in the given chunk texts."""
    blocks = [b for t in texts for b in map(one_line, t.split("\n\n")) if len(b) >= 40]
    return [p for p in paragraphs if any(b in p for b in blocks)]


def references(qa: dict) -> list[dict]:
    """Gold answers per annotator, as in the official evaluator with --text_evidence_only."""
    refs = []
    for annotation in qa["answers"]:
        a = annotation["answer"]
        if a["unanswerable"]:
            refs.append({"answer": "Unanswerable", "evidence": [], "type": "none"})
            continue
        if a["extractive_spans"]:
            answer, kind = ", ".join(a["extractive_spans"]), "extractive"
        elif a["free_form_answer"]:
            answer, kind = a["free_form_answer"], "abstractive"
        elif a["yes_no"] is not None:
            answer, kind = ("Yes" if a["yes_no"] else "No"), "boolean"
        else:
            continue
        evidence = [one_line(t) for t in a["evidence"] if "FLOAT SELECTED" not in t]
        refs.append({"answer": answer, "evidence": evidence, "type": kind})
    return refs


# ------------------------------------------------------------------ official QASPER scoring (qasper_evaluator.py)
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def token_f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def paragraph_f1_score(prediction, ground_truth):
    if not ground_truth and not prediction:
        return 1.0
    num_same = len(set(ground_truth).intersection(set(prediction)))
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)
