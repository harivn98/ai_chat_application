"""The evaluation's four phases. Each runs with one model loaded, so Ollama doesn't swap models for every paper:
(1) ingest the papers (context model), (2) retrieve + pre-judge and (3) answer + score (answering model, which also
does the pre-judge), (4) judge (JUDGE_MODEL)."""
import logging
import time
from dataclasses import dataclass, field

from .. import db, ollama, prejudge, reranker, retrieval
from ..answer_prompt import build_messages
from ..chunker import chunk_markdown
from ..config import settings
from ..contextual import contextualize, indexed_content
from ..modes import Mode
from . import scoring
from .console import Status, duration, say, short
from .qasper import paper_markdown, references

log = logging.getLogger("evaluate")

Question = tuple[str, dict]  # (paper id, QASPER question)

VERDICT_TEXT = {prejudge.ALL: "ALL, will answer", prejudge.PARTIAL: "PARTIAL, will answer and say what is missing",
                prejudge.NONE: "NONE, not enough content"}


def _load_model(model: str, num_ctx: int) -> None:
    """Load a model into Ollama before timing anything; on CPU this alone can take several minutes.

    num_ctx must be the one the model's requests use, or Ollama loads it again on the first (timed) request.
    """
    st = Status(f"Loading {model} into Ollama (instant if already loaded; minutes if Ollama runs on CPU)…")
    ollama.load(model, num_ctx)
    st.done(f"{model} loaded in {duration(st.elapsed)}")


# ------------------------------------------------------------------ phase 1: ingest the papers
@dataclass
class IngestedPapers:
    paragraphs: dict[str, list[str]] = field(default_factory=dict)  # paper id -> its paragraphs
    errors: dict[str, str] = field(default_factory=dict)  # paper id -> why it failed to ingest
    context_times: list[float] = field(default_factory=list)  # seconds per paper
    embedding_times: list[float] = field(default_factory=list)  # seconds per paper


def ingest_papers(data: dict, paper_ids: list[str], doc_ids: dict[str, str], mode: Mode) -> IngestedPapers:
    say(f"\n--- Phase 1/4: ingesting papers "
        f"({'with contexts from ' + mode.context_model if settings.contextual_embedding else 'no contexts'}) ---")
    if settings.contextual_embedding and mode.local:
        _load_model(settings.context_model, settings.context_num_ctx)
    ingested = IngestedPapers()
    for n, pid in enumerate(paper_ids, start=1):
        paper = data[pid]
        markdown, paragraphs = paper_markdown(paper)
        say(f"\n[paper {n}/{len(paper_ids)}] {short(paper['title'])} ({len(paper['qas'])} questions)")
        try:
            _ingest_paper(doc_ids[pid], markdown, ingested, mode)
            ingested.paragraphs[pid] = paragraphs
        except Exception as e:  # noqa: BLE001
            log.exception("Ingesting paper %s failed", pid)
            say(f"  FAILED to ingest paper: {e}")
            ingested.errors[pid] = str(e)
    return ingested


def _ingest_paper(doc_id: str, markdown: str, ingested: IngestedPapers, mode: Mode) -> None:
    """Chunk, add contexts, embed and store one paper like an upload, timing the context and embedding steps."""
    chunks = chunk_markdown(markdown, settings.chunk_size, settings.chunk_overlap)
    contexts = [""] * len(chunks)
    if settings.contextual_embedding:
        st = Status("  adding context…")
        t0 = time.perf_counter()
        contexts = contextualize(markdown, chunks, mode, lambda done, total: st.set(f"{done}/{total} chunks"))
        ingested.context_times.append(time.perf_counter() - t0)
        st.done(f"  context: {len(chunks)} chunks in {duration(ingested.context_times[-1])} "
                f"(e.g. \"{short(contexts[len(contexts) // 2], 80)}\")")
    contents = [indexed_content(c, ctx) for c, ctx in zip(chunks, contexts)]

    st = Status("  embedding…")
    t0 = time.perf_counter()
    vectors = mode.embed_passages(contents)
    ingested.embedding_times.append(time.perf_counter() - t0)
    st.done(f"  embedding: {len(chunks)} chunks in {duration(ingested.embedding_times[-1])}")
    db.replace_chunks(doc_id, chunks, contexts, contents, vectors, mode)

    st = Status("  waiting for the vector index…")
    db.wait_until_searchable(doc_id, len(chunks), mode)  # index sync is excluded from the timings
    st.done(f"  vector index ready in {duration(st.elapsed)}")


# ------------------------------------------------------------------ phase 2: retrieve + pre-judge
@dataclass
class Retrieved:
    """One question's retrieval + pre-judge result."""
    question: str
    passages: list[dict]  # what the pre-judge and the answering model get
    candidates: list[dict]  # the fused chunks the reranker chose the passages from (without it: the passages)
    verdict: str  # the pre-judge's ALL / PARTIAL / NONE; ALL when the mode doesn't pre-judge
    retrieval_s: float
    rerank_s: float | None
    prejudge_s: float | None
    bm25_hits: int  # chunks BM25 returned (at most BM25_CANDIDATES)
    vector_hits: int  # vector hits, after the mode's score cutoff if any (at most VECTOR_CANDIDATES)
    vector_score_min: float | None  # cosine range of those hits
    vector_score_max: float | None

    @property
    def can_answer(self) -> bool:
        return self.verdict != prejudge.NONE


@dataclass
class Retrievals:
    """Phase 2's results for the questions of every ingested paper."""
    results: dict[str, Retrieved] = field(default_factory=dict)  # question id -> its result
    errors: dict[str, str] = field(default_factory=dict)  # question id -> why retrieval or the pre-judge failed

    def will_answer(self, question_id: str) -> bool:
        """Phase 3 generates an answer: retrieval worked and the pre-judge let the question through."""
        result = self.results.get(question_id)
        return result is not None and result.can_answer


def retrieve_all(questions: list[Question], doc_ids: dict[str, str], ingested: IngestedPapers,
                 mode: Mode) -> Retrievals:
    """Retrieval + pre-judge for every question. The answering model does the pre-judge; in private mode it is
    loaded here, straight after the context model is unloaded, and stays loaded for phase 3."""
    say("\n--- Phase 2/4: retrieval + pre-judge "
        f"({'with ' + mode.prejudge_model if mode.prejudge else 'pre-judge off'}) ---")
    if mode.local:
        _load_model(settings.llm_model, settings.llm_num_ctx)  # instant if it is still loaded
    retrievals = Retrievals()
    for n, (pid, qa) in enumerate(questions, start=1):
        if pid in ingested.errors:
            continue
        say(f"  [Q {n}/{len(questions)}] {short(qa['question'])}")
        try:
            retrievals.results[qa["question_id"]] = _retrieve_and_prejudge(doc_ids[pid], qa, mode, indent="      ")
        except Exception as e:  # noqa: BLE001
            log.exception("Question %s failed", qa["question_id"])
            say(f"      FAILED: {e}")
            retrievals.errors[qa["question_id"]] = f"rag: {e}"
    return retrievals


def _retrieve_and_prejudge(doc_id: str, qa: dict, mode: Mode, indent: str) -> Retrieved:
    """retrieval.search() stage by stage, timing each, then the pre-judge."""
    question = qa["question"].strip()
    t0 = time.perf_counter()
    rankings = retrieval.candidate_rankings(doc_id, question, mode)
    candidates = retrieval.fuse(doc_id, rankings, mode.candidate_pool if mode.reranker else mode.top_k)
    retrieval_s = time.perf_counter() - t0
    vector_scores = [score for _, score in rankings["vector"]]
    lowest, highest = min(vector_scores, default=None), max(vector_scores, default=None)
    score_range = f" {lowest:.3f}–{highest:.3f}" if vector_scores else ""
    cutoff = "" if mode.vector_min_score is None else f" ≥ {mode.vector_min_score}"
    say(f"{indent}retrieval: {len(candidates)} chunks in {retrieval_s:.2f}s "
        f"(BM25 {len(rankings['bm25'])} · vector {len(vector_scores)}{cutoff}{score_range})")

    passages, rerank_s = candidates, None
    if mode.reranker:
        t0 = time.perf_counter()
        passages = reranker.rerank(question, candidates, mode.top_k)
        rerank_s = time.perf_counter() - t0
        say(f"{indent}rerank: kept fused #{', #'.join(str(p['fused_rank']) for p in passages)} in {rerank_s:.2f}s")

    verdict, prejudge_s = prejudge.ALL, None
    if mode.prejudge:
        st = Status(f"{indent}pre-judge…")
        t0 = time.perf_counter()
        verdict = prejudge.verdict(question, passages, mode)
        prejudge_s = time.perf_counter() - t0
        st.done(f"{indent}pre-judge: {VERDICT_TEXT[verdict]} ({duration(prejudge_s)})")
    return Retrieved(question=question, passages=passages, candidates=candidates, verdict=verdict,
                     retrieval_s=retrieval_s, rerank_s=rerank_s, prejudge_s=prejudge_s,
                     bm25_hits=len(rankings["bm25"]), vector_hits=len(vector_scores),
                     vector_score_min=lowest, vector_score_max=highest)


# ------------------------------------------------------------------ phase 3: answer + score
def answer_all(questions: list[Question], retrievals: Retrievals, ingested: IngestedPapers,
               mode: Mode) -> list[dict]:
    """One result row per question: the answer and its scores, or the error that stopped it."""
    to_answer = sum(1 for r in retrievals.results.values() if r.can_answer)
    say(f"\n--- Phase 3/4: answering {to_answer} of {len(questions)} questions with {mode.llm_model} ---")
    rows: list[dict] = []
    answer_times: list[float] = []  # for the time-left estimate
    for n, (pid, qa) in enumerate(questions, start=1):
        qid = qa["question_id"]
        base = {"paper_id": pid, "question_id": qid, "question": qa["question"]}
        if pid in ingested.errors:
            rows.append({**base, "error": f"ingest: {ingested.errors[pid]}"})
            continue
        if qid in retrievals.errors:
            rows.append({**base, "error": retrievals.errors[qid]})
            continue
        step = retrievals.results[qid]
        eta = ""
        if answer_times and step.can_answer:
            left = sum(1 for _, q in questions[n - 1:] if retrievals.will_answer(q["question_id"]))
            eta = f" · ~{duration(sum(answer_times) / len(answer_times) * left)} left in phase 3"
        say(f"  [Q {n}/{len(questions)}{eta}] {short(qa['question'])}")
        t0 = time.perf_counter()
        try:
            row = _answer_and_score(step, qa, ingested.paragraphs[pid], mode, indent="      ")
        except Exception as e:  # noqa: BLE001
            log.exception("Question %s failed", qid)
            say(f"      FAILED: {e}")
            row = {"question_id": qid, "question": qa["question"], "error": f"rag: {e}"}
        if step.can_answer:
            answer_times.append(time.perf_counter() - t0)
        rows.append({"paper_id": pid, **row})
    return rows


def _generate_answer(step: Retrieved, mode: Mode, indent: str) -> tuple[str, float]:
    """The answering model's reply, streamed as in the chat, and how long it took."""
    st = Status(f"{indent}generating…")
    st.set("waiting for first token")
    pieces: list[str] = []
    t0 = time.perf_counter()
    messages = build_messages(step.question, step.passages, [], partial=step.verdict == prejudge.PARTIAL)
    for piece in mode.stream_answer(messages):
        pieces.append(piece)
        st.set(f"{len(pieces)} tokens")
    generation_s = time.perf_counter() - t0
    response = "".join(pieces)
    st.done(f"{indent}generation: {len(pieces)} tokens in {duration(generation_s)}")
    say(f"{indent}answer: {short(response, 90)}")
    return response, generation_s


def _answer_and_score(step: Retrieved, qa: dict, paragraphs: list[str], mode: Mode, indent: str) -> dict:
    refs = references(qa)
    generation_s = None
    if step.can_answer:
        response, generation_s = _generate_answer(step, mode, indent)
    else:
        response = prejudge.NOT_ENOUGH_CONTENT
        say(f"{indent}skipped: pre-judge said not enough content")

    passages = step.passages
    cited = scoring.cited_passages(response, len(passages))
    predicted_answer, predicted_evidence = scoring.prediction(response, step.can_answer, cited, passages, paragraphs)
    answer_f1, answer_type = scoring.answer_f1(predicted_answer, refs)
    evidence_f1 = scoring.evidence_f1(predicted_evidence, refs)
    recall = scoring.evidence_recall([p["text"] for p in passages], paragraphs, refs)
    # the same over all the fused chunks the reranker chose from: the most reranking can put into the top k
    candidate_recall = (scoring.evidence_recall([c["text"] for c in step.candidates], paragraphs, refs)
                        if mode.reranker else None)

    recall_text = "n/a" if recall is None else f"{100 * recall:.0f}%"
    if candidate_recall is not None:
        recall_text += f" (in the {len(step.candidates)} candidates: {100 * candidate_recall:.0f}%)"
    say(f"{indent}scores: retrieval recall {recall_text} · answer F1 {100 * answer_f1:.0f} "
        f"· evidence F1 {100 * evidence_f1:.0f}")

    return {
        "question_id": qa["question_id"],
        "question": step.question,
        "references": refs,
        "response": response,
        "predicted_answer": predicted_answer,
        "bm25_hits": step.bm25_hits,
        "vector_hits": step.vector_hits,
        "vector_score_min": step.vector_score_min,
        "vector_score_max": step.vector_score_max,
        "retrieved_chunks": [p["index"] for p in passages],
        "fused_ranks": [p.get("fused_rank") for p in passages],  # where the kept chunks were before reranking
        "cited_passages": cited,
        "answer_f1": answer_f1,
        "answer_type": answer_type,
        "evidence_f1": evidence_f1,
        "retrieval_recall": recall,  # None: no text evidence (e.g. unanswerable)
        "candidate_recall": candidate_recall,
        "prejudge_can_answer": step.can_answer if mode.prejudge else None,
        "prejudge_verdict": step.verdict if mode.prejudge else None,  # all / partial / none
        "gold_unanswerable": any(r["type"] == "none" for r in refs),  # some annotator says the paper can't answer
        "retrieval_s": step.retrieval_s,
        "rerank_s": step.rerank_s,
        "prejudge_s": step.prejudge_s,
        "generation_s": generation_s,  # None when the pre-judge skipped generation
    }


# ------------------------------------------------------------------ phase 4: judge
def judge_all(rows: list[dict], mode: Mode) -> None:
    """Adds the judge's verdict (judge_correct, judge_output, judging_s) to every answered row."""
    say(f"\n--- Phase 4/4: judging answers with {settings.judge_model} ---")
    if not mode.local or settings.judge_model != settings.llm_model:  # else it is loaded since phase 2
        _load_model(settings.judge_model, settings.llm_num_ctx)
    for n, row in enumerate(rows, start=1):
        if "error" in row:
            say(f"  [judge {n}/{len(rows)}] skipped (question failed)")
            continue
        st = Status(f"  [judge {n}/{len(rows)}] grading…")
        try:
            t0 = time.perf_counter()
            prompt = scoring.judge_prompt(row["question"], row["references"], row["response"])
            out = ollama.complete(settings.judge_model, prompt, settings.llm_num_ctx,
                                  read_timeout=settings.llm_timeout)
            row["judging_s"] = time.perf_counter() - t0
            row["judge_correct"], row["judge_output"] = scoring.parse_judge_verdict(out), out
            verdict = {True: "CORRECT", False: "INCORRECT", None: "no verdict"}[row["judge_correct"]]
            st.done(f"  [judge {n}/{len(rows)}] {verdict:<9} ({duration(row['judging_s'])}) "
                    f"{short(row['question'], 60)}")
        except Exception as e:  # noqa: BLE001
            log.exception("Judging failed for %s", row["question_id"])
            row["judge_error"] = str(e)
            st.done(f"  [judge {n}/{len(rows)}] FAILED: {e}")
