"""QASPER evaluation of the RAG pipeline (allenai/qasper: NLP papers + questions + gold answers + evidence).

Runs only when triggered:
    python -m app.evaluate --run-id <id> [--papers 5] [--seed 0] [--split test]
                           [--mode private|cloud-rerank|cloud-prejudge|cloud-rerank-prejudge]
                           [--sweep 30,25,20,15 | --compare rerank,prejudge,rerank-prejudge]

Runs in four phases, one Ollama model at a time: (1) each sampled paper is converted to Markdown and
ingested like an upload (chunk + optional Contextual Retrieval contexts + embed + store), (2) every
question goes through hybrid retrieval + the pre-judge, (3) the questions the pre-judge let through are
answered, (4) the judge grades the answers. Each stage is timed; model switches are not.
The cloud modes run steps 1-3 with the cloud models (see modes.py); the judge is always JUDGE_MODEL in Ollama.
--sweep N,M,... ingests the papers once, then runs steps 2-4 once per N with BM25 top N + vector top N, one result
row each (<run-id>-bNvN). --compare ingests the papers once in cloud mode, then runs steps 2-4 once per cloud
variant (reranker, pre-judge, or both), one result row each (<run-id>-rerank, <run-id>-prejudge, ...).

Metrics:
  - Answer F1     official QASPER token F1 against the best-matching annotator answer
  - Evidence F1   official QASPER paragraph F1; predicted evidence = paper paragraphs inside the cited passages
  - Retrieval recall@k  share of gold evidence paragraphs found anywhere in the top-k retrieved chunks
  - Judge correct LLM judge: does the answer convey the same information as a reference answer?

Each run writes to <EVAL_DIR> (evaluation_metrics/):
  <run_id>.json  complete details: metrics, timings, environment variables, and every question's
                 answer, references, scores and judge output
  result_40.md   one row per run, for side-by-side comparison

Modules:
  __main__  command-line arguments
  run       one run: samples the papers, ingests them once, runs phases 2-4 per variant, cleans up
  phases    the four phases
  scoring   how one answer is scored: QASPER answer and evidence F1, retrieval recall, the judge's prompt and verdict
  report    the run summary, its row in result_40.md, <run_id>.json and the summary printed at the end
  console   terminal output (status lines that update in place)
  qasper    the dataset and the official QASPER scoring functions
"""
