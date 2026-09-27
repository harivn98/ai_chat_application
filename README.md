# AI_chat_application — local RAG over your documents

Upload a PDF, TXT or Markdown file, wait for it to be indexed, then chat with it.
Everything (UI, API, vector DB, LLM) runs in Docker on your machine.

```
            ┌──────────────────────── docker compose ─────────────────────────┐
 Browser ──►│ frontend  (Next.js :3000) ── /api/* proxy ──► backend (FastAPI)  │
            │                                               │   │    │         │
            │                     MongoDB Atlas Local ◄─────┘   │    └──► Ollama│
            │                     (docs, chunks, vectors,       │      qwen3:8b │
            │                      $vectorSearch index)   bge-small-en-v1.5    │
            └────────────────────────────────────────────────────────────────┘
```

## Pipeline

**Ingestion** (`POST /documents`, runs in the background; UI polls status)

1. **Convert to Markdown** — PDF via `pymupdf4llm` (keeps headings, lists, tables), TXT wrapped as Markdown (encoding-safe), MD cleaned (front matter removed). The Markdown is stored in MongoDB and in `/data/markdown/<id>.md`.
2. **Chunk** — heading-aware: split by `#` structure first, then packed into ~1000-char chunks on paragraph/sentence boundaries with 150-char overlap. Code fences and tables stay intact. Each chunk carries its heading path (`Manual > Braking > Pads`).
3. **Add context** (Contextual Retrieval, `CONTEXTUAL_EMBEDDING=true`) — `qwen3:4b-instruct` reads the document and writes one sentence per chunk saying where it fits (e.g. *"This chunk describes the similarity measure used in the sentence-based approach…"*). The context is prepended to the text that gets embedded and BM25-indexed. The answering LLM and the UI still see the original chunk. The document goes first in every prompt, so Ollama reuses its cache and each chunk takes about 1 s on a GPU. Documents longer than `CONTEXT_NUM_CTX` are read in windows that each start with the document's opening.
4. **Embed** — `BAAI/bge-small-en-v1.5` (384-d, normalised). The model is baked into the backend image, so it runs offline.
5. **Store** — chunks, contexts and vectors go into MongoDB `ragdb.chunks`, with an Atlas Vector Search index (`cosine`, filtered by `doc_id`).
6. **Wait for the index** — mongot syncs asynchronously, so the document only flips to `ready` once `$vectorSearch` can see every chunk. **The chat window appears only after this point.**

**Query** (`POST /chat`, streamed as NDJSON)

1. **BM25** (rank-bm25, BM25+ variant, Snowball-stemmed tokens) over the document's chunks → top 10.
2. **Dense** — query embedded with the bge query instruction, `$vectorSearch` → top 20, keeping only chunks with cosine similarity ≥ 0.85.
3. **Reciprocal Rank Fusion** (k=60) → top 5 passages.
4. **Pre-judge** (`PREJUDGE_ENABLED=true`) — Qwen3 8B, the answering model, reads the 5 passages and the question and answers YES or NO: do the passages contain the specific information the question asks for? It is strict: passages that are only on the same topic get NO, so questions the document doesn't actually answer are caught. Answers that follow clearly from stated facts (e.g. settling a yes/no question) still get YES. On **NO**, answer generation is skipped and the reply is *"There isn't enough content in the document to answer this question."*, with the passages still shown.
5. **Qwen3 8B** (Ollama, on the GPU, `think: false`, `num_ctx` 8192) answers only from the passages, citing them as `[1]`, `[2]`. The UI shows each passage with its BM25/vector rank; clicking a citation jumps to it.

**GPU use:** all models run on the GPU, one at a time. `qwen3:4b-instruct` is used only to write chunk contexts during an upload. As soon as contextualizing finishes, it is unloaded and Qwen3 8B starts loading in the background, while the upload is still embedding and indexing, so it is usually ready before the first question. Qwen3 8B then does both the pre-judge and the answer, so chatting never swaps models. Loading takes about 45 s for Qwen3 8B and 25 s for `qwen3:4b-instruct` on an RTX 4070 laptop GPU; unloading is instant.

## Run it

Requirements: Docker Desktop / Docker Engine with Compose v2, ~12 GB free disk (Qwen3 8B ≈ 5.2 GB), 16 GB RAM recommended.

Start it with the script. It uses your NVIDIA GPU when Docker can reach one, and falls back to CPU otherwise:

```powershell
.\start.ps1 --build        # Windows (PowerShell)
```
```bash
./start.sh --build         # Linux / macOS
```

First start pulls images, builds both apps and downloads `qwen3:8b` (the `ollama-init` job). Then open **http://localhost:3000**.

- Use `--build` after code changes. Without it, the script just starts the app or applies `.env` changes.
- The script saves its GPU/CPU choice as `COMPOSE_FILE` in `.env`, so plain `docker compose exec / logs / stop` commands use the same setup afterwards. Rerun the script if the GPU situation changes.
- If PowerShell blocks the script, run `powershell -ExecutionPolicy Bypass -File .\start.ps1 --build`.
- Check where the model runs with `docker compose exec ollama ollama ps`. It should say `100% GPU`.
- Stop the app with `docker compose stop` (or `docker compose down` to remove the containers; data is kept).

**GPU requirements:** on Windows, Docker Desktop with the WSL 2 backend and a current NVIDIA driver; on Linux, the NVIDIA Container Toolkit. An 8 GB card fits Qwen3 8B (Q4) with the 8k context. Even when the GPU is attached, Ollama falls back to CPU by itself if the model doesn't fit.

### Useful URLs

| What | URL |
|---|---|
| App | http://localhost:3000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health | http://localhost:8000/health |
| MongoDB (Compass) | `mongodb://admin:admin@localhost:27017/?directConnection=true` |

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `qwen3:8b` | Any Ollama chat model |
| `LLM_THINK` | `false` | `true` enables Qwen3 reasoning (slower; reasoning is not shown) |
| `LLM_NUM_CTX` | `8192` | Ollama context window |
| `LLM_TEMPERATURE` | `0.2` | Answer randomness. `0` makes answers repeatable, which keeps evaluation runs comparable |
| `LLM_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded after the last request |
| `LLM_TIMEOUT` / `LLM_LOAD_TIMEOUT` | `600` / `1800` | Seconds to wait for Ollama output / for the model to load (it is preloaded when the backend starts) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Characters |
| `TOP_K` | `5` | Passages sent to the LLM after fusion |
| `BM25_CANDIDATES` | `10` | Chunks taken from BM25 before fusion |
| `VECTOR_MIN_SCORE` | `0.85` | Minimum cosine similarity for embedding hits |
| `VECTOR_CANDIDATES` | `20` | Max chunks from vector search before the score cutoff |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant (higher = flatter blend of the two rankings) |
| `HISTORY_TURNS` | `6` | Previous chat messages sent with each question |
| `CONTEXTUAL_EMBEDDING` | `true` | `true` / `false`: add an LLM-written context to every chunk before embedding + BM25. Affects newly uploaded documents |
| `CONTEXT_MODEL` | `qwen3:4b-instruct` | Ollama model that writes the contexts. Use a non-thinking model: plain `qwen3:4b` is thinking-only and writes its reasoning instead |
| `CONTEXT_NUM_CTX` | `16384` | Tokens of the document the context model reads at once. Longer documents are split into windows |
| `PREJUDGE_ENABLED` | `true` | `true` / `false`: check whether the retrieved passages can answer before calling the answering LLM |
| `MAX_UPLOAD_MB` | `25` | Upload limit |
| `JUDGE_MODEL` | `qwen3:8b` | Ollama model that grades answers in the [evaluation](#evaluation-qasper) |

Changing chunking settings only affects newly uploaded documents.

## API

| Method | Path | |
|---|---|---|
| `POST` | `/documents` | multipart `file` → `{doc_id, status}` (202) |
| `GET` | `/documents/{id}` | status: `queued → converting → chunking → embedding → storing → indexing → ready` (or `failed` + `error`) |
| `GET` | `/documents/{id}/markdown` | the converted Markdown |
| `DELETE` | `/documents/{id}` | removes doc, chunks and files |
| `POST` | `/chat` | `{doc_id, question, history}` → NDJSON: `sources`, `token`…, `done` / `error` |

## Project layout

```
backend/app/
  converter.py   PDF/TXT/MD → Markdown
  chunker.py     heading-aware chunking
  embeddings.py  bge-small-en-v1.5
  db.py          MongoDB + vector index creation
  ingest.py      ingestion pipeline + status updates
  retrieval.py   BM25 + $vectorSearch + RRF
  llm.py         prompt + Ollama streaming
  main.py        FastAPI routes
  contextual.py  Contextual Retrieval (per-chunk context from a small LLM)
  prejudge.py    YES/NO check: can the retrieved passages answer the question?
  evaluate.py    QASPER evaluation (CLI)
frontend/
  app/page.tsx                 upload → chat flow
  app/api/[...path]/route.ts   proxy to the backend (single exposed origin)
  components/Uploader.tsx      upload + ingestion progress
  components/ChatWindow.tsx    streaming chat, citations, sources
```

## Evaluation (QASPER)

The RAG pipeline can be scored on [QASPER](https://huggingface.co/datasets/allenai/qasper): NLP research papers with questions, gold answers and the evidence paragraphs for each answer. It only runs when you trigger it:

```bash
.\start.ps1 --build              # or ./start.sh --build; the code is baked into the image
docker compose exec backend python -m app.evaluate --run-id baseline-1 --papers 5 --seed 0
```

| Option | Meaning |
|---|---|
| `--run-id` | Unique name for the run (letters, digits, `.`, `_`, `-`). Reusing one is refused. |
| `--papers` | How many papers to sample (default 5). Every question on a sampled paper is asked, about 3.5 per paper. |
| `--seed` | Which papers get sampled (default 0). Same seed = same papers, so runs are comparable. |
| `--split` | `test` (default, 416 papers / 1,451 questions) or `validation`. |

On CPU, expect roughly 30–90 s per question: `--papers 1` checks that it works, `--papers 5` is a quick comparison, and `--papers 20` or more gives more stable numbers. The dataset (~4 MB) is downloaded on the first run.

**What happens.** It runs in four phases, loading each model once per run:

1. **Ingest** (`qwen3:4b-instruct`): each sampled paper is converted to Markdown and ingested like an upload (chunk, add context if `CONTEXTUAL_EMBEDDING=true`, embed, store).
2. **Retrieve + pre-judge** (`qwen3:8b`, loaded right after the context model is unloaded): every question goes through hybrid retrieval and the pre-judge.
3. **Answer** (`qwen3:8b`, still loaded): questions the pre-judge let through are answered as in the chat. Rejected ones get the "not enough content" reply and count as answered "Unanswerable".
4. **Judge** (`JUDGE_MODEL`): the judge model grades each answer.

The timings exclude model loading. The chat behaves the same way: the pre-judge and the answer use the same loaded model, so a question never waits for a model swap.

The papers' chunks are deleted afterwards, so nothing shows up in the app.

**Timings** (averages): contextualization (the context model writing contexts for one paper; `–` when off), embedding (embed one paper), retrieval (query embedding + BM25 + vector search + RRF), pre-judge (the YES/NO check), generation (full answer, for questions the pre-judge let through), and judging (one judge call). **Pre-judge rejected** shows how many questions got "not enough content", and how many of those an annotator also marked unanswerable (those rejections were right). Time spent waiting for the vector index to sync isn't counted.

**Scores** (0–100):

- **Retrieval recall@k**: share of gold evidence paragraphs that are in the top-k retrieved chunks. This measures retrieval on its own.
- **Evidence F1**: official QASPER paragraph F1, using the paper paragraphs inside the passages the answer cites.
- **Answer F1**: official QASPER token F1 against the closest annotator answer, also broken down by answer type (extractive / abstractive / yes-no / unanswerable). It penalizes long answers, even correct ones, so compare it between your own runs rather than with published results.
- **Judge correct**: `JUDGE_MODEL` decides whether the answer matches a reference answer.

Low recall means the answer never reached the LLM: tune `TOP_K`, `BM25_CANDIDATES`, `VECTOR_MIN_SCORE` or the chunk size. High recall with a low judge score points to the prompt or the model instead.

**Output**, in `evaluation_metrics/`:

- `<run-id>.json`: the complete details. It holds the metrics, timings, all the environment variables the run used (MongoDB password masked), and every question's answer, references, retrieved and cited chunks, scores and judge output.
- `results.md`: one row per run, for comparing runs side by side.

**Comparing settings.** Change one value in `.env`, apply it with `.\start.ps1` (or `./start.sh`), then rerun with a new run ID and the same seed:

```bash
docker compose exec backend python -m app.evaluate --run-id minscore-075 --papers 5 --seed 0
```

**A/B test contextual embedding.** Run once with `CONTEXTUAL_EMBEDDING=false` and once with `true`, using the same seed. The Retrieval config column shows `no ctx` or `ctx <model>` for each run.

**Caveats**

- The papers are NLP research papers, not your documents, so the scores describe the pipeline in general.
- `JUDGE_MODEL` defaults to the answering model. A model judging its own answers is lenient, so use a different one when possible (`docker compose exec ollama ollama pull <model>`).
- About 4% of gold evidence is a section heading rather than a paragraph, which caps recall slightly for every run.

## Troubleshooting

- **Scanned PDFs** fail with "no extractable text" — OCR isn't included. Adding `tesseract-ocr` to the backend image enables pymupdf4llm's OCR path.
- **Slow answers on CPU** — expected for an 8B model; use the GPU override or a smaller model (`LLM_MODEL=qwen3:4b`).
- **Vector index not ready** — `docker compose logs backend` shows the index status. If mongot never becomes available, the backend falls back to in-process cosine search, so chat still works.
- **Reset everything** — `docker compose down -v` (deletes documents, vectors and the downloaded model).
