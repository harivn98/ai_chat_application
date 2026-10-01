# AI Chat Application (ACAP): chat with your documents

Upload a PDF, TXT or Markdown file, wait for it to be indexed, then ask questions about it. Every answer cites the
passages it used, and clicking a citation opens the document with that passage highlighted.

The app has two modes, picked per upload:

- **Private mode** (default): everything runs on your machine in Docker, including the LLM. Nothing leaves it.
- **Cloud mode**: embeddings and answers come from hosted models through OpenRouter. It is faster and more accurate, but
  the document text and your questions leave the machine.

This README covers [how to run it](#1-how-to-run-it), [what I built](#2-what-i-built),
[the decisions I took and why](#3-decisions-and-why), and [what I would do next](#4-what-i-would-do-next). Reference
material (configuration, API, project layout, evaluation, troubleshooting) follows after that.

## 1. How to run it

**Requirements**

- Docker Desktop or Docker Engine with Compose v2.
- About 15 GB of free disk and 16 GB of RAM. The first start downloads two Ollama models (`qwen3:8b`, about 5.2 GB, and
  `qwen3:4b-instruct`) and builds a backend image that contains the embedding and reranker models.
- Optional: an NVIDIA GPU with 8 GB of memory for private mode. Without one, private mode works but is slow (roughly
  30–90 s per answer on CPU); use cloud mode instead.
- Optional: an [OpenRouter API key](https://openrouter.ai/settings/keys) for cloud mode.

**Start**

```powershell
.\start.ps1 --build        # Windows (PowerShell)
```
```bash
./start.sh --build         # Linux / macOS
```

Then open **http://localhost:3000**.

The script starts the stack on your NVIDIA GPU when Docker can reach one and on CPU otherwise. It saves that choice as
`COMPOSE_FILE` in `.env`, so plain `docker compose logs / exec / stop` commands use the same setup afterwards. Without
the script, `docker compose up -d --build` starts the CPU setup.

- Use `--build` on the first start and after code changes. Without it the script starts the app or applies `.env`
  changes.
- If PowerShell blocks the script: `powershell -ExecutionPolicy Bypass -File .\start.ps1 --build`.
- Check where the model runs: `docker compose exec ollama ollama ps` (it should say `100% GPU`).
- Stop: `docker compose stop`. Remove everything including data and models: `docker compose down -v`.
- GPU setup: on Windows, Docker Desktop with the WSL 2 backend and a current NVIDIA driver; on Linux, the NVIDIA
  Container Toolkit.

**Environment variables**

All settings live in the committed [.env](.env) and have working defaults, so private mode needs nothing from you. The
full list is in [Configuration](#configuration-env).

Only cloud mode needs a secret: `OPENROUTER_API_KEY`, with credits on the account. Don't put it in `.env`, because
that file is committed. Set it in your environment and start the app from a new terminal:

```powershell
setx OPENROUTER_API_KEY "sk-or-..."      # Windows; then open a new terminal
.\start.ps1
```
```bash
export OPENROUTER_API_KEY=sk-or-...      # Linux / macOS
./start.sh
```

`setx` only reaches programs started afterwards. In a terminal that was already open (including every VS Code terminal
until VS Code is restarted), load the key first:

```powershell
$env:OPENROUTER_API_KEY = [Environment]::GetEnvironmentVariable('OPENROUTER_API_KEY', 'User')
.\start.ps1
```

Check that the backend has it (prints `True` or `False`, never the key):
`docker compose exec backend python -c "import os; print(bool(os.getenv('OPENROUTER_API_KEY')))"`.
Without the key, the cloud option is shown but disabled.

| What | URL |
|---|---|
| App | http://localhost:3000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health | http://localhost:8000/health |
| MongoDB (Compass) | `mongodb://admin:admin@localhost:27017/?directConnection=true` |

## 2. What I built

```
            ┌──────────────────────── docker compose ─────────────────────────┐
 Browser ──►│ frontend (Next.js :3000) ── /api/* proxy ──► backend (FastAPI)   │
            │                                               │   │    │         │
            │                     MongoDB Atlas Local ◄─────┘   │    └──► Ollama│
            │                     (documents, chunks, chats,    │    (local LLMs)
            │                      $vectorSearch index)         └──► OpenRouter (cloud mode only)
            └──────────────────────────────────────────────────────────────────┘
```

Four containers: a Next.js frontend, a FastAPI backend, MongoDB Atlas Local (document store and vector index in one)
and Ollama. The frontend and backend are written from scratch: no RAG framework, no chat UI kit, no component library.

### Ingestion (`POST /documents`, runs in the background, the UI shows progress)

1. **Convert to Markdown.** PDFs go through `pymupdf4llm`, which keeps headings, lists and tables. TXT is wrapped as
   Markdown and MD is cleaned. The Markdown is stored, because it is also what the document viewer shows.
2. **Chunk.** Split along the heading structure first, then pack into ~1000-character chunks on paragraph and sentence
   boundaries with 150 characters of overlap. Code fences and tables stay whole. Each chunk keeps its heading path
   (`Manual > Braking > Pads`) and its position in the Markdown.
3. **Add context** (Contextual Retrieval). A small LLM reads the document and writes one sentence per chunk saying
   where it fits. That sentence is prepended to the text that gets embedded and BM25-indexed. The answering model and
   the UI still see the original chunk.
4. **Embed** and **store** the chunks, contexts and vectors in MongoDB.
5. **Wait for the vector index.** It syncs asynchronously, so the document only becomes `ready` once
   `$vectorSearch` can see every chunk. The chat opens right after the upload; questions typed before that are queued
   and sent when the document is ready.

### Answering (`POST /chat`, streamed as NDJSON)

1. **Rewrite follow-ups.** When the chat has history, the answering model rewrites the question so it stands on its own
   (*"What about its limitations?"* becomes *"What are the limitations of the proposed method?"*). Retrieval uses the
   rewritten question; the answer is still written for the question as asked.
2. **Hybrid retrieval.** BM25 (stemmed, top 30) and vector search (top 30) each rank the document's chunks.
3. **Fuse** the two rankings with Reciprocal Rank Fusion.
4. **Select 8 passages.** Private mode: a cross-encoder reranker reads the question with every fused candidate and
   keeps the best 8. Cloud mode: the top 8 by fused rank.
5. **Pre-judge** (cloud mode; optional in private mode). A cheap model checks whether the 8 passages hold ALL, PART or
   NONE of the answer. On NONE the answering model is not called and the reply is *"There isn't enough content in the
   document to answer this question."* On PARTIAL the answer says which part the document doesn't cover.
6. **Answer.** The model answers only from the passages and cites them as `[1]`, `[2]`.

### The two modes

| | Private mode (default) | Cloud mode |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (384-d, CPU, baked into the image) | Gemini Embedding 2, cut to 768-d |
| Chunk contexts | `qwen3:4b-instruct` (Ollama) | Gemini 3.5 Flash-Lite |
| Passage selection | cross-encoder reranker (`ms-marco-MiniLM-L6-v2`, CPU) | top 8 by fused rank |
| Pre-judge | off by default (`PREJUDGE_ENABLED`) | always, Gemini 3.5 Flash-Lite |
| Answers and follow-up rewriting | `qwen3:8b` (Ollama) | DeepSeek V4.1 Flash |
| Data leaving the machine | none | document text, questions and retrieved passages, to OpenRouter and on to Google and DeepSeek |

Both modes share the same chunking, BM25, fusion, MongoDB and prompts. The mode belongs to the document: a document is
always answered in the mode it was uploaded with.

### The interface

- Upload page with the mode choice, a list of private mode's limitations, and a warning before anything is sent to the
  cloud.
- Streaming answers with Markdown, tables, code highlighting and KaTeX math, a Stop button and a copy button.
- **Citations.** Each `[n]` is a chip. Clicking it opens the document in a side panel, scrolled to the cited passage,
  which is highlighted. Under each answer the passages are listed with their BM25, vector and pre-rerank ranks, so you
  can see why a passage was picked.
- **Chats.** A document can have up to 10 chats. Chats are saved in MongoDB and survive a refresh. **Chat history**
  lists every document with its chats; **New session** deletes everything.
- Progress for the slow parts: indexing stages during an upload, and a bar while the local answering model loads.

One file is uploaded at a time. You can keep many documents, but each chat is about one document.

### Evaluation

`python -m app.evaluate` scores the pipeline on [QASPER](https://huggingface.co/datasets/allenai/qasper) (NLP papers
with questions, gold answers and gold evidence paragraphs). It reports retrieval recall, the official QASPER answer and
evidence F1, an LLM judge's verdict and per-stage timings. Every run in [evaluation_metrics/](evaluation_metrics/) is
committed with its full settings, and the decisions below cite those runs. Details are in
[Evaluation](#evaluation-qasper).

## 3. Decisions and why

Where a decision rests on a measurement, the run ID in brackets is a row in
[evaluation_metrics/results.md](evaluation_metrics/results.md) (5 papers, 20 questions) or
[evaluation_metrics/result_40.md](evaluation_metrics/result_40.md) (40 papers, 135 questions). Read the 20-question
runs as direction only: one question is 5 points.

### Measure first

I built the evaluation right after the first working pipeline, before tuning anything, and changed one setting at a time against a fixed seed.
Most decisions below come from it, and it overturned several of my assumptions. The numbers that matter most:

| Step (private mode) | Recall@k | Judge correct | Run |
|---|---|---|---|
| Basic pipeline: hybrid search, top 5 | 65.8 | 60 | `poc_basic_pipeline` (20 q) |
| + chunk contexts | 70.2 | 70–75 | `context_encoding`, `ce_gpu_opt` (20 q) |
| + cross-encoder reranker | 83.3 | 80 | `ce_reranker` (20 q) |
| Same settings on 135 questions | 75.3 | 70.4 | `best_40p_v2` |
| No vector cutoff, 30 + 30 candidates, top 8 | 85.1 | 74.1 | `private40-b30v30` |

| Cloud mode (135 questions) | Recall@8 | Evidence F1 | Judge correct | Run |
|---|---|---|---|---|
| Reranker only | 86.1 | 39.1 | 88.1 | `cloud40-rerank` |
| Pre-judge only (shipped) | 89.5 | 48.0 | 89.6 | `cloud40-prejudge` |
| Reranker + pre-judge | 86.1 | 47.3 | 88.9 | `cloud40-rerank-prejudge` |

### Two modes instead of one

A document chat is often used for documents that shouldn't leave the building, so I wanted a version where nothing does.
An 8B local model is clearly weaker than a hosted one, though (74.1 against 89.6 judge-correct), and needs a GPU. So
the user chooses per document, and the UI states what each choice costs. I rejected a cloud-only app (no privacy story)
and a local-only app (unusable without a GPU).

### Models

- **`qwen3:8b` for local answers.** It is the largest model that fits an 8 GB GPU with an 8k context. Thinking is
  switched off: it adds latency and the answers come from the passages anyway.
- **`bge-small-en-v1.5` for local embeddings.** It is small enough to run on the CPU, which keeps the GPU for the LLM,
  and it is baked into the image so the backend works offline. The cost is that it is English-only.
- **DeepSeek V4.1 Flash and Gemini via OpenRouter for the cloud.** One API key and one bill for three models, and
  swapping a model is a one-line `.env` change. The rejected option was separate Google and DeepSeek SDKs.
- **Gemini embeddings cut from 3072 to 768 dimensions.** The model is trained so that the leading values form an
  embedding on their own; the backend keeps the first 768 and renormalises. That is a quarter of the storage and index
  size.

### Retrieval

- **Hybrid search rather than vectors only.** Technical documents are full of exact terms (part numbers, names,
  abbreviations) that embeddings blur and BM25 matches exactly. Reciprocal Rank Fusion combines the two by rank, so no
  score normalisation or weight tuning is needed.
- **Chunk contexts (Contextual Retrieval).** A chunk such as "it increased by 3%" is unfindable on its own. Contexts
  raised recall from 65.8 to 70.2 and judge-correct from 60 to 70–75. The cost is upload time: about 39 s per paper on
  the GPU. I tried `qwen3:1.7b` to cut that; it was 39% faster but recall fell from 83.3 to 67.5 (`ctx_qwen3_1_7b`), so
  I kept the 4B model.
- **A reranker in private mode.** It was the largest single gain: recall 70.2 → 83.3 (`ce_reranker`).
- **The small reranker, not the bigger one.** I expected `bge-reranker-base` to beat `ms-marco-MiniLM-L6-v2`. At the
  same settings it scored 67.5 recall against 78.1 and took 8.7 s per question against 1.2 s on CPU
  (`ce_rr_0_30_20_bge_frontend` against `ce_rr_0_30_20`). Both are in the image; `RERANKER_MODEL` switches.
- **No similarity cutoff.** I started with a minimum cosine score of 0.85 for vector hits. The 135-question run showed
  it left on average 0.0 vector hits per question: vector search was effectively switched off and nothing in the app
  showed it. Taking the top 30 by rank and letting the reranker judge relevance raised recall from 75.3 to 85.1. This
  is why the result table now reports how many chunks each search contributed.
- **30 + 30 candidates, 8 passages.** A sweep over 15, 20, 25 and 30 candidates gave the same recall (85.1–85.5) and
  judge-correct within noise (71.9–74.1), so the pool size hardly matters; I kept 30. Sending 15 passages instead of 8
  raised recall but not judge-correct (`cloud-prejudge_moredata`), so more context was not worth the longer prompt.
- **No reranker in cloud mode.** With Gemini embeddings the reranker made retrieval worse (recall 86.1 against 89.5)
  and added about 1 s per question. Fused rank alone picks the 8 passages there.

### Pre-judge

The idea: before answering, ask whether the passages contain the answer, so the app says "not in the document" instead
of guessing.

- **On in cloud mode.** Gemini Flash-Lite takes about 0.6 s. It rejected 16 of 135 questions, and 10 of those were
  marked unanswerable by the QASPER annotators too. Evidence F1 rose from 39.1 to 48.0.
- **Off by default in private mode.** With local models it didn't pay. Across four variants on 20 questions it
  rejected 0 to 5 questions, at most 2 of them truly unanswerable, and judge-correct stayed at 70 or fell to 60
  (`ce_cpu_gpu`, `ce_prejedge_lenient`, `ce_8b_as_prejudge`, `ce_strict`). It remains available as
  `PREJUDGE_ENABLED=true`.
- **Fail open.** If the pre-judge or the follow-up rewrite fails or returns nonsense, the question is answered anyway.
  A helper step should never be the reason the user gets no answer.

### One model on the GPU at a time

An 8 GB card can't hold the context model and the answering model together. Running the second one on the CPU took
16.9 s per call against 0.3 s on the GPU (`ce_cpu_gpu` against `ce_gpu_load_unload`). So the backend sequences them:
the context model is loaded when an upload arrives, unloaded as soon as contexts are written, and the answering model
starts loading while the upload is still embedding and indexing. The answering model also does the follow-up rewrite
and the pre-judge, so chatting never swaps models. A load takes about 45 s on an RTX 4070 laptop GPU, and the UI shows
it instead of appearing to hang.

### Storage: MongoDB Atlas Local for everything

One container holds documents, chunks, chats and the vector index, so there is no second store to keep in sync and a
delete is one place. The rejected options were a dedicated vector database (a second service for a few thousand
vectors) and an in-process index such as FAISS (no persistence or filtering without extra code). Two details:

- If the vector index is unavailable, the backend falls back to an in-process cosine search, so chat keeps working.
- Vectors from different embedding models can't be compared, so each mode has its own chunk field and index.

### Markdown as the single intermediate format

Every file type becomes Markdown first. That gives one chunker for all formats, headings to chunk along, and a
document the viewer can render. Chunks store their character range in that Markdown, which is what makes citation
highlighting exact rather than a text search.

### Frontend

- **Next.js with hand-written components and CSS.** The UI is small (an upload page, a chat and a document panel), and
  the unusual parts, citation chips and range highlighting, needed custom code anyway.
- **All backend calls go through a Next.js proxy route** (`/api/*`), so the browser talks to one origin and there is
  no CORS configuration.
- **NDJSON over a streamed `fetch`, not WebSockets or SSE.** The stream is one-directional and needs a POST body.
  Typed events (`sources`, `prejudge`, `token`, `done`, `error`) let the UI show the passages before the first token.

### What I left out on purpose

- **Authentication and multi-user separation.** It runs locally for one user; MongoDB uses `admin`/`admin` and its
  port is exposed for inspection. This is not a production setup.
- **Questions across several documents.** Retrieval is filtered to one document, which keeps answers and citations
  unambiguous.
- **OCR and images.** Scanned PDFs are rejected with a clear message; pictures and charts are skipped.
- **A RAG framework.** The pipeline is about 2,100 lines of Python, and every stage can be timed and swapped
  separately, which the evaluation depends on.
- **Automated tests.** There are none. The QASPER evaluation checks the pipeline end to end, but it is not a
  regression suite.

### Limits of the numbers

- The judge is `qwen3:8b`, which is also private mode's answering model. A model grading its own answers is lenient, so
  private mode's judge scores are probably flattering.
- QASPER papers are English NLP research papers. The scores describe the pipeline, not your documents.
- The cloud comparison was run with BM25 20 + vector 30 candidates. The shipped setting is 30 + 30 and has not been
  re-measured.
- Answer F1 is low everywhere (20–27) because it penalises answers longer than the terse gold answers. I used it only
  to compare my own runs.

## 4. What I would do next

In order of value:

1. **Questions across several documents.** Upload a set, retrieve across it, and show the source document in each
   citation. The brief asks for "one or more documents"; today that means one per chat.
2. **Whole-document questions.** "Summarise this" and "list every…" can't be answered from 8 passages. I would route
   those to a map-reduce summary, or for documents that fit the context window, send the whole document.
3. **An independent judge and a second dataset.** Grade with a stronger model than the one being graded, and add a set
   of manuals and reports with their own questions, since the real documents won't be NLP papers.
4. **Tests and CI.** Unit tests for the chunker, converter and fusion, an API test against a tiny document, and the
   5-paper evaluation as a regression check with thresholds.
5. **Faster uploads.** Chunk contexts are most of the upload time and are written one by one. Batching several chunks
   per call, or running calls in parallel in cloud mode, would cut it considerably.
6. **OCR, tables and images.** Add OCR for scanned PDFs and describe figures with a vision model.
7. **Other languages.** Private mode's embeddings and BM25 stemming are English-only. A multilingual embedding model
   and language-aware tokenising would fix that; it matters for German documents.
8. **Re-tune cloud mode.** Measure the 30 + 30 setting, try a stronger reranker now that MiniLM has been shown to hurt
   there, and tune the unused similarity cutoff.
9. **Production hardening.** Authentication, per-user data, secrets out of `.env`, upload scanning, rate limits, and
   an ingestion queue in place of in-process background tasks, which lose work if the backend restarts.

## Reference

### Configuration (`.env`)

Values are the ones in the committed `.env`. Apply changes with `.\start.ps1` or `./start.sh`.

| Variable | Value | Notes |
|---|---|---|
| `LLM_MODEL` | `qwen3:8b` | Private mode's answering model (any Ollama chat model) |
| `LLM_THINK` | `false` | `true` enables reasoning: slower, and the reasoning is not shown |
| `LLM_NUM_CTX` | `8192` | Ollama context window |
| `LLM_TEMPERATURE` | `0` | `0` makes answers repeatable, which keeps evaluation runs comparable |
| `LLM_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded after the last request |
| `LLM_TIMEOUT` / `LLM_LOAD_TIMEOUT` | `600` / `1800` | Seconds to wait for Ollama output / for a model to load |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Characters. Affects newly uploaded documents only |
| `TOP_K` | `8` | Passages sent to the pre-judge and the answering model |
| `BM25_CANDIDATES` / `VECTOR_CANDIDATES` | `30` / `30` | Private mode: chunks each search contributes to the fusion |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RERANKER_ENABLED` | `true` | Private mode: rerank the fused candidates. `false` sends the top `TOP_K` by fused rank |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Or `BAAI/bge-reranker-base`. Both are in the image |
| `PREJUDGE_ENABLED` | `false` | Private mode: the answering model checks the passages (YES / NO) before answering |
| `QUERY_REWRITE_ENABLED` | `true` | Rewrite follow-up questions into standalone ones before retrieval |
| `HISTORY_TURNS` | `6` | Previous chat messages sent with each question |
| `MAX_CHATS_PER_DOCUMENT` | `10` | Not in `.env`; default from `docker-compose.yml` |
| `CONTEXTUAL_EMBEDDING` | `true` | Write a context for every chunk. Affects newly uploaded documents only |
| `CONTEXT_MODEL` | `qwen3:4b-instruct` | Must be a non-thinking model: plain `qwen3:4b` writes its reasoning instead |
| `CONTEXT_NUM_CTX` | `16384` | Tokens of the document the context model reads at once; longer documents are read in windows |
| `MAX_UPLOAD_MB` | `25` | Upload limit |
| `JUDGE_MODEL` | `qwen3:8b` | Ollama model that grades answers in the evaluation |
| `CLOUD_EMBED_MODEL` / `CLOUD_EMBED_DIM` | `google/gemini-embedding-2` / `768` | Changing the dimension needs the `chunk_vector_index_cloud` index to be dropped |
| `CLOUD_CONTEXT_MODEL` | `google/gemini-3.5-flash-lite` | Writes chunk contexts in cloud mode |
| `CLOUD_LLM_MODEL` | `deepseek/deepseek-v4.1-flash` | Answers and rewrites in cloud mode. Must allow reasoning to be switched off |
| `CLOUD_PREJUDGE_MODEL` | `google/gemini-3.5-flash-lite` | ALL / PARTIAL / NONE for every cloud question |
| `CLOUD_BM25_CANDIDATES` / `CLOUD_VECTOR_CANDIDATES` | `30` / `30` | Cloud mode: chunks each search contributes |
| `CLOUD_VECTOR_MIN_SCORE` | `0` | Minimum cosine similarity for cloud vector hits; `0` = no cutoff |
| `MONGO_USER` / `MONGO_PASSWORD` | `admin` / `admin` | Local MongoDB credentials |
| `OPENROUTER_API_KEY` | – | From your environment, not `.env` (see [How to run it](#1-how-to-run-it)) |
| `COMPOSE_FILE` | – | Written by the start script: the GPU or CPU setup |

Cloud model IDs are OpenRouter's. Rate-limited or overloaded requests (429 / 5xx) are retried up to 5 times, 2–32 s
apart.

### API

| Method | Path | |
|---|---|---|
| `GET` | `/health` | status, vector index availability, configured models |
| `GET` | `/modes` | private and cloud mode, and the API keys cloud mode still needs |
| `GET` | `/answering-model` | private mode's answering model: `loaded` / `loading` / `not_loaded`, with load timing |
| `POST` | `/documents` | multipart `file` + `mode` (`private` / `cloud`) → `{doc_id, status}` (202) |
| `GET` | `/documents` | every document, newest first, with `chat_count` |
| `GET` | `/documents/{id}` | status: `queued → converting → chunking → contextualizing → embedding → storing → indexing → ready` (or `failed` + `error`) |
| `GET` | `/documents/{id}/markdown` | the converted Markdown |
| `DELETE` | `/documents/{id}` | removes the document with its chunks, chats and file |
| `DELETE` | `/documents` | New session: removes every document |
| `GET` | `/documents/{id}/chats` | the document's chats, most recently used first |
| `POST` | `/documents/{id}/chats` | a new empty chat (or the existing empty one); 409 at the limit |
| `GET` | `/chats/{id}` | the chat with its messages, sources and verdicts |
| `POST` | `/chats/{id}/messages` | saves a question and its answer; the first question becomes the title |
| `DELETE` | `/chats/{id}` | deletes the chat |
| `POST` | `/chat` | `{doc_id, question, history}` → NDJSON: `sources`, `prejudge`, `token`…, `done` / `error` |

### Project layout

```
backend/app/
  main.py           FastAPI routes
  config.py         settings, read from environment variables
  modes.py          private / cloud: models, retrieval sizes, embedding field and index
  db.py             MongoDB (the only module that talks to it)
  ingest.py         ingestion pipeline and status updates
  converter.py      PDF / TXT / MD → Markdown
  chunker.py        heading-aware chunking
  contextual.py     Contextual Retrieval (a context per chunk)
  embeddings.py     bge-small-en-v1.5 (private mode)
  retrieval.py      BM25 + vector search → RRF fusion → reranking, as separate stages
  reranker.py       cross-encoder reranking
  rewrite.py        follow-up question rewriting
  prejudge.py       do the passages hold all, part or none of the answer?
  answer_prompt.py  the answering prompt, shared by both modes
  ollama.py         Ollama client (load / unload, completions, streaming)
  openrouter.py     OpenRouter client (embeddings, completions, streaming, retries)
  evaluate/         QASPER evaluation (python -m app.evaluate)
frontend/
  app/page.tsx                     upload → chat flow, top bar
  app/api/[...path]/route.ts       proxy to the backend
  hooks/                           open document, chat list, chat state and streaming, model status
  components/Uploader.tsx          mode choice + file + upload (ModePicker, FileDropzone)
  components/ChatWindow.tsx        the chat: header, messages, composer, document panel
  components/AssistantMessage.tsx  answer with Markdown, math, code, citation chips and sources
  components/DocumentViewer.tsx    the document with the cited passage highlighted
  components/DocumentList.tsx      Chat history: documents and their chats
  lib/api.ts                       backend API types and calls
  lib/markRange.ts                 rehype plugin that highlights a range of the Markdown source
evaluation_metrics/                one JSON per run, results.md and result_40.md
```

### Evaluation (QASPER)

It only runs when you trigger it:

```bash
docker compose exec backend python -m app.evaluate --run-id baseline-1 --papers 5 --seed 0
```

| Option | Meaning |
|---|---|
| `--run-id` | Unique name for the run. Reusing one is refused |
| `--papers` | How many papers to sample (default 5); about 3.5 questions per paper |
| `--seed` | Which papers get sampled (default 0). Same seed = same papers |
| `--split` | `test` (default) or `validation` |
| `--mode` | `private` (default), `cloud-rerank`, `cloud-prejudge` or `cloud-rerank-prejudge` |
| `--sweep` | e.g. `30,25,20,15`: ingest once, then evaluate BM25 top N + vector top N for each N |
| `--compare` | e.g. `rerank,prejudge,rerank-prejudge`: ingest once in cloud mode, then evaluate each variant |

It runs in four phases, one model at a time: ingest each paper like an upload; retrieve and pre-judge every question;
answer; judge. The papers' chunks are deleted afterwards, so nothing shows up in the app. On a GPU, 40 papers take
about 45 minutes; on CPU expect 30–90 s per question.

Scores (0–100):

- **Retrieval recall@k**: share of gold evidence paragraphs in the top-k passages. Measures retrieval on its own.
- **Evidence F1**: official QASPER paragraph F1 over the passages the answer cites.
- **Answer F1**: official QASPER token F1 against the closest annotator answer.
- **Judge correct**: `JUDGE_MODEL` decides whether the answer matches a reference answer.

Low recall means the answer never reached the LLM (tune retrieval). High recall with a low judge score points to the
prompt or the model.

Output in `evaluation_metrics/`: `<run-id>.json` holds the metrics, timings, every setting the run used and every
question's answer, passages and judge output; `result_40.md` gets one row per run (`results.md` holds the earlier
runs). The run `cloud40-prejudge-jev` is a failed experiment (102 of 135 questions errored); ignore its scores.

### Troubleshooting

- **Scanned PDFs** fail with "no extractable text": OCR isn't included.
- **Slow answers on CPU**: expected for an 8B model. Use cloud mode, a GPU, or a smaller `LLM_MODEL`.
- **The first answer takes a minute**: the answering model is loading; the chat shows a loading bar.
- **Cloud option disabled, or "needs OPENROUTER_API_KEY"**: the backend was started without the key. See
  [How to run it](#1-how-to-run-it) and restart with the start script.
- **"Reasoning is mandatory for this endpoint"** (OpenRouter 400): the model set as `CLOUD_LLM_MODEL` doesn't allow
  reasoning to be switched off. Pick one that does.
- **"OpenRouter error 404"**: the model ID in `.env` doesn't exist on OpenRouter.
- **Vector index not ready**: `docker compose logs backend` shows its status. If it never becomes available, the
  backend uses in-process cosine search and chat still works.
- **Reset everything**: `docker compose down -v` deletes documents, vectors and the downloaded models.
