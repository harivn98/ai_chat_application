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

1. **Convert to Markdown** — PDF via `pymupdf4llm` (keeps headings, lists, tables), TXT wrapped as Markdown (encoding-safe), MD cleaned (front matter removed). The Markdown is stored in MongoDB.
2. **Chunk** — heading-aware: split by `#` structure first, then packed into ~1000-char chunks on paragraph/sentence boundaries with 150-char overlap. Code fences and tables stay intact. Each chunk carries its heading path (`Manual > Braking > Pads`) and its position in the Markdown, used to highlight it in the document.
3. **Add context** (Contextual Retrieval, `CONTEXTUAL_EMBEDDING=true`) — `qwen3:4b-instruct` reads the document and writes one sentence per chunk saying where it fits (e.g. *"This chunk describes the similarity measure used in the sentence-based approach…"*). The context is prepended to the text that gets embedded and BM25-indexed. The answering LLM and the UI still see the original chunk. The document goes first in every prompt, so Ollama reuses its cache and each chunk takes about 1 s on a GPU. Documents longer than `CONTEXT_NUM_CTX` are read in windows that each start with the document's opening.
4. **Embed** — `BAAI/bge-small-en-v1.5` (384-d, normalised). The model is baked into the backend image, so it runs offline.
5. **Store** — chunks, contexts and vectors go into MongoDB `ragdb.chunks`, with an Atlas Vector Search index (`cosine`, filtered by `doc_id`).
6. **Wait for the index** — mongot syncs asynchronously, so the document only flips to `ready` once `$vectorSearch` can see every chunk. The chat opens right after the upload; questions typed before this point wait and are sent once the document is ready.

**Query** (`POST /chat`, streamed as NDJSON)

1. **BM25** (rank-bm25, BM25+ variant, Snowball-stemmed tokens) over the document's chunks → top 10.
2. **Dense** — query embedded with the bge query instruction, `$vectorSearch` → top 20, keeping only chunks with cosine similarity ≥ 0.85.
3. **Reciprocal Rank Fusion** (k=60) → top 20 candidates (top 5 when the reranker is off).
4. **Rerank** (`RERANKER_ENABLED=true`) — the cross-encoder `BAAI/bge-reranker-base` (278M parameters, on the CPU so the GPU stays free for Qwen3 8B) reads the question together with each candidate (`RERANK_CANDIDATES`, default 20) and keeps the 5 best. It replaced `cross-encoder/ms-marco-MiniLM-L6-v2`, which ranked well among BM25's ~10 candidates but not among 30 mixed ones. Expect roughly a few seconds per question on 4 CPU cores. BM25 and the embeddings score the question and a chunk separately; a cross-encoder reads them together, so it ranks far more precisely. It can only choose among the candidates, so it can't recover a chunk neither retriever found. The UI shows each kept passage's pre-rerank position as *Fused #n*.
5. **Pre-judge** (`PREJUDGE_ENABLED=true`) — Qwen3 8B, the answering model, reads the 5 passages and the question and answers YES or NO: do the passages contain the specific information the question asks for? It is strict: passages that are only on the same topic get NO, so questions the document doesn't actually answer are caught. Answers that follow clearly from stated facts (e.g. settling a yes/no question) still get YES. On **NO**, answer generation is skipped and the reply is *"There isn't enough content in the document to answer this question."*, with the passages still shown.
6. **Qwen3 8B** (Ollama, on the GPU, `think: false`, `num_ctx` 8192) answers only from the passages, citing them as `[1]`, `[2]`. The UI shows each passage with its BM25/vector rank. Clicking a citation opens the document in a side panel, scrolled to the cited passage, which is highlighted (uploads from before this feature fall back to the passage list; re-upload them to get highlighting).

**GPU use:** all models run on the GPU, one at a time. `qwen3:4b-instruct` is used only to write chunk contexts during an upload. Because a new session starts with an upload, the backend preloads `qwen3:4b-instruct` at startup; a later upload starts loading it the moment the file arrives, while the PDF is still converted and chunked. (With `CONTEXTUAL_EMBEDDING=false`, Qwen3 8B is preloaded at startup instead.) As soon as contextualizing finishes, it is unloaded and Qwen3 8B starts loading in the background, while the upload is still embedding and indexing, so it is usually ready before the first question. Qwen3 8B then does both the pre-judge and the answer, so chatting never swaps models. Loading takes about 45 s for Qwen3 8B and 25 s for `qwen3:4b-instruct` on an RTX 4070 laptop GPU; unloading is instant.

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

## Private and cloud mode

Before each upload you pick where the document is processed:

| | Private mode (default) | Cloud mode |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (CPU) | Gemini Embedding 2 (`google/gemini-embedding-2`, cut to 768-d) |
| Chunk contexts | `qwen3:4b-instruct` (Ollama) | Gemini Flash-Lite (`google/gemini-3.5-flash-lite`) |
| Answers + pre-judge | `qwen3:8b` (Ollama) | DeepSeek V4.1 Flash (`deepseek/deepseek-v4.1-flash`, reasoning off) |
| BM25, reranker, MongoDB | this machine | this machine (same MongoDB) |
| Data leaving the machine | none | the document text and your questions with the retrieved passages, to OpenRouter and on to Google and DeepSeek |

All cloud models run through [OpenRouter](https://openrouter.ai), with one API key and one bill. OpenRouter
returns Gemini Embedding 2's full 3072 values; the model is trained so that the first values form an embedding on
their own, so the backend keeps the first `CLOUD_EMBED_DIM` and renormalises them.

The mode belongs to the document: vectors from different embedding models can't be compared, so each mode keeps
its vectors in its own chunk field (`embedding` / `embedding_cloud`) with its own vector index, and a document is
always answered in the mode it was uploaded in. The UI shows the mode in the top bar and the chat header.

**API key.** Cloud mode needs `OPENROUTER_API_KEY` ([OpenRouter keys](https://openrouter.ai/settings/keys), with
credits on the account). Don't put it in `.env`: it is committed to git. Set it as an environment variable of your
user, then start the app from a new terminal:

```powershell
setx OPENROUTER_API_KEY "sk-or-..."      # Windows; then open a new terminal
.\start.ps1
```
```bash
export OPENROUTER_API_KEY=sk-or-...      # Linux / macOS (e.g. in ~/.bashrc)
./start.sh
```

Without the key the cloud option is shown but disabled. Rate-limited or overloaded requests (429 / 5xx) are
retried up to 5 times, 2-32 s apart.

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
| `RERANKER_ENABLED` | `true` | `true` / `false`: rerank the fused candidates with the cross-encoder before taking the top `TOP_K` |
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | Which reranker runs: `BAAI/bge-reranker-base` (more accurate, ~6–9 s for 30 chunks on 4 CPU cores) or `cross-encoder/ms-marco-MiniLM-L6-v2` (~1 s, weaker on large candidate pools). Both are baked into the image |
| `RERANK_CANDIDATES` | `20` | How many fused chunks the reranker chooses from |
| `HISTORY_TURNS` | `6` | Previous chat messages sent with each question |
| `CONTEXTUAL_EMBEDDING` | `true` | `true` / `false`: add an LLM-written context to every chunk before embedding + BM25. Affects newly uploaded documents |
| `CONTEXT_MODEL` | `qwen3:4b-instruct` | Ollama model that writes the contexts. Use a non-thinking model: plain `qwen3:4b` is thinking-only and writes its reasoning instead |
| `CONTEXT_NUM_CTX` | `16384` | Tokens of the document the context model reads at once. Longer documents are split into windows |
| `PREJUDGE_ENABLED` | `true` | `true` / `false`: check whether the retrieved passages can answer before calling the answering LLM |
| `MAX_UPLOAD_MB` | `25` | Upload limit |
| `CLOUD_EMBED_MODEL` / `CLOUD_EMBED_DIM` | `google/gemini-embedding-2` / `768` | Cloud embeddings. Changing the dimension needs the `chunk_vector_index_cloud` index to be dropped so it is recreated |
| `CLOUD_VECTOR_MIN_SCORE` | `0` | Minimum cosine similarity for cloud embedding hits; `0` = no cutoff. Not tuned yet: measure it with `--mode cloud` |
| `CLOUD_CONTEXT_MODEL` | `google/gemini-3.5-flash-lite` | Writes the chunk contexts in cloud mode |
| `CLOUD_LLM_MODEL` | `deepseek/deepseek-v4.1-flash` | Answers and pre-judges in cloud mode (`LLM_THINK` and `LLM_TEMPERATURE` apply to it too) |
| `OPENROUTER_API_KEY` | – | From your environment, not `.env` (see [Private and cloud mode](#private-and-cloud-mode)). Cloud model IDs are OpenRouter's |
| `JUDGE_MODEL` | `qwen3:8b` | Ollama model that grades answers in the [evaluation](#evaluation-qasper) |

Changing chunking settings only affects newly uploaded documents.

## API

| Method | Path | |
|---|---|---|
| `GET` | `/modes` | private and cloud mode with their models, and the API keys cloud mode still needs |
| `POST` | `/documents` | multipart `file` + `mode` (`private` / `cloud`) → `{doc_id, status}` (202) |
| `GET` | `/documents/{id}` | status: `queued → converting → chunking → contextualizing → embedding → storing → indexing → ready` (or `failed` + `error`) |
| `GET` | `/documents/{id}/markdown` | the converted Markdown |
| `DELETE` | `/documents/{id}` | removes doc, chunks and files |
| `POST` | `/chat` | `{doc_id, question, history}` → NDJSON: `sources`, `token`…, `done` / `error` |

## Project layout

```
backend/app/
  main.py        FastAPI routes
  config.py      settings, read from environment variables
  db.py          MongoDB, vector index creation and sync
  ingest.py      ingestion pipeline + status updates
  converter.py   PDF/TXT/MD → Markdown
  chunker.py     heading-aware chunking (each chunk keeps its position in the Markdown)
  contextual.py  Contextual Retrieval (per-chunk context from a small LLM)
  embeddings.py  bge-small-en-v1.5
  retrieval.py   BM25 + $vectorSearch + RRF
  reranker.py    cross-encoder reranking of the fused candidates
  prejudge.py    YES/NO check: can the retrieved passages answer the question?
  llm.py         Ollama client (load/unload, completions, streaming) + answering prompt
  modes.py       private / cloud mode: which models, embedding field and vector index a document uses
  cloud.py       OpenRouter client for cloud mode (Gemini embeddings and contexts, DeepSeek answers)
  evaluate.py    QASPER evaluation (CLI)
  qasper.py      QASPER dataset download, papers as Markdown, official scoring
frontend/
  app/page.tsx                     upload → chat flow
  app/api/[...path]/route.ts       proxy to the backend (single exposed origin)
  components/Uploader.tsx          file picker + upload
  components/ModePicker.tsx        private / cloud mode choice for the upload
  components/IngestProgress.tsx    indexing progress, shown in the chat
  components/ChatWindow.tsx        chat state: question queue, streaming, composer
  components/AssistantMessage.tsx  answer with Markdown, KaTeX math, code highlighting, citation chips and sources
  components/DocumentViewer.tsx    source document panel with the cited passage highlighted
  lib/api.ts                       backend API types and calls
  lib/answerMarkdown.ts            prepares the answer's Markdown (math delimiters, citation links)
  lib/markRange.ts                 rehype plugin that highlights a range of the Markdown source
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
| `--mode` | `private` (default) or `cloud`: ingest, retrieve and answer with the cloud models (sends the papers through OpenRouter). The judge stays `JUDGE_MODEL` in Ollama, so both modes are graded the same way. Cloud rows start with `cloud · emb <model>` in the Retrieval config column. |

On CPU, expect roughly 30–90 s per question: `--papers 1` checks that it works, `--papers 5` is a quick comparison, and `--papers 20` or more gives more stable numbers. The dataset (~4 MB) is downloaded on the first run.

**What happens.** It runs in four phases, loading each model once per run:

1. **Ingest** (`qwen3:4b-instruct`): each sampled paper is converted to Markdown and ingested like an upload (chunk, add context if `CONTEXTUAL_EMBEDDING=true`, embed, store).
2. **Retrieve + pre-judge** (`qwen3:8b`, loaded right after the context model is unloaded): every question goes through hybrid retrieval and the pre-judge.
3. **Answer** (`qwen3:8b`, still loaded): questions the pre-judge let through are answered as in the chat. Rejected ones get the "not enough content" reply and count as answered "Unanswerable".
4. **Judge** (`JUDGE_MODEL`): the judge model grades each answer.

The timings exclude model loading. The chat behaves the same way: the pre-judge and the answer use the same loaded model, so a question never waits for a model swap.

The papers' chunks are deleted afterwards, so nothing shows up in the app.

**Timings** (averages): contextualization (the context model writing contexts for one paper; `–` when off), embedding (embed one paper), retrieval (query embedding + BM25 + vector search + RRF), rerank (cross-encoder scoring of the candidates; `–` when off), pre-judge (the YES/NO check), generation (full answer, for questions the pre-judge let through), and judging (one judge call). **Pre-judge rejected** shows how many questions got "not enough content", and how many of those an annotator also marked unanswerable (those rejections were right). With the reranker on, the console and `<run-id>.json` also report **candidate recall**: how often the gold evidence was among the chunks the reranker chose from. That is the most reranking can reach; the gap between it and retrieval recall@k is what the reranker missed. Time spent waiting for the vector index to sync isn't counted.

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
