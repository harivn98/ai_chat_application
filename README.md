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
3. **Embed** — `BAAI/bge-small-en-v1.5` (384-d, normalised). The model is baked into the backend image, so it runs offline.
4. **Store** — chunks + vectors go into MongoDB `ragdb.chunks`, with an Atlas Vector Search index (`cosine`, filtered by `doc_id`).
5. **Wait for the index** — mongot syncs asynchronously, so the document only flips to `ready` once `$vectorSearch` can see every chunk. **The chat window appears only after this point.**

**Query** (`POST /chat`, streamed as NDJSON)

1. **BM25** (rank-bm25, BM25+ variant, Snowball-stemmed tokens) over the document's chunks → top 10.
2. **Dense** — query embedded with the bge query instruction, `$vectorSearch` → top 20, keeping only chunks with cosine similarity ≥ 0.85.
3. **Reciprocal Rank Fusion** (k=60) → top 5 passages.
4. **Qwen3 8B** (Ollama, `think: false`, `num_ctx` 8192) answers only from the passages, citing them as `[1]`, `[2]`. The UI shows each passage with its BM25/vector rank; clicking a citation jumps to it.

## Run it

Requirements: Docker Desktop / Docker Engine with Compose v2, ~12 GB free disk (Qwen3 8B ≈ 5.2 GB), 16 GB RAM recommended.

```bash
cp .env.example .env              # optional, defaults work
docker compose up --build
```

First start pulls images, builds both apps and downloads `qwen3:8b` (the `ollama-init` job). When the logs show the frontend is up, open **http://localhost:3000**.

### NVIDIA GPU (strongly recommended for Qwen3 8B)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

On Windows this needs Docker Desktop with the WSL 2 backend and a current NVIDIA driver; on Linux the NVIDIA Container Toolkit. An 8 GB card fits Qwen3 8B (Q4) with the 8k context. Check it's on the GPU with `docker compose exec ollama ollama ps` (should say `100% GPU`).

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
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Characters |
| `TOP_K` | `5` | Passages sent to the LLM after fusion |
| `BM25_CANDIDATES` | `10` | Chunks taken from BM25 before fusion |
| `VECTOR_MIN_SCORE` | `0.85` | Minimum cosine similarity for embedding hits |
| `MAX_UPLOAD_MB` | `25` | Upload limit |

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
docker compose exec backend python -m app.evaluate --run-id baseline-1 --papers 5 --seed 0
```

Each sampled paper is converted to Markdown and ingested like an upload (chunk, embed, store). Every question on it then goes through hybrid retrieval and Qwen3, and each stage is timed. Scores:

- **Answer F1**: official QASPER token F1 against the closest annotator answer, also broken down by answer type. It penalizes long answers, even correct ones.
- **Evidence F1**: official QASPER paragraph F1, using the paper paragraphs inside the passages the answer cites.
- **Retrieval recall@k**: share of gold evidence paragraphs that are in the top-k retrieved chunks. This measures retrieval on its own.
- **Judge correct**: `JUDGE_MODEL` decides whether the answer matches a reference answer.

Notes:

- One row per run is appended to `evaluation/results.md`. Run IDs must be unique.
- Per-question answers, references and judge output go to `evaluation/runs/<run-id>.json`.
- Keep `--seed` fixed to compare runs on the same papers. `--split validation` uses the dev set instead of test.
- `JUDGE_MODEL` defaults to the answering model. A model judging its own answers is lenient, so use a different one when possible (`docker compose exec ollama ollama pull <model>`).

## Troubleshooting

- **Scanned PDFs** fail with "no extractable text" — OCR isn't included. Adding `tesseract-ocr` to the backend image enables pymupdf4llm's OCR path.
- **Slow answers on CPU** — expected for an 8B model; use the GPU override or a smaller model (`LLM_MODEL=qwen3:4b`).
- **Vector index not ready** — `docker compose logs backend` shows the index status. If mongot never becomes available, the backend falls back to in-process cosine search, so chat still works.
- **Reset everything** — `docker compose down -v` (deletes documents, vectors and the downloaded model).
