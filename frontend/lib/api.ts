// Types and calls for the backend API (proxied under /api by app/api/[...path]/route.ts)

export type DocStatus =
  | "queued"
  | "converting"
  | "chunking"
  | "contextualizing"
  | "embedding"
  | "storing"
  | "indexing"
  | "ready"
  | "failed";

/** private: everything runs on this machine; the cloud modes use Gemini + DeepSeek via OpenRouter (the document
 * leaves this machine), with the local reranker or with the Flash-Lite pre-judge */
export type ModeId = "private" | "cloud-rerank" | "cloud-prejudge";

/** The pre-judge's finding: the passages hold all, part or none of the answer */
export type Verdict = "all" | "partial" | "none";

export interface ModeInfo {
  id: ModeId;
  label: string;
  description: string;
  embed_model: string;
  context_model: string | null; // null when contextual embedding is off
  llm_model: string;
  reranker_model: string | null; // null: this mode doesn't rerank
  prejudge_model: string | null; // null: this mode doesn't pre-judge
  missing_keys: string[]; // API keys the backend still needs for this mode
}

export interface DocInfo {
  doc_id: string;
  filename: string;
  mode: ModeId; // chosen at upload; the document is answered in this mode
  mode_label: string;
  embed_model: string;
  llm_model: string;
  status: DocStatus;
  progress: number;
  num_chunks: number | null;
  contextual: boolean;
  context_model: string | null;
  context_done: number | null;
  error: string | null;
  failed_stage: DocStatus | null;
}

/** A retrieved passage; `id` is the number the answer cites it with, e.g. [1]. */
export interface Source {
  id: number;
  section: string;
  text: string;
  start: number | null; // position in the document's Markdown (UTF-16 units); null for older uploads
  end: number | null;
  bm25_rank: number | null;
  vector_rank: number | null;
  fused_rank: number | null; // position after BM25 + vector fusion, before reranking
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

type StreamEvent =
  | { type: "sources"; sources: Source[] }
  | { type: "prejudge"; verdict: Verdict }
  | { type: "token"; content: string }
  | { type: "done" }
  | { type: "error"; message: string };

async function errorText(res: Response): Promise<string> {
  try {
    const j = await res.json();
    return typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
  } catch {
    return `Request failed (${res.status})`;
  }
}

export async function getModes(): Promise<ModeInfo[]> {
  const res = await fetch("/api/modes", { cache: "no-store" });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function uploadDocument(file: File, mode: ModeId): Promise<DocInfo> {
  const form = new FormData();
  form.append("file", file);
  form.append("mode", mode);
  const res = await fetch("/api/documents", { method: "POST", body: form });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function getDocument(docId: string): Promise<DocInfo> {
  const res = await fetch(`/api/documents/${docId}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function getMarkdown(docId: string): Promise<string> {
  const res = await fetch(`/api/documents/${docId}/markdown`);
  if (!res.ok) throw new Error(await errorText(res));
  return res.text();
}

export async function deleteDocument(docId: string): Promise<void> {
  await fetch(`/api/documents/${docId}`, { method: "DELETE" });
}

export async function streamChat(
  body: { doc_id: string; question: string; history: ChatTurn[] },
  onEvent: (e: StreamEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(await errorText(res));

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (line) onEvent(JSON.parse(line) as StreamEvent);
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer) as StreamEvent);
}
