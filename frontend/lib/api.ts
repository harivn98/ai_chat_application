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

export interface DocInfo {
  doc_id: string;
  filename: string;
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

export async function uploadDocument(file: File): Promise<DocInfo> {
  const form = new FormData();
  form.append("file", file);
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
