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

/** private: everything runs on this machine; cloud: cloud models via OpenRouter (the document leaves this machine),
 * with the pre-judge always on */
export type ModeId = "private" | "cloud";

/** The pre-judge's finding: the passages hold all, part or none of the answer */
export type Verdict = "all" | "partial" | "none";

export interface ModeInfo {
  id: ModeId;
  label: string;
  description: string;
  missing_keys: string[]; // API keys the backend still needs for this mode
}

export interface DocInfo {
  doc_id: string;
  filename: string;
  mode: ModeId; // chosen at upload; the document is answered in this mode
  mode_label: string; // e.g. "Cloud · pre-judge"
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
  created_at: string | null; // ISO date of the upload
}

/** A document in the "Your documents" list */
export interface ListedDocument extends DocInfo {
  chat_count: number;
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

/** Private mode's answering model in Ollama. Ollama doesn't report how far a load is, so the UI compares
 * elapsed_s with expected_s, the seconds its last load took (null until one was measured). */
export interface ModelStatus {
  model: string;
  state: "loaded" | "loading" | "not_loaded";
  elapsed_s: number | null; // while loading
  expected_s: number | null;
}

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

/** A chat about a document, as the chat list shows it. `title` is its first question ("" while it is empty). */
export interface ChatInfo {
  chat_id: string;
  doc_id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

/** A question or an answer as a chat keeps it. */
export interface StoredMessage extends ChatTurn {
  sources?: Source[];
  verdict?: Verdict;
  error?: string;
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

export async function getAnsweringModel(): Promise<ModelStatus> {
  const res = await fetch("/api/answering-model", { cache: "no-store" });
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

export async function listDocuments(): Promise<ListedDocument[]> {
  const res = await fetch("/api/documents", { cache: "no-store" });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function deleteDocument(docId: string): Promise<void> {
  const res = await fetch(`/api/documents/${docId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await errorText(res));
}

/** New session: every document with its chats. */
export async function deleteAllDocuments(): Promise<void> {
  const res = await fetch("/api/documents", { method: "DELETE" });
  if (!res.ok) throw new Error(await errorText(res));
}

/** The document's chats, most recently used first, and how many a document can have. */
export async function listChats(docId: string): Promise<{ chats: ChatInfo[]; max_chats: number }> {
  const res = await fetch(`/api/documents/${docId}/chats`, { cache: "no-store" });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

/** A new empty chat about the document (or its existing empty chat: there is never more than one). */
export async function createChat(docId: string): Promise<ChatInfo> {
  const res = await fetch(`/api/documents/${docId}/chats`, { method: "POST" });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function getChat(chatId: string): Promise<ChatInfo & { messages: StoredMessage[] }> {
  const res = await fetch(`/api/chats/${chatId}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

/** Save a question and its answer at the end of the chat. */
export async function saveMessages(chatId: string, messages: StoredMessage[]): Promise<ChatInfo> {
  const res = await fetch(`/api/chats/${chatId}/messages`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  if (!res.ok) throw new Error(await errorText(res));
  return res.json();
}

export async function deleteChat(chatId: string): Promise<void> {
  const res = await fetch(`/api/chats/${chatId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await errorText(res));
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
