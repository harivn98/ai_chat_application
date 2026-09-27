"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AssistantMessage from "@/components/AssistantMessage";
import DocumentViewer, { Highlight } from "@/components/DocumentViewer";
import IngestProgress from "@/components/IngestProgress";
import { ChatTurn, DocInfo, Source, streamChat } from "@/lib/api";

const uid = () => Math.random().toString(36).slice(2, 10);

// A question asked while the document is still being indexed waits in a `queued` assistant message
// (with the `question` it answers) and is sent once the document is ready
type ChatMessage = ChatTurn & {
  id: string;
  sources?: Source[];
  streaming?: boolean;
  error?: string;
  queued?: boolean;
  question?: string;
};

export default function ChatWindow({ doc, onNewDocument }: { doc: DocInfo; onNewDocument: () => void }) {
  const ready = doc.status === "ready";
  const failed = doc.status === "failed";
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const sendingRef = useRef(false); // guards against sending the same queued question twice
  // Source document panel: null = closed; highlight null = whole document without a highlight
  const [viewer, setViewer] = useState<{ highlight: Highlight | null } | null>(null);
  const closeViewer = useCallback(() => setViewer(null), []);

  function showSource(s: Source) {
    const label = `passage ${s.id} · ${s.section || "Untitled section"}`;
    setViewer({ highlight: { start: s.start!, end: s.end!, label } });
  }

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [input]);

  const update = (id: string, patch: (m: ChatMessage) => Partial<ChatMessage>) =>
    setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, ...patch(m) } : m)));

  // Every question is queued; the effect below sends queued questions one at a time once the document is ready
  function send() {
    const question = input.trim();
    if (!question || failed) return;
    setMessages((ms) => [
      ...ms,
      { id: uid(), role: "user", content: question },
      { id: uid(), role: "assistant", content: "", queued: true, question },
    ]);
    setInput("");
  }

  useEffect(() => {
    if (!ready || busy || sendingRef.current) return;
    const i = messages.findIndex((m) => m.queued);
    if (i < 0) return;
    const history = messages
      .slice(0, i - 1) // everything before this question's user message
      .filter((m) => !m.error && !m.queued && m.content)
      .map((m) => ({ role: m.role, content: m.content }));
    ask(messages[i].id, messages[i].question ?? "", history);
  }, [ready, busy, messages]); // eslint-disable-line react-hooks/exhaustive-deps

  // Indexing failed: questions that were waiting can't be answered
  useEffect(() => {
    if (!failed) return;
    setMessages((ms) =>
      ms.map((m) =>
        m.queued ? { ...m, queued: false, error: "The document could not be indexed, so this question was not sent." } : m,
      ),
    );
  }, [failed]);

  async function ask(botId: string, question: string, history: ChatTurn[]) {
    sendingRef.current = true;
    update(botId, () => ({ queued: false, streaming: true }));
    setBusy(true);

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await streamChat(
        { doc_id: doc.doc_id, question, history },
        (e) => {
          if (e.type === "sources") update(botId, () => ({ sources: e.sources }));
          else if (e.type === "token") update(botId, (m) => ({ content: m.content + e.content }));
          else if (e.type === "error") update(botId, () => ({ error: e.message }));
        },
        ctrl.signal,
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") update(botId, () => ({ error: (err as Error).message }));
    } finally {
      update(botId, (m) => ({ streaming: false, content: m.content || (m.error ? "" : "_Stopped._") }));
      sendingRef.current = false;
      setBusy(false);
      abortRef.current = null;
      inputRef.current?.focus();
    }
  }

  return (
    <section className="chat card">
      <header className="chat-head">
        <div className="doc-meta">
          <span className="doc-icon" aria-hidden>
            {doc.filename.split(".").pop()?.toUpperCase()}
          </span>
          <div>
            <div className="doc-name">{doc.filename}</div>
            <div className="muted small">
              {ready
                ? `Indexed · ${doc.num_chunks} chunks · hybrid BM25 + vector retrieval`
                : failed
                  ? "Indexing failed"
                  : `Indexing… ${doc.progress}%`}
            </div>
          </div>
        </div>
        <div className="head-actions">
          {ready && (
            <button className="btn btn-ghost" onClick={() => setViewer({ highlight: null })}>
              View document
            </button>
          )}
          <button className="btn btn-ghost" onClick={onNewDocument} disabled={busy}>
            New document
          </button>
        </div>
      </header>

      {!ready && <IngestProgress doc={doc} onRetry={onNewDocument} />}

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <h3>Ask anything about this document</h3>
            <p className="muted">
              {ready
                ? "Answers are grounded in retrieved passages and cite them like [1]."
                : "You can type questions now. They're answered as soon as indexing finishes."}
            </p>
          </div>
        )}
        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="msg msg-user">
              <div className="bubble">{m.content}</div>
            </div>
          ) : (
            <AssistantMessage key={m.id} msg={m} onShowSource={showSource} />
          ),
        )}
        <div ref={endRef} />
      </div>

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <textarea
          ref={inputRef}
          rows={1}
          value={input}
          disabled={failed}
          placeholder={
            ready
              ? `Ask a question about ${doc.filename}…`
              : `Ask a question about ${doc.filename}… (it's sent when indexing finishes)`
          }
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send();
            }
          }}
        />
        {busy ? (
          <button type="button" className="btn btn-stop" onClick={() => abortRef.current?.abort()}>
            Stop
          </button>
        ) : (
          <button type="submit" className="btn btn-primary" disabled={!input.trim() || failed}>
            Send
          </button>
        )}
      </form>

      {viewer && (
        <DocumentViewer docId={doc.doc_id} filename={doc.filename} highlight={viewer.highlight} onClose={closeViewer} />
      )}
    </section>
  );
}
