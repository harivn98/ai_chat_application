"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import IngestProgress from "@/components/IngestProgress";
import { DocInfo, Message, Source, streamChat } from "@/lib/api";

const uid = () => Math.random().toString(36).slice(2, 10);

// A question asked while the document is still being indexed waits here and is sent once it is ready
type ChatMessage = Message & { queued?: boolean; question?: string };

// Turn "[3]" citations into in-page links we can render as clickable chips
const linkCitations = (md: string) => md.replace(/\[(\d{1,2})\](?!\()/g, "[[$1]](#cite-$1)");

function SourceList({ sources, active, id }: { sources: Source[]; active: number | null; id: string }) {
  return (
    <ol className="sources">
      {sources.map((s) => (
        <li key={s.id} id={`${id}-src-${s.id}`} className={active === s.id ? "source active" : "source"}>
          <div className="source-head">
            <span className="source-num">{s.id}</span>
            <span className="source-section">{s.section || "Untitled section"}</span>
            <span className="ranks">
              {s.bm25_rank ? <span className="rank" title="BM25 rank">BM25 #{s.bm25_rank}</span> : null}
              {s.vector_rank ? <span className="rank" title="Vector rank">Vector #{s.vector_rank}</span> : null}
              {s.fused_rank ? (
                <span className="rank" title="Rank after BM25 + vector fusion, before the reranker">
                  Fused #{s.fused_rank}
                </span>
              ) : null}
            </span>
          </div>
          <p className="source-text">{s.text}</p>
        </li>
      ))}
    </ol>
  );
}

function AssistantMessage({ msg }: { msg: ChatMessage }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState<number | null>(null);

  function cite(n: number) {
    setOpen(true);
    setActive(n);
    requestAnimationFrame(() =>
      document.getElementById(`${msg.id}-src-${n}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" }),
    );
  }

  const thinking = msg.streaming && !msg.content;
  return (
    <div className="msg msg-assistant">
      <div className="avatar" aria-hidden>
        AI
      </div>
      <div className="msg-body">
        {msg.queued ? (
          <p className="queued muted">Waiting for the document to finish indexing. This question is sent automatically.</p>
        ) : thinking ? (
          <div className="typing" aria-label="Generating">
            <span />
            <span />
            <span />
          </div>
        ) : (
          <div className={`markdown ${msg.streaming ? "streaming" : ""}`}>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                a: ({ href, children }) => {
                  if (href?.startsWith("#cite-")) {
                    const n = Number(href.slice(6));
                    return (
                      <button className="cite" onClick={() => cite(n)} title={`Show source ${n}`}>
                        {n}
                      </button>
                    );
                  }
                  return (
                    <a href={href} target="_blank" rel="noreferrer">
                      {children}
                    </a>
                  );
                },
              }}
            >
              {linkCitations(msg.content)}
            </ReactMarkdown>
          </div>
        )}
        {msg.error && <p className="error">{msg.error}</p>}
        {msg.sources && msg.sources.length > 0 && (
          <div className="sources-wrap">
            <button className="sources-toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
              {open ? "Hide" : "Show"} {msg.sources.length} retrieved passages
            </button>
            {open && <SourceList sources={msg.sources} active={active} id={msg.id} />}
          </div>
        )}
      </div>
    </div>
  );
}

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

  async function ask(botId: string, question: string, history: { role: string; content: string }[]) {
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
            <a className="btn btn-ghost" href={`/api/documents/${doc.doc_id}/markdown`} target="_blank" rel="noreferrer">
              View Markdown
            </a>
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
            <AssistantMessage key={m.id} msg={m} />
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
    </section>
  );
}
