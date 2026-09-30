"use client";

import { useEffect, useState } from "react";
import { ChatInfo, deleteDocument, DocInfo, listChats, listDocuments, ListedDocument } from "@/lib/api";

const POLL_MS = 2000; // while a document in the list is still being indexed

function formatDate(iso: string | null) {
  if (!iso) return "";
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function status(d: ListedDocument) {
  if (d.status === "ready") return d.chat_count ? `${d.chat_count} ${d.chat_count === 1 ? "chat" : "chats"}` : "No chats yet";
  if (d.status === "failed") return "Indexing failed";
  return `Indexing… ${d.progress}%`;
}

/** A document's chats that have questions, most recently used first; clicking one opens it. */
function DocumentChats({ doc, onOpen }: { doc: ListedDocument; onOpen: (doc: DocInfo, chatId?: string) => void }) {
  const [chats, setChats] = useState<ChatInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listChats(doc.doc_id)
      .then((r) => setChats(r.chats.filter((c) => c.message_count > 0)))
      .catch((e) => setError((e as Error).message));
  }, [doc.doc_id]);

  if (error) return <p className="error doc-chats-note">{error}</p>;
  if (!chats) return <p className="muted small doc-chats-note">Loading chats…</p>;
  if (!chats.length) return <p className="muted small doc-chats-note">No questions asked yet.</p>;
  return (
    <ul className="doc-chats">
      {chats.map((c) => (
        <li key={c.chat_id}>
          <button className="chat-item" onClick={() => onOpen(doc, c.chat_id)} title={c.title}>
            <span className="chat-title">{c.title}</span>
            <span className="chat-when">
              {c.message_count / 2} question{c.message_count === 2 ? "" : "s"} · {formatDate(c.updated_at)}
            </span>
          </button>
        </li>
      ))}
    </ul>
  );
}

/** Chat history: the documents uploaded earlier, newest first. Clicking a document reopens it at its last chat;
 * the arrow shows its chats, each of which can be opened directly; the bin deletes the document with its chats. */
export default function DocumentList({
  onOpen,
  onHome,
}: {
  onOpen: (doc: DocInfo, chatId?: string) => void;
  onHome: () => void;
}) {
  const [docs, setDocs] = useState<ListedDocument[] | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const indexing = !!docs?.some((d) => d.status !== "ready" && d.status !== "failed");

  useEffect(() => {
    let alive = true;
    const load = () =>
      listDocuments()
        .then((ds) => {
          if (!alive) return;
          setDocs(ds);
          setError(null);
        })
        .catch((e) => alive && setError((e as Error).message));
    load();
    const timer = indexing ? setInterval(load, POLL_MS) : undefined;
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [indexing]);

  async function remove(d: ListedDocument) {
    const chats = d.chat_count ? ` and its ${d.chat_count === 1 ? "chat" : `${d.chat_count} chats`}` : "";
    if (!confirm(`Delete "${d.filename}"${chats}? This can't be undone.`)) return;
    try {
      await deleteDocument(d.doc_id);
      setDocs((ds) => ds?.filter((x) => x.doc_id !== d.doc_id) ?? null);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  return (
    <section className="card doc-list">
      <div className="doc-list-head">
        <div>
          <p className="eyebrow">Chat history</p>
          <h2>Your documents</h2>
        </div>
        <button className="btn btn-ghost" onClick={onHome}>
          Home
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {docs && !docs.length && <p className="muted">No documents yet. Upload one to start chatting.</p>}
      <ul>
        {docs?.map((d) => {
          const open = expanded === d.doc_id;
          return (
            <li key={d.doc_id} className={d.status === "failed" ? "failed" : ""}>
              <div className="doc-row">
                <button className="doc-item" onClick={() => onOpen(d)} title={`Open ${d.filename} at its last chat`}>
                  <span className="doc-icon" aria-hidden>
                    {d.filename.split(".").pop()?.toUpperCase()}
                  </span>
                  <span className="doc-item-text">
                    <span className="doc-item-name">{d.filename}</span>
                    <span className="muted small">
                      {d.mode_label} · {status(d)} · {formatDate(d.created_at)}
                    </span>
                  </span>
                </button>
                {d.status === "ready" && d.chat_count > 0 && (
                  <button
                    className="chat-delete doc-expand"
                    onClick={() => setExpanded(open ? null : d.doc_id)}
                    aria-expanded={open}
                    aria-label={`${open ? "Hide" : "Show"} the chats about ${d.filename}`}
                    title={open ? "Hide chats" : "Show chats"}
                  >
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden>
                      <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  </button>
                )}
                <button
                  className="chat-delete"
                  onClick={() => remove(d)}
                  aria-label={`Delete ${d.filename}`}
                  title="Delete document and its chats"
                >
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden>
                    <path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 002 2h6a2 2 0 002-2l1-12M9 7V4h6v3" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              </div>
              {open && <DocumentChats doc={d} onOpen={onOpen} />}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
