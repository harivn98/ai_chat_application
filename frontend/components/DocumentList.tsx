"use client";

import { useEffect, useState } from "react";
import { deleteDocument, DocInfo, listDocuments, ListedDocument } from "@/lib/api";

const POLL_MS = 2000; // while a document in the list is still being indexed

function formatDate(iso: string | null) {
  if (!iso) return "";
  return new Date(iso).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function status(d: ListedDocument) {
  if (d.status === "ready") return `${d.chat_count} ${d.chat_count === 1 ? "chat" : "chats"}`;
  if (d.status === "failed") return "Indexing failed";
  return `Indexing… ${d.progress}%`;
}

/** The documents uploaded earlier, newest first: clicking one reopens it with its chats; the bin deletes it (and
 * its chats). `refreshKey` reloads the list (after New session). */
export default function DocumentList({ onOpen, refreshKey }: { onOpen: (doc: DocInfo) => void; refreshKey: number }) {
  const [docs, setDocs] = useState<ListedDocument[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const indexing = !!docs?.some((d) => d.status !== "ready" && d.status !== "failed");

  useEffect(() => {
    let alive = true;
    const load = () =>
      listDocuments()
        .then((ds) => alive && (setDocs(ds), setError(null)))
        .catch((e) => alive && setError((e as Error).message));
    load();
    const timer = indexing ? setInterval(load, POLL_MS) : undefined;
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [indexing, refreshKey]);

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

  if (!docs?.length && !error) return null;
  return (
    <section className="card doc-list">
      <p className="eyebrow">Your documents</p>
      {error && <p className="error">{error}</p>}
      <ul>
        {docs?.map((d) => (
          <li key={d.doc_id} className={d.status === "failed" ? "failed" : ""}>
            <button className="doc-item" onClick={() => onOpen(d)} title={`Open ${d.filename} and its chats`}>
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
            <button className="chat-delete" onClick={() => remove(d)} aria-label={`Delete ${d.filename}`} title="Delete document and its chats">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden>
                <path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 002 2h6a2 2 0 002-2l1-12M9 7V4h6v3" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
