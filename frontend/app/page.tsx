"use client";

import { useCallback, useEffect, useState } from "react";
import ChatWindow from "@/components/ChatWindow";
import Uploader from "@/components/Uploader";
import { DocInfo, deleteDocument, getDocument } from "@/lib/api";

const STORAGE_KEY = "AI_chat_application.doc_id";

export default function Home() {
  const [doc, setDoc] = useState<DocInfo | null>(null);
  const [loaded, setLoaded] = useState(false);

  // Re-open the last document after a page refresh (also while it is still being indexed)
  useEffect(() => {
    let id: string | null = null;
    try {
      id = sessionStorage.getItem(STORAGE_KEY);
    } catch {}
    if (!id) {
      setLoaded(true);
      return;
    }
    getDocument(id)
      .then((d) => {
        if (d.status !== "failed") setDoc(d);
      })
      .catch(() => {})
      .finally(() => setLoaded(true));
  }, []);

  // Poll ingestion status until the document is indexed (or failed); the chat is usable meanwhile
  const indexing = !!doc && doc.status !== "ready" && doc.status !== "failed";
  const docId = doc?.doc_id;
  useEffect(() => {
    if (!indexing || !docId) return;
    let alive = true;
    const t = setInterval(async () => {
      try {
        const d = await getDocument(docId);
        if (alive) setDoc(d);
      } catch {
        // transient network error: keep polling
      }
    }, 800);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [indexing, docId]);

  const handleUploaded = useCallback((d: DocInfo) => {
    try {
      sessionStorage.setItem(STORAGE_KEY, d.doc_id);
    } catch {}
    setDoc(d);
  }, []);

  function newDocument() {
    if (!doc) return;
    // a failed document has nothing worth keeping, so no confirmation
    const ok =
      doc.status === "failed" ||
      confirm("Start over with a new document? The current document will be removed from the index.");
    if (!ok) return;
    deleteDocument(doc.doc_id);
    try {
      sessionStorage.removeItem(STORAGE_KEY);
    } catch {}
    setDoc(null);
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden />
          AI_chat_application
        </div>
        {doc && (
          <div className="pipeline">
            <span className={`mode-tag ${doc.mode === "private" ? "" : "mode-cloud"}`}>{doc.mode_label}</span>
            <span>{doc.embed_model}</span>
            <span>BM25 + vector · RRF</span>
            <span>{doc.llm_model}</span>
          </div>
        )}
      </header>

      {!loaded ? null : doc ? (
        <ChatWindow key={doc.doc_id} doc={doc} onNewDocument={newDocument} />
      ) : (
        <Uploader onUploaded={handleUploaded} />
      )}
    </main>
  );
}
