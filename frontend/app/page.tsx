"use client";

import { useCallback, useEffect, useState } from "react";
import ChatWindow from "@/components/ChatWindow";
import Uploader from "@/components/Uploader";
import { DocInfo, deleteDocument, getDocument } from "@/lib/api";

const STORAGE_KEY = "AI_chat_application.doc_id";

export default function Home() {
  const [doc, setDoc] = useState<DocInfo | null>(null);
  const [resume, setResume] = useState<DocInfo | null>(null);
  const [loaded, setLoaded] = useState(false);

  // Re-open the last document after a page refresh
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
        if (d.status === "ready") setDoc(d);
        else if (d.status !== "failed") setResume(d);
      })
      .catch(() => {})
      .finally(() => setLoaded(true));
  }, []);

  const handleReady = useCallback((d: DocInfo) => {
    try {
      sessionStorage.setItem(STORAGE_KEY, d.doc_id);
    } catch {}
    setDoc(d);
  }, []);

  function newDocument() {
    if (doc && confirm("Start over with a new document? The current document will be removed from the index.")) {
      deleteDocument(doc.doc_id);
      try {
        sessionStorage.removeItem(STORAGE_KEY);
      } catch {}
      setResume(null);
      setDoc(null);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden />
          AI_chat_application
        </div>
        <div className="pipeline">
          <span>bge-small-en-v1.5</span>
          <span>BM25 + vector · RRF</span>
          <span>qwen3:8b</span>
        </div>
      </header>

      {!loaded ? null : doc ? (
        <ChatWindow key={doc.doc_id} doc={doc} onNewDocument={newDocument} />
      ) : (
        <Uploader resumeDoc={resume} onReady={handleReady} />
      )}
    </main>
  );
}
