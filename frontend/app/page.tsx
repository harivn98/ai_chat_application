"use client";

import { useState } from "react";
import ChatWindow from "@/components/ChatWindow";
import DocumentList from "@/components/DocumentList";
import Uploader from "@/components/Uploader";
import { useCurrentDocument } from "@/hooks/useCurrentDocument";
import { deleteAllDocuments } from "@/lib/api";

export default function Home() {
  const { doc, restored, open, close } = useCurrentDocument();
  const [chatLocked, setChatLocked] = useState(false); // an answer is streaming or questions wait to be sent
  const [clearing, setClearing] = useState(false);
  const [listKey, setListKey] = useState(0); // bumped to reload the documents list
  const [error, setError] = useState<string | null>(null);

  /** New session: delete every document with its chats, then show the empty upload page. */
  async function newSession() {
    if (!confirm("Start a new session? All documents and all their chats will be deleted. This can't be undone.")) return;
    setClearing(true);
    setError(null);
    try {
      await deleteAllDocuments();
      close();
      setListKey((k) => k + 1);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setClearing(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden />
          AI_chat_application
        </div>
        <div className="topbar-right">
          {doc && (
            <div className="pipeline">
              <span className={`mode-tag ${doc.mode === "private" ? "" : "mode-cloud"}`}>{doc.mode_label}</span>
              <span>{doc.embed_model}</span>
              <span>BM25 + vector · RRF</span>
              <span>{doc.llm_model}</span>
            </div>
          )}
          <button
            className="btn btn-ghost"
            onClick={newSession}
            disabled={clearing || (!!doc && chatLocked)}
            title="Delete all documents and chats"
          >
            {clearing ? "Clearing…" : "New session"}
          </button>
        </div>
      </header>

      {error && <p className="error">{error}</p>}

      {!restored ? null : doc ? (
        <ChatWindow key={doc.doc_id} doc={doc} onNewDocument={close} onLockedChange={setChatLocked} />
      ) : (
        <>
          <Uploader onUploaded={open} />
          <DocumentList onOpen={open} refreshKey={listKey} />
        </>
      )}
    </main>
  );
}
