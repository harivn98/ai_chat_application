"use client";

import { useState } from "react";
import ChatWindow from "@/components/ChatWindow";
import DocumentList from "@/components/DocumentList";
import Uploader from "@/components/Uploader";
import { useCurrentDocument } from "@/hooks/useCurrentDocument";
import { deleteAllDocuments, DocInfo } from "@/lib/api";
import { openChatId } from "@/lib/remembered";

export default function Home() {
  const { doc, restored, open, close } = useCurrentDocument();
  const [chatLocked, setChatLocked] = useState(false); // an answer is streaming or questions wait to be sent
  const [clearing, setClearing] = useState(false);
  const [history, setHistory] = useState(false); // the Previous docs & chats page instead of the upload page
  const [error, setError] = useState<string | null>(null);

  /** Open a document, at `chatId` if given (else at the chat this tab last had open, or its most recent). */
  function openDocument(d: DocInfo, chatId?: string) {
    if (chatId) openChatId.set(d.doc_id, chatId);
    setHistory(false);
    open(d);
  }

  function showHistory() {
    close();
    setHistory(true);
  }

  /** New session: delete every document with its chats, then show the empty upload page. */
  async function newSession() {
    if (!confirm("Start a new session? All documents and all their chats will be deleted. This can't be undone.")) return;
    setClearing(true);
    setError(null);
    try {
      await deleteAllDocuments();
      close();
      setHistory(false);
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
            onClick={showHistory}
            disabled={(!doc && history) || (!!doc && chatLocked)}
            title="Your earlier documents and their chats"
          >
            Previous docs & chats
          </button>
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
      ) : history ? (
        <DocumentList onOpen={openDocument} onUpload={() => setHistory(false)} />
      ) : (
        <Uploader onUploaded={openDocument} />
      )}
    </main>
  );
}
