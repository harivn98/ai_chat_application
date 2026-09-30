"use client";

import { useState } from "react";
import ChatWindow from "@/components/ChatWindow";
import DocumentList from "@/components/DocumentList";
import Logo from "@/components/Logo";
import Uploader from "@/components/Uploader";
import { useCurrentDocument } from "@/hooks/useCurrentDocument";
import { deleteAllDocuments, DocInfo } from "@/lib/api";
import { openChatId } from "@/lib/remembered";

export default function Home() {
  const { doc, restored, open, close } = useCurrentDocument();
  const [chatLocked, setChatLocked] = useState(false); // an answer is streaming or questions wait to be sent
  const [clearing, setClearing] = useState(false);
  const [history, setHistory] = useState(false); // the Chat history page instead of the upload page
  const [error, setError] = useState<string | null>(null);

  /** Open a document, at `chatId` if given (else at the chat this tab last had open, or its most recent). */
  function openDocument(d: DocInfo, chatId?: string) {
    if (chatId) openChatId.set(d.doc_id, chatId);
    setHistory(false);
    open(d);
  }

  const onHome = restored && !doc && !history; // the upload page is showing
  const locked = !!doc && chatLocked; // don't leave the chat while an answer streams or questions wait

  function goHome() {
    close();
    setHistory(false);
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
    <main className={doc ? "shell shell-chat" : "shell"}>
      <header className="topbar">
        <div>
          <div className="brand">
            <Logo />
            AI Chat Application <span className="brand-short">ACAP</span>
          </div>
          {restored && <p className="page-label">{doc ? "Current chat" : history ? "Chat history" : "Home"}</p>}
        </div>
        <div className="topbar-right">
          <button
            className="btn btn-ghost"
            onClick={goHome}
            disabled={onHome || locked}
            title="The upload page. Documents and their chats stay saved under Chat history."
          >
            Home
          </button>
          <button
            className="btn btn-ghost"
            onClick={showHistory}
            disabled={(!doc && history) || locked}
            title="Your earlier documents and their chats"
          >
            Chat history
          </button>
          <button
            className="btn btn-ghost"
            onClick={newSession}
            disabled={clearing || locked}
            title="Delete all documents and chats"
          >
            {clearing ? "Clearing…" : "New session"}
          </button>
        </div>
      </header>

      {error && <p className="error">{error}</p>}

      {!restored ? null : doc ? (
        <ChatWindow key={doc.doc_id} doc={doc} onHome={close} onLockedChange={setChatLocked} />
      ) : history ? (
        <DocumentList onOpen={openDocument} onHome={() => setHistory(false)} />
      ) : (
        <Uploader onUploaded={openDocument} />
      )}
    </main>
  );
}
