"use client";

import ChatWindow from "@/components/ChatWindow";
import Uploader from "@/components/Uploader";
import { useCurrentDocument } from "@/hooks/useCurrentDocument";

export default function Home() {
  const { doc, restored, open, discard } = useCurrentDocument();

  function newDocument() {
    if (!doc) return;
    // a failed document has nothing worth keeping, so no confirmation
    const ok =
      doc.status === "failed" ||
      confirm("Start over with a new document? The current document and all its chats will be removed.");
    if (ok) discard();
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

      {!restored ? null : doc ? (
        <ChatWindow key={doc.doc_id} doc={doc} onNewDocument={newDocument} />
      ) : (
        <Uploader onUploaded={open} />
      )}
    </main>
  );
}
