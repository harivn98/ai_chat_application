"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AssistantMessage from "@/components/AssistantMessage";
import ChatList from "@/components/ChatList";
import Composer from "@/components/Composer";
import DocumentViewer, { Highlight } from "@/components/DocumentViewer";
import IngestProgress from "@/components/IngestProgress";
import ModelLoading from "@/components/ModelLoading";
import { useChat } from "@/hooks/useChat";
import { useChats } from "@/hooks/useChats";
import { useModelStatus } from "@/hooks/useModelStatus";
import { ChatInfo, DocInfo, Source } from "@/lib/api";

/** The chats about one document. `onNewDocument` goes back to the upload page (the document stays saved);
 * `onLockedChange` says whether an answer is streaming or questions wait to be sent. */
export default function ChatWindow({
  doc,
  onNewDocument,
  onLockedChange,
}: {
  doc: DocInfo;
  onNewDocument: () => void;
  onLockedChange: (locked: boolean) => void;
}) {
  const ready = doc.status === "ready";
  const failed = doc.status === "failed";
  const chats = useChats(doc.doc_id);
  const { messages, loading, loadError, busy, locked, ask, stop } = useChat(doc, chats.currentId, chats.saved);
  const model = useModelStatus(doc.mode === "private", busy);
  const endRef = useRef<HTMLDivElement>(null);
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
    onLockedChange(locked);
  }, [locked, onLockedChange]);
  useEffect(() => () => onLockedChange(false), [onLockedChange]);

  function deleteChat(chat: ChatInfo) {
    // an empty chat has nothing worth keeping, so no confirmation
    const ok = !chat.message_count || confirm(`Delete the chat "${chat.title}"? Its questions and answers are deleted too.`);
    if (ok) chats.remove(chat.chat_id);
  }

  const indexStatus = ready
    ? `Indexed · ${doc.num_chunks} chunks · hybrid BM25 + vector retrieval`
    : failed
      ? "Indexing failed"
      : `Indexing… ${doc.progress}%`;

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
              {doc.mode_label} · {indexStatus}
            </div>
          </div>
        </div>
        <div className="head-actions">
          <button
            className="btn btn-ghost"
            onClick={chats.newChat}
            disabled={locked || !chats.canCreate}
            title={
              chats.current && !chats.current.message_count
                ? "This chat is still empty"
                : chats.canCreate
                  ? "Start a new chat about the same document"
                  : `A document can have ${chats.maxChats} chats. Delete one to start a new chat.`
            }
          >
            New chat
          </button>
          {ready && (
            <button className="btn btn-ghost" onClick={() => setViewer({ highlight: null })}>
              View document
            </button>
          )}
          <button
            className="btn btn-ghost"
            onClick={onNewDocument}
            disabled={locked}
            title="Upload another document. This one and its chats stay saved under Previous docs & chats."
          >
            New document
          </button>
        </div>
      </header>

      {!ready && <IngestProgress doc={doc} onRetry={onNewDocument} />}
      {model?.state === "loading" && <ModelLoading status={model} />}

      <div className="chat-body">
        <ChatList
          chats={chats.chats}
          currentId={chats.currentId}
          maxChats={chats.maxChats}
          locked={locked}
          onOpen={chats.open}
          onDelete={deleteChat}
        />
        <div className="conversation">
          {(chats.error || loadError) && <p className="error chat-error">{chats.error || loadError}</p>}
          <div className="messages">
            {loading && !loadError && <p className="muted empty">Loading the chat…</p>}
            {!loading && messages.length === 0 && (
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

          <Composer
            placeholder={
              ready
                ? `Ask a question about ${doc.filename}…`
                : `Ask a question about ${doc.filename}… (it's sent when indexing finishes)`
            }
            disabled={failed || loading}
            busy={busy}
            onSend={ask}
            onStop={stop}
          />
        </div>
      </div>

      {viewer && (
        <DocumentViewer docId={doc.doc_id} filename={doc.filename} highlight={viewer.highlight} onClose={closeViewer} />
      )}
    </section>
  );
}
