"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import AssistantMessage from "@/components/AssistantMessage";
import Composer from "@/components/Composer";
import DocumentViewer, { Highlight } from "@/components/DocumentViewer";
import IngestProgress from "@/components/IngestProgress";
import ModelLoading from "@/components/ModelLoading";
import { useChat } from "@/hooks/useChat";
import { useModelStatus } from "@/hooks/useModelStatus";
import { DocInfo, Source } from "@/lib/api";

export default function ChatWindow({ doc, onNewDocument }: { doc: DocInfo; onNewDocument: () => void }) {
  const ready = doc.status === "ready";
  const failed = doc.status === "failed";
  const { messages, busy, ask, stop } = useChat(doc);
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
          {ready && (
            <button className="btn btn-ghost" onClick={() => setViewer({ highlight: null })}>
              View document
            </button>
          )}
          <button className="btn btn-ghost" onClick={onNewDocument} disabled={busy}>
            New document
          </button>
        </div>
      </header>

      {!ready && <IngestProgress doc={doc} onRetry={onNewDocument} />}
      {model?.state === "loading" && <ModelLoading status={model} />}

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
        disabled={failed}
        busy={busy}
        onSend={ask}
        onStop={stop}
      />

      {viewer && (
        <DocumentViewer docId={doc.doc_id} filename={doc.filename} highlight={viewer.highlight} onClose={closeViewer} />
      )}
    </section>
  );
}
