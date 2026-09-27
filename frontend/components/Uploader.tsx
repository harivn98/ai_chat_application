"use client";

import { useEffect, useRef, useState } from "react";
import { DocInfo, DocStatus, getDocument, uploadDocument } from "@/lib/api";

const ACCEPT = ".pdf,.txt,.md,.markdown";
const STEPS: { key: DocStatus; label: string }[] = [
  { key: "converting", label: "Converting to Markdown" },
  { key: "chunking", label: "Chunking by headings" },
  { key: "contextualizing", label: "Adding context to each chunk" },
  { key: "embedding", label: "Embedding with bge-small-en-v1.5" },
  { key: "storing", label: "Storing in MongoDB" },
  { key: "indexing", label: "Syncing vector index" },
];
const ORDER: DocStatus[] = [
  "queued",
  "converting",
  "chunking",
  "contextualizing",
  "embedding",
  "storing",
  "indexing",
  "ready",
];

function fmtSize(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

export default function Uploader({
  resumeDoc,
  onReady,
}: {
  resumeDoc: DocInfo | null;
  onReady: (doc: DocInfo) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [doc, setDoc] = useState<DocInfo | null>(resumeDoc);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Poll ingestion status until the document is indexed in the vector store
  useEffect(() => {
    if (!doc || doc.status === "ready" || doc.status === "failed") return;
    let alive = true;
    const t = setInterval(async () => {
      try {
        const d = await getDocument(doc.doc_id);
        if (!alive) return;
        setDoc(d);
        if (d.status === "ready") onReady(d);
      } catch (e) {
        if (alive) setError((e as Error).message);
      }
    }, 800);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [doc, onReady]);

  function pick(f: File | undefined | null) {
    setError(null);
    if (!f) return;
    const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    if (![".pdf", ".txt", ".md", ".markdown"].includes(ext)) {
      setError("Please choose a PDF, TXT or MD file.");
      return;
    }
    setFile(f);
  }

  async function submit() {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      setDoc(await uploadDocument(file));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }

  function reset() {
    setDoc(null);
    setFile(null);
    setError(null);
  }

  // ---- processing view
  if (doc) {
    const failed = doc.status === "failed";
    const current = Math.max(1, ORDER.indexOf(failed ? doc.failed_stage ?? "converting" : doc.status));
    return (
      <section className="card upload-card" aria-live="polite">
        <p className="eyebrow">{failed ? "Ingestion failed" : "Preparing your document"}</p>
        <h2 className="file-title">{doc.filename}</h2>
        <div className={`bar ${failed ? "bar-failed" : ""}`}>
          <span style={{ width: `${Math.max(4, doc.progress)}%` }} />
        </div>
        <ol className="steps">
          {STEPS.filter((s) => s.key !== "contextualizing" || doc.contextual).map((s) => {
            const idx = ORDER.indexOf(s.key);
            const state =
              idx < current ? "done" : idx === current ? (failed ? "error" : "active") : "todo";
            return (
              <li key={s.key} className={`step step-${state}`}>
                <span className="dot" aria-hidden />
                {s.label}
                {s.key === "contextualizing" && doc.context_model ? (
                  <span className="muted">
                    {" "}
                    · {doc.context_model}
                    {doc.num_chunks ? ` · ${doc.context_done ?? 0}/${doc.num_chunks} chunks` : ""}
                  </span>
                ) : null}
                {s.key === "embedding" && doc.num_chunks ? (
                  <span className="muted"> · {doc.num_chunks} chunks</span>
                ) : null}
              </li>
            );
          })}
        </ol>
        {failed && (
          <>
            <p className="error">{doc.error}</p>
            <button className="btn btn-primary" onClick={reset}>
              Try another file
            </button>
          </>
        )}
        {error && !failed && <p className="error">{error}</p>}
      </section>
    );
  }

  // ---- upload view
  return (
    <section className="card upload-card">
      <p className="eyebrow">Step 1</p>
      <h2>Upload a document to start chatting</h2>
      <p className="muted">
        PDF, TXT and Markdown are converted to Markdown, chunked, embedded and indexed. The chat opens as soon as the
        document is searchable.
      </p>

      <div
        className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          pick(e.dataTransfer.files?.[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => pick(e.target.files?.[0])}
        />
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" aria-hidden>
          <path d="M12 16V4m0 0l-4.5 4.5M12 4l4.5 4.5M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        {file ? (
          <div>
            <strong>{file.name}</strong>
            <div className="muted">{fmtSize(file.size)} · click to change</div>
          </div>
        ) : (
          <div>
            <strong>Drop a file here</strong> or click to browse
            <div className="muted">.pdf · .txt · .md</div>
          </div>
        )}
      </div>

      {error && <p className="error">{error}</p>}

      <button className="btn btn-primary btn-wide" disabled={!file || uploading} onClick={submit}>
        {uploading ? "Uploading…" : "Upload & index"}
      </button>
    </section>
  );
}
