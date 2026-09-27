"use client";

import { DocInfo, DocStatus } from "@/lib/api";

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

/** Ingestion progress for a document that is still being indexed (or failed). */
export default function IngestProgress({ doc, onRetry }: { doc: DocInfo; onRetry: () => void }) {
  const failed = doc.status === "failed";
  const current = Math.max(1, ORDER.indexOf(failed ? doc.failed_stage ?? "converting" : doc.status));
  return (
    <div className="ingest" aria-live="polite">
      <p className="eyebrow">
        {failed ? "Indexing failed" : "Indexing the document · you can already type your questions"}
      </p>
      <div className={`bar ${failed ? "bar-failed" : ""}`}>
        <span style={{ width: `${Math.max(4, doc.progress)}%` }} />
      </div>
      <ol className="steps steps-compact">
        {STEPS.filter((s) => s.key !== "contextualizing" || doc.contextual).map((s) => {
          const idx = ORDER.indexOf(s.key);
          const state = idx < current ? "done" : idx === current ? (failed ? "error" : "active") : "todo";
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
          <button className="btn btn-primary" onClick={onRetry}>
            Try another file
          </button>
        </>
      )}
    </div>
  );
}
