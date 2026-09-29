"use client";

import { DocInfo, DocStatus } from "@/lib/api";

const STEPS: { key: DocStatus; label: string }[] = [
  { key: "converting", label: "Converting to Markdown" },
  { key: "chunking", label: "Chunking by headings" },
  { key: "contextualizing", label: "Adding context to each chunk" },
  { key: "embedding", label: "Embedding" },
  { key: "storing", label: "Storing in MongoDB" },
  { key: "indexing", label: "Syncing vector index" },
];

/** The model and chunk count shown after a step's label, if it has them. */
function detail(doc: DocInfo, step: DocStatus): string {
  const parts: string[] = [];
  if (step === "contextualizing" && doc.context_model) {
    parts.push(doc.context_model);
    if (doc.num_chunks) parts.push(`${doc.context_done ?? 0}/${doc.num_chunks} chunks`);
  } else if (step === "embedding") {
    parts.push(doc.embed_model);
    if (doc.num_chunks) parts.push(`${doc.num_chunks} chunks`);
  }
  return parts.map((p) => ` · ${p}`).join("");
}

/** Ingestion progress for a document that is still being indexed (or failed): the bar, and one line saying which
 * step is running (or failed). */
export default function IngestProgress({ doc, onRetry }: { doc: DocInfo; onRetry: () => void }) {
  const failed = doc.status === "failed";
  const steps = STEPS.filter((s) => s.key !== "contextualizing" || doc.contextual);
  const current = failed ? (doc.failed_stage ?? "converting") : doc.status;
  const index = steps.findIndex((s) => s.key === current);
  const step = index < 0 ? null : `step ${index + 1} of ${steps.length}: ${steps[index].label}`;
  const line = failed
    ? `Failed${step ? ` at ${step}` : ""}`
    : step
      ? `${step[0].toUpperCase()}${step.slice(1)}${detail(doc, current)}`
      : "Waiting to start";

  return (
    <div className="ingest" aria-live="polite">
      <p className="eyebrow">
        {failed ? "Indexing failed" : "Indexing the document · you can already type your questions"}
      </p>
      <div className={`bar ${failed ? "bar-failed" : ""}`}>
        <span style={{ width: `${Math.max(4, doc.progress)}%` }} />
      </div>
      <p className={`ingest-step ${failed ? "ingest-step-failed" : ""}`}>{line}</p>
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
