"use client";

import { ModelStatus } from "@/lib/api";

/** Private mode: the answering model is being loaded into memory. Ollama doesn't report how far a load is, so
 * the bar compares the time so far with the model's last load (it stops at 95% until the load ends); before a
 * load was ever measured it only shows that loading is under way. */
export default function ModelLoading({ status }: { status: ModelStatus }) {
  const elapsed = Math.round(status.elapsed_s ?? 0);
  const expected = status.expected_s ? Math.round(status.expected_s) : null;
  const pct = expected ? Math.min(95, (100 * elapsed) / expected) : null;

  let timing = `${elapsed} s`;
  if (expected) {
    timing += elapsed <= expected ? ` of about ${expected} s` : ` · taking longer than the last load (${expected} s)`;
  }

  return (
    <div className="ingest model-loading">
      <p className="eyebrow" aria-live="polite">
        Loading the answering model · {status.model}
      </p>
      <div className={`bar ${pct === null ? "bar-indeterminate" : ""}`}>
        <span style={pct === null ? undefined : { width: `${Math.max(4, pct)}%` }} />
      </div>
      <p className="muted small">{timing}. Questions are answered as soon as it is in memory.</p>
    </div>
  );
}
