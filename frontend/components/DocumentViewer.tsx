"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getMarkdown } from "@/lib/api";
import rehypeMarkRange from "@/lib/markRange";

export type Highlight = { start: number; end: number; label: string };

const markdownCache = new Map<string, Promise<string>>();

export default function DocumentViewer({
  docId,
  filename,
  highlight,
  onClose,
}: {
  docId: string;
  filename: string;
  highlight: Highlight | null;
  onClose: () => void;
}) {
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let alive = true;
    if (!markdownCache.has(docId)) markdownCache.set(docId, getMarkdown(docId));
    markdownCache
      .get(docId)!
      .then((md) => alive && setMarkdown(md))
      .catch((e: Error) => {
        markdownCache.delete(docId);
        if (alive) setError(e.message);
      });
    return () => {
      alive = false;
    };
  }, [docId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const start = highlight?.start;
  const end = highlight?.end;
  const rendered = useMemo(
    () =>
      markdown === null ? null : (
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={start != null && end != null ? [[rehypeMarkRange, { start, end }]] : []}
        >
          {markdown}
        </ReactMarkdown>
      ),
    [markdown, start, end],
  );

  // Bring the highlighted passage into view once the document is rendered
  useEffect(() => {
    const first = bodyRef.current?.querySelector(".hl-block, .hl");
    if (first) first.scrollIntoView({ block: "center" });
    else bodyRef.current?.scrollTo({ top: 0 });
  }, [rendered]);

  return (
    <div className="viewer-backdrop" onClick={onClose}>
      <aside
        className="viewer card"
        role="dialog"
        aria-modal="true"
        aria-label={`${filename} source document`}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="viewer-head">
          <div>
            <div className="doc-name">{filename}</div>
            <div className="muted small">
              {highlight ? `Highlighted: ${highlight.label}` : "Converted Markdown used for retrieval"}
            </div>
          </div>
          <button className="btn btn-ghost" onClick={onClose}>
            Close
          </button>
        </header>
        <div className="viewer-body markdown" ref={bodyRef}>
          {error ? <p className="error">{error}</p> : (rendered ?? <p className="muted">Loading document…</p>)}
        </div>
      </aside>
    </div>
  );
}
