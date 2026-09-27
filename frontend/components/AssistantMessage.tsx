"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { citedPassage, prepareAnswerMarkdown } from "@/lib/answerMarkdown";
import { Source } from "@/lib/api";

const remarkPlugins = [remarkGfm, remarkMath];
// Math is rendered with KaTeX; fenced code with a language (```python) is syntax highlighted
const rehypePlugins = [rehypeKatex, rehypeHighlight];

type Reply = {
  id: string;
  content: string;
  sources?: Source[];
  streaming?: boolean;
  queued?: boolean; // asked while the document was still being indexed; sent once it is ready
  error?: string;
};

type ShowSource = (s: Source) => void;

function SourceList({
  sources,
  active,
  id,
  onShow,
}: {
  sources: Source[];
  active: number | null;
  id: string;
  onShow: ShowSource;
}) {
  return (
    <ol className="sources">
      {sources.map((s) => (
        <li key={s.id} id={`${id}-src-${s.id}`} className={active === s.id ? "source active" : "source"}>
          <div className="source-head">
            <span className="source-num">{s.id}</span>
            <span className="source-section">{s.section || "Untitled section"}</span>
            <span className="ranks">
              {s.bm25_rank ? <span className="rank" title="BM25 rank">BM25 #{s.bm25_rank}</span> : null}
              {s.vector_rank ? <span className="rank" title="Vector rank">Vector #{s.vector_rank}</span> : null}
              {s.fused_rank ? (
                <span className="rank" title="Rank after BM25 + vector fusion, before the reranker">
                  Fused #{s.fused_rank}
                </span>
              ) : null}
            </span>
            {s.start != null && (
              <button className="source-show" onClick={() => onShow(s)}>
                Show in document
              </button>
            )}
          </div>
          <p className="source-text">{s.text}</p>
        </li>
      ))}
    </ol>
  );
}

/** An answer with its citation chips and the passages it was generated from. */
export default function AssistantMessage({ msg, onShowSource }: { msg: Reply; onShowSource: ShowSource }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState<number | null>(null);

  // A citation opens the document at the cited passage; uploads indexed before positions were stored
  // fall back to the passage list
  function cite(n: number) {
    const source = msg.sources?.find((s) => s.id === n);
    if (source?.start != null) return onShowSource(source);
    setOpen(true);
    setActive(n);
    requestAnimationFrame(() =>
      document.getElementById(`${msg.id}-src-${n}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" }),
    );
  }

  const thinking = msg.streaming && !msg.content;
  return (
    <div className="msg msg-assistant">
      <div className="avatar" aria-hidden>
        AI
      </div>
      <div className="msg-body">
        {msg.queued ? (
          <p className="queued muted">Waiting for the document to finish indexing. This question is sent automatically.</p>
        ) : thinking ? (
          <div className="typing" aria-label="Generating">
            <span />
            <span />
            <span />
          </div>
        ) : (
          <div className={`markdown ${msg.streaming ? "streaming" : ""}`}>
            <ReactMarkdown
              remarkPlugins={remarkPlugins}
              rehypePlugins={rehypePlugins}
              components={{
                a: ({ href, children }) => {
                  const n = citedPassage(href);
                  if (n !== null) {
                    return (
                      <button className="cite" onClick={() => cite(n)} title={`Show source ${n}`}>
                        {n}
                      </button>
                    );
                  }
                  return (
                    <a href={href} target="_blank" rel="noreferrer">
                      {children}
                    </a>
                  );
                },
              }}
            >
              {prepareAnswerMarkdown(msg.content)}
            </ReactMarkdown>
          </div>
        )}
        {msg.error && <p className="error">{msg.error}</p>}
        {msg.sources && msg.sources.length > 0 && (
          <div className="sources-wrap">
            <button className="sources-toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
              {open ? "Hide" : "Show"} {msg.sources.length} retrieved passages
            </button>
            {open && <SourceList sources={msg.sources} active={active} id={msg.id} onShow={onShowSource} />}
          </div>
        )}
      </div>
    </div>
  );
}
