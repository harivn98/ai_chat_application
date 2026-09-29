"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { ChatMessage } from "@/hooks/useChat";
import { citedPassage, prepareAnswerMarkdown } from "@/lib/answerMarkdown";
import { Source, Verdict } from "@/lib/api";

const remarkPlugins = [remarkGfm, remarkMath];
// Math is rendered with KaTeX; fenced code with a language (```python) is syntax highlighted
const rehypePlugins = [rehypeKatex, rehypeHighlight];

const VERDICT_TEXT: Record<Verdict, string> = {
  all: "Pre-judge: the passages hold all the information asked for",
  partial: "Pre-judge: the passages hold only part of the information asked for",
  none: "Pre-judge: the passages don't hold the information asked for",
};

type ShowSource = (s: Source) => void;

const sourceElementId = (messageId: string, sourceId: number) => `${messageId}-src-${sourceId}`;

function SourceList({
  sources,
  active,
  messageId,
  onShow,
}: {
  sources: Source[];
  active: number | null;
  messageId: string;
  onShow: ShowSource;
}) {
  return (
    <ol className="sources">
      {sources.map((s) => (
        <li
          key={s.id}
          id={sourceElementId(messageId, s.id)}
          className={active === s.id ? "source active" : "source"}
        >
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
export default function AssistantMessage({ msg, onShowSource }: { msg: ChatMessage; onShowSource: ShowSource }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState<number | null>(null);

  // A citation opens the document at the cited passage; uploads indexed before positions were stored
  // fall back to the passage list
  function openCitation(n: number) {
    const source = msg.sources?.find((s) => s.id === n);
    if (source?.start != null) return onShowSource(source);
    setOpen(true);
    setActive(n);
    requestAnimationFrame(() =>
      document.getElementById(sourceElementId(msg.id, n))?.scrollIntoView({ behavior: "smooth", block: "nearest" }),
    );
  }

  const awaitingFirstToken = msg.streaming && !msg.content;
  return (
    <div className="msg msg-assistant">
      <div className="avatar" aria-hidden>
        AI
      </div>
      <div className="msg-body">
        {msg.queued ? (
          <p className="queued muted">Waiting for the document to finish indexing. This question is sent automatically.</p>
        ) : awaitingFirstToken ? (
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
                      <button className="cite" onClick={() => openCitation(n)} title={`Show source ${n}`}>
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
        {msg.verdict && <p className={`verdict verdict-${msg.verdict}`}>{VERDICT_TEXT[msg.verdict]}</p>}
        {msg.sources && msg.sources.length > 0 && (
          <div className="sources-wrap">
            <button className="sources-toggle" onClick={() => setOpen((o) => !o)} aria-expanded={open}>
              {open ? "Hide" : "Show"} {msg.sources.length} retrieved passages
            </button>
            {open && <SourceList sources={msg.sources} active={active} messageId={msg.id} onShow={onShowSource} />}
          </div>
        )}
      </div>
    </div>
  );
}
