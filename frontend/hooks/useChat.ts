"use client";

import { useEffect, useRef, useState } from "react";
import { ChatTurn, DocInfo, Source, streamChat, Verdict } from "@/lib/api";

const uid = () => Math.random().toString(36).slice(2, 10);

/** A question asked while the document is still being indexed waits in a `queued` assistant message (with the
 * `question` it answers) and is sent once the document is ready. */
export type ChatMessage = ChatTurn & {
  id: string;
  sources?: Source[];
  verdict?: Verdict; // set when the document's mode pre-judges
  streaming?: boolean;
  error?: string;
  queued?: boolean;
  question?: string;
};

/** The conversation about one document. Every question is queued; queued questions are sent one at a time once
 * the document is ready, and each answer streams into its message. */
export function useChat(doc: DocInfo) {
  const ready = doc.status === "ready";
  const failed = doc.status === "failed";
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const sendingRef = useRef(false); // guards against sending the same queued question twice

  const update = (id: string, patch: (m: ChatMessage) => Partial<ChatMessage>) =>
    setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, ...patch(m) } : m)));

  function ask(question: string) {
    setMessages((ms) => [
      ...ms,
      { id: uid(), role: "user", content: question },
      { id: uid(), role: "assistant", content: "", queued: true, question },
    ]);
  }

  /** Stop the answer that is streaming. */
  function stop() {
    abortRef.current?.abort();
  }

  useEffect(() => {
    if (!ready || busy || sendingRef.current) return;
    const i = messages.findIndex((m) => m.queued);
    if (i < 0) return;
    const history = messages
      .slice(0, i - 1) // everything before this question's user message
      .filter((m) => !m.error && !m.queued && m.content)
      .map((m) => ({ role: m.role, content: m.content }));
    streamAnswer(messages[i].id, messages[i].question ?? "", history);
  }, [ready, busy, messages]); // eslint-disable-line react-hooks/exhaustive-deps

  // Indexing failed: questions that were waiting can't be answered
  useEffect(() => {
    if (!failed) return;
    setMessages((ms) =>
      ms.map((m) =>
        m.queued ? { ...m, queued: false, error: "The document could not be indexed, so this question was not sent." } : m,
      ),
    );
  }, [failed]);

  async function streamAnswer(replyId: string, question: string, history: ChatTurn[]) {
    sendingRef.current = true;
    update(replyId, () => ({ queued: false, streaming: true }));
    setBusy(true);

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await streamChat(
        { doc_id: doc.doc_id, question, history },
        (e) => {
          if (e.type === "sources") update(replyId, () => ({ sources: e.sources }));
          else if (e.type === "prejudge") update(replyId, () => ({ verdict: e.verdict }));
          else if (e.type === "token") update(replyId, (m) => ({ content: m.content + e.content }));
          else if (e.type === "error") update(replyId, () => ({ error: e.message }));
        },
        ctrl.signal,
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") update(replyId, () => ({ error: (err as Error).message }));
    } finally {
      update(replyId, (m) => ({ streaming: false, content: m.content || (m.error ? "" : "_Stopped._") }));
      sendingRef.current = false;
      setBusy(false);
      abortRef.current = null;
    }
  }

  return { messages, busy, ask, stop };
}
