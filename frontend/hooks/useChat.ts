"use client";

import { useEffect, useRef, useState } from "react";
import { ChatInfo, ChatTurn, DocInfo, getChat, saveMessages, Source, streamChat, Verdict } from "@/lib/api";

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

/** One chat about a document: its saved messages are loaded when it opens, every question is queued, queued
 * questions are sent one at a time once the document is ready, each answer streams into its message, and a
 * question with its answer is saved to the chat once the answer has ended (also when stopped or failed).
 * `onSaved` gets the chat's updated summary. */
export function useChat(doc: DocInfo, chatId: string | null, onSaved: (chat: ChatInfo) => void) {
  const ready = doc.status === "ready";
  const failed = doc.status === "failed";
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loadedId, setLoadedId] = useState<string | null>(null); // the chat `messages` belong to
  const [loadError, setLoadError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const sendingRef = useRef(false); // guards against sending the same queued question twice
  const loading = !!chatId && chatId !== loadedId;

  const update = (id: string, patch: (m: ChatMessage) => Partial<ChatMessage>) =>
    setMessages((ms) => ms.map((m) => (m.id === id ? { ...m, ...patch(m) } : m)));

  // Load the chat's saved messages when it opens
  useEffect(() => {
    if (!chatId) return;
    let alive = true;
    getChat(chatId)
      .then((chat) => {
        if (!alive) return;
        setMessages(chat.messages.map((m) => ({ ...m, id: uid() })));
        setLoadError(null);
        setLoadedId(chatId);
      })
      .catch((e) => alive && setLoadError((e as Error).message));
    return () => {
      alive = false;
    };
  }, [chatId]);

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
    if (!ready || busy || sendingRef.current || loading || !chatId) return;
    const i = messages.findIndex((m) => m.queued);
    if (i < 0) return;
    const history = messages
      .slice(0, i - 1) // everything before this question's user message
      .filter((m) => !m.error && !m.queued && m.content)
      .map((m) => ({ role: m.role, content: m.content }));
    streamAnswer(chatId, messages[i].id, messages[i].question ?? "", history);
  }, [ready, busy, loading, chatId, messages]); // eslint-disable-line react-hooks/exhaustive-deps

  // Indexing failed: questions that were waiting can't be answered
  useEffect(() => {
    if (!failed) return;
    setMessages((ms) =>
      ms.map((m) =>
        m.queued ? { ...m, queued: false, error: "The document could not be indexed, so this question was not sent." } : m,
      ),
    );
  }, [failed]);

  async function streamAnswer(chat: string, replyId: string, question: string, history: ChatTurn[]) {
    sendingRef.current = true;
    update(replyId, () => ({ queued: false, streaming: true }));
    setBusy(true);

    // The answer as it ends up, to save it to the chat
    let content = "";
    let sources: Source[] | undefined;
    let verdict: Verdict | undefined;
    let error: string | undefined;

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await streamChat(
        { doc_id: doc.doc_id, question, history },
        (e) => {
          if (e.type === "sources") {
            sources = e.sources;
            update(replyId, () => ({ sources: e.sources }));
          } else if (e.type === "prejudge") {
            verdict = e.verdict;
            update(replyId, () => ({ verdict: e.verdict }));
          } else if (e.type === "token") {
            content += e.content;
            update(replyId, (m) => ({ content: m.content + e.content }));
          } else if (e.type === "error") {
            error = e.message;
            update(replyId, () => ({ error: e.message }));
          }
        },
        ctrl.signal,
      );
    } catch (err) {
      if ((err as Error).name !== "AbortError") {
        error = (err as Error).message;
        update(replyId, () => ({ error }));
      }
    } finally {
      content = content || (error ? "" : "_Stopped._");
      update(replyId, () => ({ streaming: false, content }));
      sendingRef.current = false;
      setBusy(false);
      abortRef.current = null;
    }
    // If saving fails, the answer stays on screen; it is only missing when the chat is reopened
    saveMessages(chat, [
      { role: "user", content: question },
      { role: "assistant", content, sources, verdict, error },
    ])
      .then(onSaved)
      .catch(() => {});
  }

  return {
    messages: loading ? [] : messages,
    loading,
    loadError,
    busy,
    // a question is being answered or waits to be sent: the chat can't be switched or deleted meanwhile
    locked: busy || messages.some((m) => m.queued),
    ask,
    stop,
  };
}
