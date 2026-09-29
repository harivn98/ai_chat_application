"use client";

import { ChatInfo } from "@/lib/api";

function when(iso: string) {
  return new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

/** The chats about the document, most recently used first: open one, or delete one. While a question is being
 * answered (`locked`), the open chat stays open. */
export default function ChatList({
  chats,
  currentId,
  maxChats,
  locked,
  onOpen,
  onDelete,
}: {
  chats: ChatInfo[];
  currentId: string | null;
  maxChats: number;
  locked: boolean;
  onOpen: (chatId: string) => void;
  onDelete: (chat: ChatInfo) => void;
}) {
  return (
    <aside className="chat-list" aria-label="Chats about this document">
      <p className="eyebrow">
        Chats · {chats.length}/{maxChats}
      </p>
      <ul>
        {chats.map((c) => {
          const title = c.title || "New chat";
          const active = c.chat_id === currentId;
          return (
            <li key={c.chat_id} className={active ? "active" : undefined}>
              <button
                className="chat-item"
                onClick={() => onOpen(c.chat_id)}
                disabled={locked && !active}
                aria-current={active ? "true" : undefined}
                title={title}
              >
                <span className="chat-title">{title}</span>
                <span className="chat-when">
                  {c.message_count ? `${c.message_count / 2} question${c.message_count === 2 ? "" : "s"} · ` : ""}
                  {when(c.updated_at)}
                </span>
              </button>
              <button
                className="chat-delete"
                onClick={() => onDelete(c)}
                disabled={locked}
                aria-label={`Delete chat: ${title}`}
                title="Delete chat"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden>
                  <path d="M4 7h16M10 11v6m4-6v6M6 7l1 12a2 2 0 002 2h6a2 2 0 002-2l1-12M9 7V4h6v3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
