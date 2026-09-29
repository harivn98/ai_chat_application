"use client";

import { useCallback, useEffect, useState } from "react";
import { ChatInfo, createChat, deleteChat, listChats } from "@/lib/api";
import { openChatId } from "@/lib/remembered";

/** The chats about one document (up to `maxChats`, most recently used first) and the one that is open. A document
 * always has at least one chat: an empty one is created when it has none. The backend never keeps more than one
 * empty chat, so "new chat" while an empty chat exists opens that one. */
export function useChats(docId: string) {
  const [chats, setChats] = useState<ChatInfo[]>([]);
  const [maxChats, setMaxChats] = useState(10);
  const [currentId, setCurrentId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const open = useCallback(
    (chatId: string) => {
      setCurrentId(chatId);
      openChatId.set(docId, chatId);
    },
    [docId],
  );

  // Load the chats; reopen the one this tab had open (after a refresh), else the most recent
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const listed = await listChats(docId);
        const all = listed.chats.length ? listed.chats : [await createChat(docId)];
        if (!alive) return;
        setChats(all);
        setMaxChats(listed.max_chats);
        const saved = openChatId.get(docId);
        open(all.some((c) => c.chat_id === saved) ? saved! : all[0].chat_id);
      } catch (e) {
        if (alive) setError((e as Error).message);
      }
    })();
    return () => {
      alive = false;
    };
  }, [docId, open]);

  async function newChat() {
    setError(null);
    try {
      const chat = await createChat(docId);
      setChats((cs) => [chat, ...cs.filter((c) => c.chat_id !== chat.chat_id)]);
      open(chat.chat_id);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  /** Delete a chat. Deleting the open chat opens the most recent other one, or a new empty chat if none is left. */
  async function remove(chatId: string) {
    setError(null);
    try {
      await deleteChat(chatId);
      let rest = chats.filter((c) => c.chat_id !== chatId);
      if (!rest.length) rest = [await createChat(docId)];
      setChats(rest);
      if (chatId === currentId) open(rest[0].chat_id);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  /** A question and its answer were saved to a chat: it moves to the top, with its new title. */
  const saved = useCallback((chat: ChatInfo) => {
    setChats((cs) => [chat, ...cs.filter((c) => c.chat_id !== chat.chat_id)]);
  }, []);

  const current = chats.find((c) => c.chat_id === currentId) ?? null;
  // "New chat" only makes sense when it would open a different chat: the open one isn't empty, and there is
  // room for one more (or an empty chat exists to open)
  const canCreate =
    !!current && current.message_count > 0 && (chats.length < maxChats || chats.some((c) => !c.message_count));

  return { chats, maxChats, currentId, current, canCreate, error, open, newChat, remove, saved };
}
