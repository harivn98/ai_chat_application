// What this browser remembers between page loads. Storage can be unavailable (e.g. blocked by the browser), so
// every access is guarded: the app then works the same, it just doesn't remember.
import { ModeId } from "@/lib/api";

const DOC_KEY = "AI_chat_application.doc_id";
const CHAT_KEY = "AI_chat_application.chat_id";
const MODE_KEY = "AI_chat_application.mode";

function read(storage: () => Storage, key: string): string | null {
  try {
    return storage().getItem(key);
  } catch {
    return null;
  }
}

function write(storage: () => Storage, key: string, value: string | null): void {
  try {
    if (value === null) storage().removeItem(key);
    else storage().setItem(key, value);
  } catch {}
}

const session = () => sessionStorage;
const local = () => localStorage;

/** The document open in this tab, so a refresh reopens it; null clears it. */
export const openDocId = {
  get: () => read(session, DOC_KEY),
  set: (docId: string | null) => write(session, DOC_KEY, docId),
};

/** The chat open in this tab for a document, so a refresh reopens it; null clears it. */
export const openChatId = {
  get: (docId: string) => read(session, `${CHAT_KEY}.${docId}`),
  set: (docId: string, chatId: string | null) => write(session, `${CHAT_KEY}.${docId}`, chatId),
};

/** The mode last picked for an upload, remembered in this browser. */
export const lastMode = {
  get: () => read(local, MODE_KEY),
  set: (mode: ModeId) => write(local, MODE_KEY, mode),
};
