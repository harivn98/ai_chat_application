"use client";

import { useCallback, useEffect, useState } from "react";
import { DocInfo, getDocument } from "@/lib/api";
import { openDocId } from "@/lib/remembered";

const POLL_MS = 800;

/** The document this tab works on: reopened after a page refresh, and polled while it is being indexed (the chat
 * is usable meanwhile). Closing it keeps it, with its chats, in the documents list. `restored` turns true once the
 * attempt to reopen the last document has finished. */
export function useCurrentDocument() {
  const [doc, setDoc] = useState<DocInfo | null>(null);
  const [restored, setRestored] = useState(false);

  // Reopen the last document after a page refresh (also while it is still being indexed)
  useEffect(() => {
    const id = openDocId.get();
    if (!id) {
      setRestored(true);
      return;
    }
    getDocument(id)
      .then((d) => {
        if (d.status !== "failed") setDoc(d);
      })
      .catch(() => {})
      .finally(() => setRestored(true));
  }, []);

  const indexing = !!doc && doc.status !== "ready" && doc.status !== "failed";
  const docId = doc?.doc_id;
  useEffect(() => {
    if (!indexing || !docId) return;
    let alive = true;
    const timer = setInterval(async () => {
      try {
        const d = await getDocument(docId);
        if (alive) setDoc(d);
      } catch {
        // transient network error: keep polling
      }
    }, POLL_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [indexing, docId]);

  /** Make a document (freshly uploaded, or picked from the list) the current one. */
  const open = useCallback((d: DocInfo) => {
    openDocId.set(d.doc_id);
    setDoc(d);
  }, []);

  /** Go back to the upload page. The document and its chats stay saved (and keep indexing if not done). */
  const close = useCallback(() => {
    openDocId.set(null);
    setDoc(null);
  }, []);

  return { doc, restored, open, close };
}
