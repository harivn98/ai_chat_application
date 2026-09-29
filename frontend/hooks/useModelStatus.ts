"use client";

import { useEffect, useState } from "react";
import { getAnsweringModel, ModelStatus } from "@/lib/api";

const FAST_MS = 1000; // while the model loads or a question is being answered (a question can start a load)
const SLOW_MS = 5000; // otherwise: notices the load after an upload, or after Ollama unloaded an idle model

/** Private mode's answering model, polled so the chat can show when it is being loaded. null when disabled. */
export function useModelStatus(enabled: boolean, busy: boolean) {
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const fast = busy || status?.state === "loading";

  useEffect(() => {
    if (!enabled) return;
    let alive = true;
    const check = () =>
      getAnsweringModel()
        .then((s) => alive && setStatus(s))
        .catch(() => {}); // transient network error: keep polling
    check();
    const timer = setInterval(check, fast ? FAST_MS : SLOW_MS);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, [enabled, fast]);

  return enabled ? status : null;
}
