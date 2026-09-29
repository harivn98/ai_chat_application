"use client";

import { useEffect, useRef, useState } from "react";

const MAX_HEIGHT = 180; // px: the box grows with its text up to this height, then scrolls

/** The question box. Enter sends (Shift+Enter adds a line); while an answer streams, the button stops it. */
export default function Composer({
  placeholder,
  disabled,
  busy,
  onSend,
  onStop,
}: {
  placeholder: string;
  disabled: boolean;
  busy: boolean;
  onSend: (question: string) => void;
  onStop: () => void;
}) {
  const [input, setInput] = useState("");
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Focus when the chat opens and after every answer, so the next question can be typed right away
  useEffect(() => {
    if (!busy) inputRef.current?.focus();
  }, [busy]);

  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, [input]);

  function send() {
    const question = input.trim();
    if (!question || disabled) return;
    onSend(question);
    setInput("");
  }

  return (
    <form
      className="composer"
      onSubmit={(e) => {
        e.preventDefault();
        send();
      }}
    >
      <textarea
        ref={inputRef}
        rows={1}
        value={input}
        disabled={disabled}
        placeholder={placeholder}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            send();
          }
        }}
      />
      {busy ? (
        <button type="button" className="btn btn-stop" onClick={onStop}>
          Stop
        </button>
      ) : (
        <button type="submit" className="btn btn-primary" disabled={!input.trim() || disabled}>
          Send
        </button>
      )}
    </form>
  );
}
