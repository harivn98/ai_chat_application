"use client";

import { useState } from "react";
import { ModeId, ModeInfo } from "@/lib/api";

// The main limitations of private mode, in plain words
const PRIVATE_LIMITATIONS: { topic: string; text: string }[] = [
  { topic: "Files", text: "PDF, TXT and Markdown only. Scanned PDFs (pages saved as pictures) can't be read." },
  {
    topic: "Images",
    text: "Pictures, charts and screenshots are skipped. Diagrams keep only their text labels, without the layout.",
  },
  { topic: "Language", text: "Works best with English. Other languages, especially non-Latin scripts, give weaker results." },
  {
    topic: "Questions",
    text: "Each answer is based on a few passages of one document, so questions about the whole document (summaries, \"list every…\") may be incomplete.",
  },
  { topic: "Accuracy", text: "It can make mistakes. Check the cited passages." },
  {
    topic: "Speed",
    text: "Everything runs on this computer: an upload takes about 40 seconds per 10 pages, and the first answer after an upload or a break can take about a minute while the model loads.",
  },
];

/** A "Limitations" toggle that expands the list inside the private mode card. */
function PrivateLimitations() {
  const [open, setOpen] = useState(false);
  return (
    <div className="limits">
      <button
        type="button"
        className="limits-toggle"
        aria-expanded={open}
        aria-controls="private-limitations"
        onClick={() => setOpen((o) => !o)}
      >
        {open ? "Hide limitations" : "Limitations"}
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" aria-hidden>
          <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
      {open && (
        <dl id="private-limitations" className="limits-list">
          {PRIVATE_LIMITATIONS.map(({ topic, text }) => (
            <div key={topic}>
              <dt>{topic}</dt>
              <dd>{text}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

/** The mode for the next upload. A mode whose API keys are missing can't be picked. */
export default function ModePicker({
  modes,
  value,
  onChange,
}: {
  modes: ModeInfo[];
  value: ModeId;
  onChange: (mode: ModeId) => void;
}) {
  return (
    <fieldset className="modes">
      <legend>Where should this document be processed?</legend>
      {modes.map((m) => {
        const disabled = m.missing_keys.length > 0;
        const selected = value === m.id;
        return (
          <div key={m.id} className={`mode ${selected ? "selected" : ""} ${disabled ? "disabled" : ""}`}>
            <label className="mode-choice">
              <input
                type="radio"
                name="mode"
                value={m.id}
                checked={selected}
                disabled={disabled}
                onChange={() => onChange(m.id)}
              />
              <span className="mode-label">{m.label}</span>
              <span className="mode-description">{m.description}</span>
            </label>
            {disabled && <span className="mode-missing">Needs {m.missing_keys.join(" and ")} (see README)</span>}
            {m.id === "private" && <PrivateLimitations />}
          </div>
        );
      })}
    </fieldset>
  );
}
