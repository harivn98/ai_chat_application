"use client";

import { ModeId, ModeInfo } from "@/lib/api";

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
        const models = [
          m.embed_model,
          m.context_model,
          m.llm_model,
          m.reranker_model && `rerank: ${m.reranker_model}`,
          m.prejudge_model && `pre-judge: ${m.prejudge_model}`,
          `${m.passages} passages per question`,
        ]
          .filter(Boolean)
          .join(" · ");
        return (
          <label key={m.id} className={`mode ${value === m.id ? "selected" : ""} ${disabled ? "disabled" : ""}`}>
            <input
              type="radio"
              name="mode"
              value={m.id}
              checked={value === m.id}
              disabled={disabled}
              onChange={() => onChange(m.id)}
            />
            <span className="mode-label">{m.label}</span>
            <span className="mode-description">{m.description}</span>
            <span className="mode-models">{models}</span>
            {disabled && <span className="mode-missing">Needs {m.missing_keys.join(" and ")} (see README)</span>}
          </label>
        );
      })}
    </fieldset>
  );
}
