"use client";

import { CloudOptions, ModeId, ModeInfo } from "@/lib/api";

const CLOUD_SWITCHES: { key: keyof CloudOptions; label: string; hint: string }[] = [
  { key: "reranker", label: "Reranker", hint: "re-orders the retrieved passages on this machine" },
  { key: "prejudge", label: "Pre-judge", hint: "checks the passages hold the answer before replying" },
];

/** Cloud mode's switches. The last one that is on can't be turned off, so at least one always runs. */
function CloudSwitches({ value, onChange }: { value: CloudOptions; onChange: (cloud: CloudOptions) => void }) {
  return (
    <div className="mode-switches">
      {CLOUD_SWITCHES.map(({ key, label, hint }) => {
        const lastOneOn = value[key] && CLOUD_SWITCHES.every((s) => s.key === key || !value[s.key]);
        return (
          <label key={key} className={`switch ${lastOneOn ? "locked" : ""}`}>
            <input
              type="checkbox"
              role="switch"
              checked={value[key]}
              disabled={lastOneOn}
              onChange={(e) => onChange({ ...value, [key]: e.target.checked })}
            />
            <span className="switch-track" aria-hidden />
            <span className="switch-text">
              <span className="switch-label">{label}</span>
              <span className="switch-hint">{hint}</span>
            </span>
          </label>
        );
      })}
      <span className="switch-note">At least one must be on.</span>
    </div>
  );
}

/** The mode for the next upload. A mode whose API keys are missing can't be picked. */
export default function ModePicker({
  modes,
  value,
  onChange,
  cloud,
  onCloudChange,
}: {
  modes: ModeInfo[];
  value: ModeId;
  onChange: (mode: ModeId) => void;
  cloud: CloudOptions;
  onCloudChange: (cloud: CloudOptions) => void;
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
            {m.id === "cloud" && selected && <CloudSwitches value={cloud} onChange={onCloudChange} />}
          </div>
        );
      })}
    </fieldset>
  );
}
