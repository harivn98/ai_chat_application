"use client";

import { useRef, useState } from "react";

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** A drop area for one file; clicking it (or Enter / Space) opens the file browser instead. */
export default function FileDropzone({
  file,
  accept,
  hint,
  onFile,
}: {
  file: File | null;
  accept: string[]; // file extensions the browser dialog offers, e.g. ".pdf"
  hint: string; // shown under the prompt while no file is chosen
  onFile: (file: File | undefined) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  return (
    <div
      className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        onFile(e.dataTransfer.files?.[0]);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept.join(",")}
        hidden
        onChange={(e) => onFile(e.target.files?.[0])}
      />
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" aria-hidden>
        <path d="M12 16V4m0 0l-4.5 4.5M12 4l4.5 4.5M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
      {file ? (
        <div>
          <strong>{file.name}</strong>
          <div className="muted">{formatSize(file.size)} · click to change</div>
        </div>
      ) : (
        <div>
          <strong>Drop a file here</strong> or click to browse
          <div className="muted">{hint}</div>
        </div>
      )}
    </div>
  );
}
