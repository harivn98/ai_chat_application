"use client";

import { useEffect, useRef, useState } from "react";
import ModePicker from "@/components/ModePicker";
import { CloudOptions, DocInfo, getModes, ModeId, ModeInfo, uploadDocument } from "@/lib/api";

const EXTENSIONS = [".pdf", ".txt", ".md", ".markdown"];
const MODE_KEY = "AI_chat_application.mode"; // the last mode picked, remembered in this browser
const CLOUD_KEY = "AI_chat_application.cloud"; // the last cloud switches, likewise

function fmtSize(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

export default function Uploader({ onUploaded }: { onUploaded: (doc: DocInfo) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [modes, setModes] = useState<ModeInfo[]>([]);
  const [mode, setMode] = useState<ModeId>("private");
  const [cloud, setCloud] = useState<CloudOptions>({ reranker: true, prejudge: true });
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    getModes()
      .then((ms) => {
        setModes(ms);
        let saved: string | null = null;
        try {
          saved = localStorage.getItem(MODE_KEY);
        } catch {}
        const usable = ms.find((m) => m.id === saved && m.missing_keys.length === 0);
        if (usable) setMode(usable.id);
        try {
          const c = JSON.parse(localStorage.getItem(CLOUD_KEY) ?? "null");
          if (typeof c?.reranker === "boolean" && typeof c?.prejudge === "boolean" && (c.reranker || c.prejudge)) {
            setCloud({ reranker: c.reranker, prejudge: c.prejudge });
          }
        } catch {}
      })
      .catch(() => {}); // without the list, uploads use private mode
  }, []);

  function chooseMode(m: ModeId) {
    setMode(m);
    try {
      localStorage.setItem(MODE_KEY, m);
    } catch {}
  }

  function chooseCloud(c: CloudOptions) {
    setCloud(c);
    try {
      localStorage.setItem(CLOUD_KEY, JSON.stringify(c));
    } catch {}
  }

  function pick(f: File | undefined | null) {
    setError(null);
    if (!f) return;
    const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    if (!EXTENSIONS.includes(ext)) {
      setError("Please choose a PDF, TXT or MD file.");
      return;
    }
    setFile(f);
  }

  async function submit() {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      onUploaded(await uploadDocument(file, mode, cloud)); // the chat opens right away; indexing continues in the background
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUploading(false);
    }
  }

  return (
    <section className="card upload-card">
      <p className="eyebrow">Step 1</p>
      <h2>Upload a document to start chatting</h2>

      {modes.length > 0 && (
        <ModePicker modes={modes} value={mode} onChange={chooseMode} cloud={cloud} onCloudChange={chooseCloud} />
      )}
      {mode !== "private" && (
        <p className="cloud-warning">
          Cloud mode sends the document text and your questions to OpenRouter, which passes them to Google and
          DeepSeek. Use private mode for confidential documents.
        </p>
      )}

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
          pick(e.dataTransfer.files?.[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept={EXTENSIONS.join(",")}
          hidden
          onChange={(e) => pick(e.target.files?.[0])}
        />
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" aria-hidden>
          <path d="M12 16V4m0 0l-4.5 4.5M12 4l4.5 4.5M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        {file ? (
          <div>
            <strong>{file.name}</strong>
            <div className="muted">{fmtSize(file.size)} · click to change</div>
          </div>
        ) : (
          <div>
            <strong>Drop a file here</strong> or click to browse
            <div className="muted">.pdf · .txt · .md</div>
          </div>
        )}
      </div>

      {error && <p className="error">{error}</p>}

      <button className="btn btn-primary btn-wide" disabled={!file || uploading} onClick={submit}>
        {uploading ? "Uploading…" : "Upload & index"}
      </button>
    </section>
  );
}
