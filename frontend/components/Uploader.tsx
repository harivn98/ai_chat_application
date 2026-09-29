"use client";

import { useEffect, useState } from "react";
import FileDropzone from "@/components/FileDropzone";
import ModePicker from "@/components/ModePicker";
import { CloudOptions, DocInfo, getModes, ModeId, ModeInfo, uploadDocument } from "@/lib/api";
import { lastCloudOptions, lastMode } from "@/lib/remembered";

const EXTENSIONS = [".pdf", ".txt", ".md", ".markdown"];

export default function Uploader({ onUploaded }: { onUploaded: (doc: DocInfo) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [modes, setModes] = useState<ModeInfo[]>([]);
  const [mode, setMode] = useState<ModeId>("private");
  const [cloud, setCloud] = useState<CloudOptions>({ reranker: true, prejudge: true });

  // The modes the backend offers, then the choices of the last upload in this browser (if still usable)
  useEffect(() => {
    getModes()
      .then((ms) => {
        setModes(ms);
        const saved = lastMode.get();
        const usable = ms.find((m) => m.id === saved && m.missing_keys.length === 0);
        if (usable) setMode(usable.id);
        const savedCloud = lastCloudOptions.get();
        if (savedCloud) setCloud(savedCloud);
      })
      .catch(() => {}); // without the list, uploads use private mode
  }, []);

  function chooseMode(m: ModeId) {
    setMode(m);
    lastMode.set(m);
  }

  function chooseCloud(c: CloudOptions) {
    setCloud(c);
    lastCloudOptions.set(c);
  }

  function chooseFile(f: File | undefined) {
    setError(null);
    if (!f) return;
    const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    if (!EXTENSIONS.includes(ext)) {
      setError("Please choose a PDF, TXT or MD file.");
      return;
    }
    setFile(f);
  }

  async function upload() {
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

      <FileDropzone file={file} accept={EXTENSIONS} hint=".pdf · .txt · .md" onFile={chooseFile} />

      {error && <p className="error">{error}</p>}

      <button className="btn btn-primary btn-wide" disabled={!file || uploading} onClick={upload}>
        {uploading ? "Uploading…" : "Upload & index"}
      </button>
    </section>
  );
}
