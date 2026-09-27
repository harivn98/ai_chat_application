import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI_chat_application",
  description: "Chat with your documents — hybrid BM25 + vector retrieval, Qwen3 via Ollama",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
