"""Step 2 of ingestion: Markdown-aware chunking.

The document is split along its heading structure first, then each section is
packed into chunks of ~chunk_size characters on paragraph / sentence boundaries,
with a character overlap between consecutive chunks of the same section.
Every chunk keeps its heading path ("Intro > Setup > Docker"), which is
prepended to the text that gets embedded and BM25-indexed, and its position
in the Markdown, so the UI can highlight a cited chunk inside the document.
"""
import re
from bisect import bisect_left
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Chunk:
    index: int
    section: str       # heading path
    text: str          # chunk body (markdown)
    start: int | None = None  # position in the Markdown, in UTF-16 code units (see _locate)
    end: int | None = None

    @property
    def content(self) -> str:
        """Text used for embedding and BM25 (heading path gives context)."""
        return f"{self.section}\n\n{self.text}" if self.section else self.text


def _clean_heading(h: str) -> str:
    h = re.sub(r"[*_`]+", "", h)                 # strip emphasis/code markers
    h = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", h)    # links -> text
    return h.strip()


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return [(heading_path, body)] keeping code fences intact."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False

    def flush():
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(t for _, t in stack), body))
        buf.clear()

    for line in markdown.split("\n"):
        if FENCE_RE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue
        m = None if in_fence else HEADING_RE.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), _clean_heading(m.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            if title:
                stack.append((level, title))
        else:
            buf.append(line)
    flush()
    return sections


def _split_blocks(body: str) -> list[str]:
    """Paragraph-level blocks; fenced code blocks and tables stay whole."""
    blocks, cur, in_fence = [], [], False
    for line in body.split("\n"):
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if not in_fence and not line.strip():
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    return blocks


def _hard_split(text: str, size: int) -> list[str]:
    """Split an oversized block by sentences, then by lines, then by characters."""
    pieces = SENTENCE_RE.split(text)
    if len(pieces) == 1:
        pieces = text.split("\n")
    out, cur = [], ""
    for p in pieces:
        while len(p) > size:                      # last resort: raw character cut
            cut = p.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:cut])
            p = p[cut:].lstrip()
        joined = f"{cur} {p}".strip() if cur else p
        if len(joined) > size and cur:
            out.append(cur)
            cur = p
        else:
            cur = joined
    if cur:
        out.append(cur)
    return out


def _tail(text: str, n: int) -> str:
    """Last ~n characters, starting at a word boundary."""
    if n <= 0 or len(text) <= n:
        return text if n > 0 else ""
    t = text[-n:]
    sp = t.find(" ")
    return t[sp + 1:] if 0 <= sp < n // 2 else t


def chunk_markdown(markdown: str, chunk_size: int = 1000, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, body in split_sections(markdown):
        units: list[str] = []
        for block in _split_blocks(body):
            units.extend([block] if len(block) <= chunk_size else _hard_split(block, chunk_size))

        cur = ""
        for u in units:
            candidate = f"{cur}\n\n{u}" if cur else u
            if len(candidate) <= chunk_size or not cur:
                cur = candidate
                continue
            chunks.append(Chunk(len(chunks), section, cur.strip()))
            prefix = _tail(cur, overlap)
            cur = f"{prefix}\n\n{u}" if prefix and len(prefix) + len(u) + 2 <= chunk_size else u
        if cur.strip():
            chunks.append(Chunk(len(chunks), section, cur.strip()))
    _locate(markdown, chunks)
    return chunks


def _locate(markdown: str, chunks: list[Chunk]) -> None:
    """Set each chunk's start/end in the Markdown.

    Chunk text differs from the source only in whitespace (blocks and sentences are re-joined), so it is
    matched with any run of whitespace between its words. Chunks come in document order, so each search
    starts at the previous chunk's start (they overlap). Positions are UTF-16 code units, the way the
    browser indexes strings; characters outside the BMP (e.g. math letters like 𝑥) count twice there.
    """
    astral = [i for i, ch in enumerate(markdown) if ord(ch) > 0xFFFF]
    pos = 0
    for c in chunks:
        pattern = r"\s+".join(map(re.escape, c.text.split()))
        m = re.compile(pattern).search(markdown, pos)
        if not m:
            continue
        c.start = m.start() + bisect_left(astral, m.start())
        c.end = m.end() + bisect_left(astral, m.end())
        pos = m.start()
