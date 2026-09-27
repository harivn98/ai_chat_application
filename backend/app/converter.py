"""Step 1 of ingestion: normalise every supported format into Markdown."""
import re
from pathlib import Path

from charset_normalizer import from_bytes


class ConversionError(Exception):
    pass


def _decode(raw: bytes) -> str:
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    # Statistical detection is unreliable on short files; only trust a confident result
    # for long inputs, otherwise assume Windows-1252 (a superset of Latin-1).
    if len(raw) > 2000:
        best = from_bytes(raw).best()
        if best is not None and best.encoding and not best.encoding.startswith(("big5", "gb", "shift_jis", "euc")):
            return str(best)
    return raw.decode("cp1252", errors="replace")


def _normalise(md: str) -> str:
    md = md.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    md = re.sub(r"[ \t]+\n", "\n", md)          # trailing spaces
    md = re.sub(r"\n{3,}", "\n\n", md)           # collapse blank runs
    return md.strip() + "\n"


def _pdf_to_markdown(path: Path) -> str:
    import pymupdf4llm  # imported lazily: heavy import

    md = pymupdf4llm.to_markdown(str(path), show_progress=False)
    if len(re.sub(r"\W", "", md)) < 20:
        raise ConversionError(
            "No extractable text found in this PDF. It is probably a scanned/image-only PDF (OCR is not enabled)."
        )
    return md


def _txt_to_markdown(raw: bytes, title: str) -> str:
    text = _decode(raw)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    body = []
    for p in paragraphs:
        # Escape characters that would unintentionally become Markdown syntax at line start
        lines = [re.sub(r"^(\s*)([#>]|[-*+]\s|\d+\.\s)", r"\1\\\2", ln) for ln in p.split("\n")]
        body.append("\n".join(ln.rstrip() for ln in lines))
    return f"# {title}\n\n" + "\n\n".join(body)


def _md_to_markdown(raw: bytes) -> str:
    text = _decode(raw)
    # Drop YAML front matter; it is metadata, not content
    return re.sub(r"\A---\n.*?\n---\n", "", text, count=1, flags=re.S)


def to_markdown(path: Path, original_name: str) -> str:
    ext = path.suffix.lower()
    title = Path(original_name).stem.replace("_", " ").replace("-", " ").strip() or "Document"
    if ext == ".pdf":
        md = _pdf_to_markdown(path)
    elif ext == ".txt":
        md = _txt_to_markdown(path.read_bytes(), title)
    elif ext in {".md", ".markdown"}:
        md = _md_to_markdown(path.read_bytes())
    else:
        raise ConversionError(f"Unsupported file type: {ext}")
    md = _normalise(md)
    if not md.strip():
        raise ConversionError("The document is empty after conversion.")
    return md
