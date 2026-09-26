"""Runtime settings, read from environment variables (see docker-compose.yml)."""
import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    mongo_uri: str = os.getenv("MONGO_URI", "mongodb://admin:admin@localhost:27017/?directConnection=true")
    mongo_db: str = os.getenv("MONGO_DB", "ragdb")
    vector_index: str = os.getenv("VECTOR_INDEX", "chunk_vector_index")

    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    llm_model: str = os.getenv("LLM_MODEL", "qwen3:8b")
    llm_num_ctx: int = int(os.getenv("LLM_NUM_CTX", "8192"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    llm_think: bool = _bool("LLM_THINK", False)

    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    embed_dim: int = int(os.getenv("EMBED_DIM", "384"))
    # bge v1.5 recommends this instruction on the *query* side only
    query_instruction: str = os.getenv(
        "QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: "
    )

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))        # characters
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))   # characters
    top_k: int = int(os.getenv("TOP_K", "6"))                     # chunks sent to the LLM
    candidates_per_retriever: int = int(os.getenv("CANDIDATES", "20"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    history_turns: int = int(os.getenv("HISTORY_TURNS", "6"))

    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "/data"))


settings = Settings()
UPLOAD_DIR = settings.data_dir / "uploads"
MARKDOWN_DIR = settings.data_dir / "markdown"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
