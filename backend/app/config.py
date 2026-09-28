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
    llm_keep_alive: str = os.getenv("LLM_KEEP_ALIVE", "30m")            # how long Ollama keeps the model loaded
    llm_timeout: int = int(os.getenv("LLM_TIMEOUT", "600"))             # max seconds without output from Ollama
    llm_load_timeout: int = int(os.getenv("LLM_LOAD_TIMEOUT", "1800"))  # max seconds to load the model
    history_turns: int = int(os.getenv("HISTORY_TURNS", "6"))           # previous chat messages sent with a question

    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    embed_dim: int = int(os.getenv("EMBED_DIM", "384"))
    # bge v1.5 recommends this instruction on the *query* side only
    query_instruction: str = os.getenv(
        "QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: "
    )

    # Contextual Retrieval: a small LLM writes a context for each chunk before embedding/BM25
    contextual_embedding: bool = _bool("CONTEXTUAL_EMBEDDING", True)
    context_model: str = os.getenv("CONTEXT_MODEL", "qwen3:4b-instruct")  # non-thinking; plain qwen3:4b is thinking-only
    context_num_ctx: int = int(os.getenv("CONTEXT_NUM_CTX", "16384"))  # tokens of document the model reads

    # Pre-judge: before answering, the answering model (LLM_MODEL) checks whether the passages can answer the question
    prejudge_enabled: bool = _bool("PREJUDGE_ENABLED", True)

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))        # characters
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))   # characters
    top_k: int = int(os.getenv("TOP_K", "8"))                     # chunks sent to the LLM
    bm25_candidates: int = int(os.getenv("BM25_CANDIDATES", "10"))
    vector_candidates: int = int(os.getenv("VECTOR_CANDIDATES", "20"))  # private mode: no score cutoff
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    # Reranker: a cross-encoder re-scores every fused BM25 + vector chunk and keeps the best TOP_K
    reranker_enabled: bool = _bool("RERANKER_ENABLED", True)
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

    # Cloud mode (chosen per upload in the UI): every cloud model runs through OpenRouter, so document text and
    # questions leave this machine. The API key comes from the machine's environment, never from the committed .env.
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_url: str = os.getenv("OPENROUTER_URL", "https://openrouter.ai/api/v1").rstrip("/")
    cloud_embed_model: str = os.getenv("CLOUD_EMBED_MODEL", "google/gemini-embedding-2")
    cloud_embed_dim: int = int(os.getenv("CLOUD_EMBED_DIM", "768"))
    cloud_vector_min_score: float = float(os.getenv("CLOUD_VECTOR_MIN_SCORE", "0"))  # 0: no cutoff (not tuned yet)
    cloud_context_model: str = os.getenv("CLOUD_CONTEXT_MODEL", "google/gemini-3.5-flash-lite")
    cloud_llm_model: str = os.getenv("CLOUD_LLM_MODEL", "deepseek/deepseek-v4.1-flash")
    cloud_prejudge_model: str = os.getenv("CLOUD_PREJUDGE_MODEL", "google/gemini-3.5-flash-lite")
    # Cloud pre-judge mode has no reranker and large-context models, so it retrieves and sends more passages
    # (cloud reranker mode uses TOP_K / BM25_CANDIDATES / VECTOR_CANDIDATES like private mode)
    cloud_prejudge_top_k: int = int(os.getenv("CLOUD_PREJUDGE_TOP_K", "8"))  # passages sent to the LLMs
    cloud_prejudge_bm25_candidates: int = int(os.getenv("CLOUD_PREJUDGE_BM25_CANDIDATES", "20"))
    cloud_prejudge_vector_candidates: int = int(os.getenv("CLOUD_PREJUDGE_VECTOR_CANDIDATES", "30"))

    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "/data"))

    # QASPER evaluation (python -m app.evaluate)
    judge_model: str = os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "qwen3:8b"))
    eval_dir: Path = Path(os.getenv("EVAL_DIR", str(Path(__file__).resolve().parents[2] / "evaluation_metrics")))


settings = Settings()
UPLOAD_DIR = settings.data_dir / "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}
