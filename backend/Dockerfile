FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    SENTENCE_TRANSFORMERS_HOME=/models

WORKDIR /app

# CPU-only torch keeps the image ~1.5 GB smaller; bge-small is fast on CPU.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image so the container runs offline.
ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ENV EMBED_MODEL=${EMBED_MODEL}
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['EMBED_MODEL'])"
# Same for the rerankers: both are baked in, RERANKER_MODEL (.env) picks which one runs.
#   BAAI/bge-reranker-base                (278M parameters, ~1.1 GB, more accurate, slower on CPU)
#   cross-encoder/ms-marco-MiniLM-L6-v2   (23M parameters, ~90 MB, fast)
ARG RERANKER_MODELS="BAAI/bge-reranker-base cross-encoder/ms-marco-MiniLM-L6-v2"
ENV RERANKER_MODELS=${RERANKER_MODELS}
RUN python -c "import os; from sentence_transformers import CrossEncoder; [CrossEncoder(m) for m in os.environ['RERANKER_MODELS'].split()]"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
