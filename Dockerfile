FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so a source change does not re-resolve the whole tree.
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install .

COPY src/ ./src/
COPY evals/ ./evals/
RUN pip install -e . --no-deps

# The embedder and cross-encoder weights are ~500MB together. Baking them in
# keeps cold start off the request path at the cost of image size.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-base-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "rag_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
