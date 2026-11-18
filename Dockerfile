# Bilingual RAG assistant: FastAPI app with a vector index built into the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    PORT=8000

# Run as a normal user, not root (uid 1000 also suits Hugging Face Spaces).
RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# 1) Dependencies first: this layer is cached while only the code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Code and the committed corpus snapshot (no raw download, no secrets).
RUN mkdir -p /app/data/index /app/.cache && chown -R appuser /app
COPY --chown=appuser src ./src
COPY --chown=appuser data/corpus ./data/corpus
USER appuser

# 3) Build the vector index at build time: downloads the embedding model
#    (about 470 MB) and embeds the chunks once. The container then starts
#    fast and needs no network for search. Only /ask calls the Mistral API.
RUN python -m rag_assistant.index

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn rag_assistant.api:app --host 0.0.0.0 --port ${PORT}"]
