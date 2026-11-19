# Bilingual RAG assistant: FastAPI app with a vector index built into the image.
# The same image runs locally (docker run), in CI, and on AWS Lambda.
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
#    The layers go from what changes least to what changes most.
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir -p /app/data/index /app/.cache && chown -R appuser /app
USER appuser

# 2) Build the vector index at build time: downloads the embedding model (about 470 MB)
#    and embeds the committed corpus snapshot once. The container then starts fast and
#    needs no network for search. Only /ask calls the LLM (Amazon Bedrock).
#    Only the files that build the index are copied before this step, so a change to the
#    rest of the code does not rebuild it (several minutes of CPU, and a new 500 MB layer
#    to upload). A change to these files or to the corpus does.
COPY --chown=appuser src/rag_assistant/__init__.py src/rag_assistant/config.py \
     src/rag_assistant/embeddings.py src/rag_assistant/index.py ./src/rag_assistant/
COPY --chown=appuser data/corpus ./data/corpus
RUN python -m rag_assistant.index

# 3) The rest of the code: changes often, and this layer is small.
COPY --chown=appuser src ./src

# 4) Ready for read-only disks (AWS Lambda): never try to download the model at run time.
#    The index is copied to /tmp at start-up if its folder is read-only (see index.py).
ENV HF_HUB_OFFLINE=1

# 5) AWS Lambda Web Adapter: on Lambda, it turns Lambda events into normal HTTP requests
#    to uvicorn, so the FastAPI code is unchanged. Outside Lambda it does nothing.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0 /lambda-adapter /opt/extensions/lambda-adapter
ENV AWS_LWA_READINESS_CHECK_PATH=/health \
    AWS_LWA_ASYNC_INIT=true

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn rag_assistant.api:app --host 0.0.0.0 --port ${PORT}"]
