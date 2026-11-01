FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    PORT=8080

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY support_rag.py app.py ./
COPY static ./static
COPY data ./data

# Download the embedding model and build the FAISS index at build time,
# so the container starts fast and needs no internet for embeddings.
RUN LLM_BACKEND=mock python -c "from support_rag import SupportRAG; SupportRAG()"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8080
# Render / Cloud Run set $PORT; default is 8080 for local `docker run`
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
