# HTTP-сервис контракта 2.0 для SyncBack (src/syncscribe_ml/server.py).
#   docker build -t syncscribe-ml .
#   docker build --build-arg EXTRAS=server,local -t syncscribe-ml:local .   # BGE-M3 в Python, без Ollama
FROM python:3.12-slim

ARG EXTRAS=server
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SYNCSCRIBE_ML_PROMPTS_DIR=/app/prompts

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[${EXTRAS}]"
COPY prompts ./prompts

RUN useradd --system --create-home ml
USER ml

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1
CMD ["uvicorn", "syncscribe_ml.server:app", "--host", "0.0.0.0", "--port", "8000"]
