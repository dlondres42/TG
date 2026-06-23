# syntax=docker/dockerfile:1.6
#
# Phase 2B — BentoML + ONNX Runtime + adaptive batching
# Hand-written (not `bentoml containerize`) to keep parity with Phase 2A.
# Model artifact is mounted at run time from project/1-model/artifacts.

FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /build
COPY pyproject.toml ./
RUN uv venv /opt/venv --python 3.11 && uv sync --no-dev


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv \
    WEB_CONCURRENCY=1 \
    ORT_INTRA_OP_NUM_THREADS=2 \
    ORT_INTER_OP_NUM_THREADS=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    MODEL_PATH=/app/model.onnx \
    SCHEMA_PATH=/app/schema.json \
    BATCHING_PATH=/srv/batching.json

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv
COPY service.py entrypoint.sh batching.json ./
RUN chmod +x entrypoint.sh

EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=2s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/livez || exit 1

ENTRYPOINT ["/srv/entrypoint.sh"]
