# Phase 2A — FastAPI + ONNX Runtime

Serves the Phase 1 ONNX artifact behind FastAPI + uvicorn. The model file is
**not** baked into the image — it is mounted at run time so a single image
covers every cell of the worker × thread matrix.

## Local (no Docker)

```bash
make setup-fastapi
cd project/2-serving/fastapi
ORT_INTRA_OP_NUM_THREADS=2 \
ORT_INTER_OP_NUM_THREADS=1 \
WEB_CONCURRENCY=1 \
MODEL_PATH=../../1-model/artifacts/model.onnx \
SCHEMA_PATH=../../1-model/artifacts/schema.json \
uv run uvicorn app:app --host 127.0.0.1 --port 8001 --workers 1 --no-access-log
```

## Docker

```bash
# Build once
docker build -t tg-serving-fastapi project/2-serving/fastapi

# Run a matrix cell (e.g. 2 workers / 1 ORT thread)
docker run --rm -d --name tg-fa \
    --cpus=2 --memory=2g --cpuset-cpus=0,1 \
    -v "$PWD/project/1-model/artifacts:/app:ro" \
    -e WEB_CONCURRENCY=2 \
    -e ORT_INTRA_OP_NUM_THREADS=1 \
    -e ORT_INTER_OP_NUM_THREADS=1 \
    -p 8001:8000 \
    tg-serving-fastapi
```

## Endpoints

```bash
curl http://localhost:8001/healthz
curl http://localhost:8001/version
curl -X POST http://localhost:8001/predict \
    -H 'content-type: application/json' \
    -d '{"features":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]}'
```
