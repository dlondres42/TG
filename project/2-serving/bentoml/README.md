# Phase 2B — BentoML + ONNX Runtime (adaptive batching)

Serves the Phase 1 ONNX artifact behind BentoML's HTTP server. Adaptive
batching parameters are read from [batching.json](batching.json), produced by
[calibrate_batching.py](calibrate_batching.py) before Phase 3 begins.

## Local (no Docker)

```bash
make setup-bentoml
cd project/2-serving/bentoml
WEB_CONCURRENCY=1 \
ORT_INTRA_OP_NUM_THREADS=2 \
ORT_INTER_OP_NUM_THREADS=1 \
MODEL_PATH=../../1-model/artifacts/model.onnx \
SCHEMA_PATH=../../1-model/artifacts/schema.json \
BATCHING_PATH=./batching.json \
uv run bentoml serve service:TgService --host 127.0.0.1 --port 8002
```

## Docker

```bash
docker build -t tg-serving-bentoml project/2-serving/bentoml

docker run --rm -d --name tg-bm \
    --cpus=2 --memory=2g --cpuset-cpus=0,1 \
    -v "$PWD/project/1-model/artifacts:/app:ro" \
    -e WEB_CONCURRENCY=2 \
    -e ORT_INTRA_OP_NUM_THREADS=1 \
    -e ORT_INTER_OP_NUM_THREADS=1 \
    -p 8002:8000 \
    tg-serving-bentoml
```

## Endpoints

BentoML defaults all `@bentoml.api` methods to POST. Liveness uses BentoML's
built-in GET `/livez`.

```bash
curl http://localhost:8002/livez                                  # GET, built-in
curl -X POST http://localhost:8002/healthz -d '{}' -H 'content-type: application/json'
curl -X POST http://localhost:8002/version -d '{}' -H 'content-type: application/json'
curl -X POST http://localhost:8002/predict \
    -H 'content-type: application/json' \
    -d '{"features":[[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]}'
```

## Adaptive batching

`batching.json` controls `max_batch_size` and `max_latency_ms`. The committed
placeholder (`max_batch_size=1, max_latency_ms=1`) effectively disables
batching. After running [calibrate_batching.py](calibrate_batching.py) the file
is overwritten with the picked values; those are held fixed across the Phase 3
matrix per the plan.
