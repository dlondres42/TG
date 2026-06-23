#!/usr/bin/env bash
set -euo pipefail

# Matrix knobs (defaults match container ENV in Dockerfile).
: "${WEB_CONCURRENCY:=1}"
: "${ORT_INTRA_OP_NUM_THREADS:=2}"
: "${ORT_INTER_OP_NUM_THREADS:=1}"
: "${FASTAPI_THREADPOOL_SIZE:=32}"

# Mirror ORT intra-op into OpenMP / BLAS so they don't spawn their own pools.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$ORT_INTRA_OP_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$ORT_INTRA_OP_NUM_THREADS}"
export ORT_INTRA_OP_NUM_THREADS ORT_INTER_OP_NUM_THREADS WEB_CONCURRENCY FASTAPI_THREADPOOL_SIZE

echo "[entrypoint] WEB_CONCURRENCY=$WEB_CONCURRENCY" \
             "ORT_INTRA=$ORT_INTRA_OP_NUM_THREADS" \
             "ORT_INTER=$ORT_INTER_OP_NUM_THREADS" \
             "OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS" \
             "TP=$FASTAPI_THREADPOOL_SIZE"

exec uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers "$WEB_CONCURRENCY" \
    --no-access-log
