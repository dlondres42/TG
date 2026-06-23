.PHONY: help setup setup-model setup-fastapi setup-bentoml setup-bench \
        download train export verify smoke visualize clean-venvs \
        build-fastapi build-bentoml build-servers \
        run-fastapi run-bentoml stop-servers \
        calibrate-batching parity matrix-smoke \
        gen-targets bench-l1 bench-l2 bench-l3 bench-pilot analyze bench-all

PHASES := 1-model 2-serving/fastapi 2-serving/bentoml 3-bench

# Matrix knobs (override on the command line, e.g.  make run-fastapi WORKERS=2 THREADS=1).
WORKERS ?= 1
THREADS ?= 2
FASTAPI_PORT ?= 8001
BENTOML_PORT ?= 8002
ARTIFACTS := $(CURDIR)/project/1-model/artifacts

help:
	@echo "TG — P99 Tail Latency for Tabular ML Serving"
	@echo ""
	@echo "Setup:"
	@echo "  setup              install all four phase venvs via uv"
	@echo "  setup-model        install Phase 1 (training/export) deps"
	@echo "  setup-fastapi      install Phase 2A (FastAPI server) deps"
	@echo "  setup-bentoml      install Phase 2B (BentoML server) deps"
	@echo "  setup-bench        install Phase 3 (benchmark/analysis) deps"
	@echo ""
	@echo "Phase 1 — Model:"
	@echo "  smoke              train+export on synthetic data (fast, ~30s)"
	@echo "  download           download HIGGS dataset (~7.5 GB)"
	@echo "  train              train XGBoost on HIGGS"
	@echo "  export             export trained model to ONNX + parity check"
	@echo "  verify             re-run XGBoost<->ONNX parity check only"
	@echo "  visualize          serve model.onnx in Netron at http://127.0.0.1:8080"
	@echo ""
	@echo "Phase 2 — Serving (Docker):"
	@echo "  build-servers      build both serving images"
	@echo "  build-fastapi      build Phase 2A image"
	@echo "  build-bentoml      build Phase 2B image"
	@echo "  run-fastapi        docker run FastAPI (override WORKERS, THREADS)"
	@echo "  run-bentoml        docker run BentoML  (override WORKERS, THREADS)"
	@echo "  stop-servers       stop and remove both serving containers"
	@echo "  calibrate-batching run BentoML batching grid -> batching.json"
	@echo "  parity             3-way parity (FastAPI vs BentoML vs local ONNX)"
	@echo "  matrix-smoke       boot all 4 (workers x threads) cells for each pipeline"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean-venvs        remove all .venv directories"

setup: setup-model setup-fastapi setup-bentoml setup-bench

setup-model:
	cd project/1-model && uv sync

setup-fastapi:
	cd project/2-serving/fastapi && uv sync

setup-bentoml:
	cd project/2-serving/bentoml && uv sync

setup-bench:
	cd project/3-bench && uv sync

download:
	cd project/1-model && uv run python download.py

train:
	cd project/1-model && uv run python train.py

export:
	cd project/1-model && uv run python export.py

verify:
	cd project/1-model && uv run python export.py --verify-only

smoke:
	cd project/1-model && uv run python download.py --synthetic 50000
	cd project/1-model && uv run python train.py --n-trees 50
	cd project/1-model && uv run python export.py

visualize:
	cd project/1-model && uv run --with netron netron artifacts/model.onnx --host 127.0.0.1 --port 8080

# Phase 2 — Serving
build-servers: build-fastapi build-bentoml

build-fastapi:
	docker build -t tg-serving-fastapi project/2-serving/fastapi

build-bentoml:
	docker build -t tg-serving-bentoml project/2-serving/bentoml

run-fastapi:
	-docker rm -f tg-fa 2>/dev/null
	docker run --rm -d --name tg-fa \
		--cpus=2 --memory=2g --cpuset-cpus=0,1 \
		-v "$(ARTIFACTS):/app:ro" \
		-e WEB_CONCURRENCY=$(WORKERS) \
		-e ORT_INTRA_OP_NUM_THREADS=$(THREADS) \
		-e ORT_INTER_OP_NUM_THREADS=1 \
		-e OMP_NUM_THREADS=$(THREADS) \
		-e MKL_NUM_THREADS=$(THREADS) \
		-p $(FASTAPI_PORT):8000 \
		tg-serving-fastapi

run-bentoml:
	-docker rm -f tg-bm 2>/dev/null
	docker run --rm -d --name tg-bm \
		--cpus=2 --memory=2g --cpuset-cpus=2,3 \
		-v "$(ARTIFACTS):/app:ro" \
		-e WEB_CONCURRENCY=$(WORKERS) \
		-e ORT_INTRA_OP_NUM_THREADS=$(THREADS) \
		-e ORT_INTER_OP_NUM_THREADS=1 \
		-e OMP_NUM_THREADS=$(THREADS) \
		-e MKL_NUM_THREADS=$(THREADS) \
		-p $(BENTOML_PORT):8000 \
		tg-serving-bentoml

stop-servers:
	-docker rm -f tg-fa tg-bm 2>/dev/null

calibrate-batching:
	cd project/2-serving/bentoml && uv run python calibrate_batching.py

parity:
	cd project/3-bench && uv run python parity.py \
		--fastapi http://127.0.0.1:$(FASTAPI_PORT) \
		--bentoml http://127.0.0.1:$(BENTOML_PORT) \
		--n 500

matrix-smoke:
	cd project/3-bench && uv run python matrix_smoke.py

# Phase 3 — Benchmark
gen-targets:
	cd project/3-bench && uv run python targets/generate_targets.py --n 1000

bench-l1:
	cd project/3-bench && uv run python layers/L1_inference_floor.py

bench-l2:
	cd project/3-bench && uv run python layers/L2_host_http.py

bench-l3:
	cd project/3-bench && uv run python layers/L3_docker_sweep.py

bench-pilot:
	cd project/3-bench && uv run python layers/L3_docker_sweep.py --pilot

analyze:
	cd project/3-bench && uv run python analyze.py

# Full Phase 3: targets -> inference floor -> host HTTP -> matrix sweep -> analysis
bench-all: gen-targets bench-l1 bench-l2 bench-l3 analyze

clean-venvs:
	rm -rf project/1-model/.venv project/2-serving/fastapi/.venv \
	       project/2-serving/bentoml/.venv project/3-bench/.venv
