# BentoML Adaptive Batching Calibration

Target load: 100 RPS open-loop, 15s per cell, single-row requests.
Fixed parallelism: WEB_CONCURRENCY=1, ORT_INTRA_OP_NUM_THREADS=2 (the most batching-friendly cell of the Phase 2 matrix).

| max_batch_size | max_latency_ms | N | errors | P50 ms | P95 ms | P99 ms | P99.9 ms | achieved RPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 2 | 1450 | 47 | 9.23 | 13.05 | 19.82 | 74.05 | 96.7 |
| 16 | 5 | 1499 | 0 | 9.47 | 13.38 | 35.45 | 82.02 | 99.9 | **(picked)**
| 16 | 10 | 1497 | 2 | 9.08 | 13.36 | 43.31 | 81.02 | 99.7 |
| 32 | 2 | 1466 | 33 | 9.03 | 13.30 | 16.38 | 76.73 | 97.7 |
| 32 | 5 | 1494 | 5 | 9.35 | 13.49 | 42.94 | 87.09 | 99.6 |
| 32 | 10 | 1498 | 1 | 9.43 | 13.02 | 42.46 | 83.81 | 99.8 |
| 64 | 2 | 1457 | 42 | 8.99 | 12.64 | 17.10 | 95.11 | 97.1 |
| 64 | 5 | 1497 | 2 | 9.22 | 12.96 | 43.77 | 81.27 | 99.7 |
| 64 | 10 | 1498 | 1 | 9.59 | 15.50 | 44.71 | 88.45 | 99.9 |

**Picked**: max_batch_size=16, max_latency_ms=5 (lowest P99 with zero errors; tie-break on P50).

This pair is written into [batching.json](batching.json) and held fixed across the Phase 3 worker × ORT-thread matrix.
