# Model benchmark (no HTTP)

- Machine: Darwin arm64, Apple M5, 10 cores
- Cold start: session load 15 ms, first prediction 22 ms; peak RSS of this process 754 MB

## Per-stage latency, 1 thread, batch 1 (200 clips)

| Stage | p50 (ms) | p95 (ms) |
|---|---:|---:|
| Decode PCM | 0.04 | 0.04 |
| Log-mel (numpy) | 0.82 | 0.84 |
| Model (ONNX fp32) | 18.96 | 19.60 |

## ONNX threads per inference (batch 1)

| Threads | Model p50 (ms) |
|---:|---:|
| 1 | 19.04 |
| 2 | 12.74 |
| 4 | 9.85 |

## Batch size (1 thread)

| Batch | ms per batch | ms per clip |
|---:|---:|---:|
| 1 | 19.3 | 19.3 |
| 2 | 38.0 | 19.0 |
| 4 | 77.4 | 19.3 |
| 8 | 158.5 | 19.8 |
| 16 | 323.7 | 20.2 |

## fp32 vs int8 (dynamic quantization)

| Variant | Size (MB) | Model p50 (ms) | PR-AUC (1,000 clips) |
|---|---:|---:|---:|
| fp32 | 32.4 | 18.96 | 0.9837 |
| int8 dynamic | 8.4 | 19.98 | 0.9819 |

Max |fp32 - int8| probability difference: 0.416
