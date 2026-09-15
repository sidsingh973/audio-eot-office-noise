# Stress test: docker_c4

- Machine: Darwin arm64, Apple M5, 10 cores
- Server: Docker Desktop (Linux arm64 VM), --cpus 4, 1 worker x 4 in flight; model B (model.onnx), 1 ORT thread(s), 4 in-flight per worker
- Closed loop, 500 requests per level; payloads: 500 real clips (median 8.0 s, max 8.0 s) as 16-bit PCM

| Clients | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14 | 72.8 | 74.1 | 82.3 | 0.0 | 1.6 | 68.7 | 0 |
| 2 | 26 | 75.8 | 89.8 | 102.5 | 0.0 | 1.6 | 70.5 | 0 |
| 4 | 46 | 85.7 | 93.8 | 119.4 | 0.0 | 1.9 | 79.4 | 0 |
| 8 | 47 | 167.3 | 174.5 | 180.9 | 81.5 | 2.0 | 81.2 | 0 |
| 16 | 47 | 337.8 | 361.0 | 378.3 | 250.6 | 2.1 | 81.6 | 0 |

Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). The load generator runs on the same machine.
