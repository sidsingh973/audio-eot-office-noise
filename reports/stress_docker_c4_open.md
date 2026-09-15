# Stress test: docker_c4_open

- Machine: Darwin arm64, Apple M5, 10 cores
- Server: Docker Desktop (Linux arm64 VM), --cpus 4, 1 worker x 4 in flight; model B (model.onnx), 1 ORT thread(s), 4 in-flight per worker
- Open loop, Poisson arrivals, 30 s per rate; payloads: 500 real clips (median 8.0 s, max 8.0 s) as 16-bit PCM

| Offered (req/s) | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 19 | 76.6 | 91.0 | 109.0 | 0.0 | 1.5 | 72.0 | 0 |
| 30 | 29 | 80.1 | 124.7 | 155.8 | 0.0 | 1.6 | 75.5 | 0 |
| 40 | 39 | 101.7 | 225.3 | 285.7 | 13.5 | 1.7 | 81.8 | 0 |
| 50 | 45 | 1099.1 | 2371.6 | 2476.3 | 1009.8 | 2.0 | 83.9 | 0 |

Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). The load generator runs on the same machine.
