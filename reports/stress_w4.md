# Stress test: w4

- Machine: Darwin arm64, Apple M5, 10 cores
- Server: 4 worker(s) x 1 ORT thread(s), 1 in-flight inference per worker; model B (model.onnx), 1 ORT thread(s), 1 in-flight per worker
- Closed loop, 1000 requests per level; payloads: 500 real clips (median 8.0 s, max 8.0 s) as 16-bit PCM

| Clients | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 47 | 21.1 | 22.4 | 24.3 | 0.0 | 0.9 | 19.2 | 0 |
| 2 | 48 | 40.8 | 43.5 | 45.6 | 19.5 | 1.0 | 19.2 | 0 |
| 4 | 83 | 68.2 | 73.7 | 76.9 | 44.2 | 1.0 | 22.3 | 0 |
| 8 | 82 | 64.5 | 152.2 | 156.7 | 37.2 | 1.1 | 22.8 | 0 |
| 16 | 114 | 129.0 | 345.6 | 350.4 | 94.6 | 1.4 | 32.0 | 0 |
| 32 | 112 | 205.6 | 545.9 | 633.6 | 142.9 | 1.7 | 31.9 | 0 |

Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). The load generator runs on the same machine.
