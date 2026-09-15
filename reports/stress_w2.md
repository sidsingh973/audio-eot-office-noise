# Stress test: w2

- Machine: Darwin arm64, Apple M5, 10 cores
- Server: 2 worker(s) x 1 ORT thread(s), 1 in-flight inference per worker; model B (model.onnx), 1 ORT thread(s), 1 in-flight per worker
- Closed loop, 1000 requests per level; payloads: 500 real clips (median 8.0 s, max 8.0 s) as 16-bit PCM

| Clients | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 21.0 | 21.2 | 21.4 | 0.0 | 0.9 | 19.1 | 0 |
| 2 | 50 | 40.4 | 40.7 | 41.5 | 19.3 | 1.0 | 19.0 | 0 |
| 4 | 50 | 80.9 | 81.3 | 82.3 | 59.8 | 1.0 | 19.0 | 0 |
| 8 | 85 | 152.5 | 166.4 | 169.3 | 129.4 | 1.0 | 21.7 | 0 |
| 16 | 85 | 124.9 | 263.8 | 266.2 | 100.3 | 1.1 | 22.0 | 0 |
| 32 | 83 | 319.9 | 461.6 | 467.1 | 294.8 | 1.1 | 22.6 | 0 |

Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). The load generator runs on the same machine.
