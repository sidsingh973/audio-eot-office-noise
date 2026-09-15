# Stress test: w8

- Machine: Darwin arm64, Apple M5, 10 cores
- Server: 8 worker(s) x 1 ORT thread(s), 1 in-flight inference per worker; model B (model.onnx), 1 ORT thread(s), 1 in-flight per worker
- Closed loop, 1000 requests per level; payloads: 500 real clips (median 8.0 s, max 8.0 s) as 16-bit PCM

| Clients | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 20.6 | 20.8 | 21.3 | 0.0 | 0.9 | 18.8 | 0 |
| 2 | 50 | 39.8 | 40.0 | 40.1 | 19.1 | 0.9 | 18.8 | 0 |
| 4 | 89 | 64.0 | 68.0 | 69.4 | 41.7 | 1.0 | 20.6 | 0 |
| 8 | 127 | 58.6 | 126.7 | 130.2 | 27.4 | 1.3 | 29.0 | 0 |
| 16 | 139 | 70.5 | 286.4 | 290.6 | 33.6 | 2.3 | 32.3 | 0 |
| 32 | 137 | 106.5 | 537.6 | 546.3 | 69.4 | 2.3 | 32.5 | 0 |

Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). The load generator runs on the same machine.
