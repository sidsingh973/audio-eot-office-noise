# Stress test: w1t4

- Machine: Darwin arm64, Apple M5, 10 cores
- Server: 1 worker(s) x 4 ORT thread(s), 1 in-flight inference per worker; model B (model.onnx), 4 ORT thread(s), 1 in-flight per worker
- Closed loop, 1000 requests per level; payloads: 500 real clips (median 8.0 s, max 8.0 s) as 16-bit PCM

| Clients | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 85 | 11.7 | 12.0 | 12.2 | 0.0 | 0.9 | 9.8 | 0 |
| 2 | 90 | 22.1 | 22.5 | 24.1 | 10.2 | 1.0 | 9.9 | 0 |
| 4 | 91 | 44.1 | 44.5 | 44.7 | 32.2 | 1.0 | 9.8 | 0 |
| 8 | 91 | 87.8 | 88.4 | 89.0 | 76.0 | 1.0 | 9.8 | 0 |
| 16 | 92 | 174.7 | 175.5 | 175.7 | 163.0 | 1.0 | 9.7 | 0 |
| 32 | 92 | 348.1 | 349.4 | 350.6 | 336.4 | 1.0 | 9.7 | 0 |

Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). The load generator runs on the same machine.
