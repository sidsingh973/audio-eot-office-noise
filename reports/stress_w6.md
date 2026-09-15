# Stress test: w6

- Machine: Darwin arm64, Apple M5, 10 cores
- Server: 6 worker(s) x 1 ORT thread(s), 1 in-flight inference per worker; model B (model.onnx), 1 ORT thread(s), 1 in-flight per worker
- Closed loop, 1000 requests per level; payloads: 500 real clips (median 8.0 s, max 8.0 s) as 16-bit PCM

| Clients | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 20.6 | 20.8 | 21.1 | 0.0 | 0.9 | 18.8 | 0 |
| 2 | 50 | 39.8 | 40.0 | 40.2 | 19.1 | 0.9 | 18.7 | 0 |
| 4 | 50 | 79.6 | 79.8 | 80.3 | 58.9 | 0.9 | 18.7 | 0 |
| 8 | 116 | 26.5 | 154.3 | 156.7 | 0.0 | 1.1 | 23.6 | 0 |
| 16 | 129 | 86.2 | 310.4 | 315.7 | 55.6 | 1.5 | 28.5 | 0 |
| 32 | 124 | 155.9 | 661.4 | 669.3 | 123.3 | 1.5 | 29.1 | 0 |

Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). The load generator runs on the same machine.
