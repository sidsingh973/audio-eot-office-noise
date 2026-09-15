# Stress test: w1i8

- Machine: Darwin arm64, Apple M5, 10 cores
- Server: 1 worker(s) x 1 ORT thread(s), 8 in-flight inference(s) per worker; model B (model.onnx), 1 ORT thread(s), 8 in-flight per worker
- Closed loop, 1000 requests per level; payloads: 500 real clips (median 8.0 s, max 8.0 s) as 16-bit PCM

| Clients | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49 | 20.6 | 20.7 | 21.0 | 0.0 | 0.9 | 18.7 | 0 |
| 2 | 80 | 24.9 | 25.6 | 25.8 | 0.0 | 1.0 | 22.4 | 0 |
| 4 | 120 | 33.3 | 35.3 | 36.3 | 0.0 | 1.2 | 30.5 | 0 |
| 8 | 168 | 47.6 | 54.7 | 58.4 | 0.0 | 2.8 | 42.5 | 0 |
| 16 | 167 | 95.6 | 105.2 | 110.6 | 45.4 | 3.0 | 44.3 | 0 |
| 32 | 165 | 193.0 | 205.1 | 215.4 | 142.5 | 3.0 | 44.8 | 0 |

Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). The load generator runs on the same machine.
