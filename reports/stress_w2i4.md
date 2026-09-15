# Stress test: w2i4

- Machine: Darwin arm64, Apple M5, 10 cores
- Server: 2 worker(s) x 1 ORT thread(s), 4 in-flight inference(s) per worker; model B (model.onnx), 1 ORT thread(s), 4 in-flight per worker
- Closed loop, 1000 requests per level; payloads: 500 real clips (median 8.0 s, max 8.0 s) as 16-bit PCM

| Clients | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49 | 20.4 | 20.9 | 23.6 | 0.0 | 0.9 | 18.6 | 0 |
| 2 | 79 | 25.3 | 26.4 | 33.6 | 0.0 | 1.1 | 22.7 | 0 |
| 4 | 129 | 30.9 | 33.7 | 42.9 | 0.0 | 1.2 | 28.3 | 0 |
| 8 | 148 | 57.1 | 67.7 | 79.0 | 22.3 | 2.3 | 30.4 | 0 |
| 16 | 162 | 126.1 | 152.3 | 160.5 | 83.5 | 2.6 | 38.8 | 0 |
| 32 | 164 | 143.6 | 317.4 | 330.9 | 71.8 | 3.1 | 44.1 | 0 |

Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). The load generator runs on the same machine.
