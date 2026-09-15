# Stress test: w1i4

- Machine: Darwin arm64, Apple M5, 10 cores
- Server: 1 worker(s) x 1 ORT thread(s), 4 in-flight inference(s) per worker; model B (model.onnx), 1 ORT thread(s), 4 in-flight per worker
- Closed loop, 1000 requests per level; payloads: 500 real clips (median 8.0 s, max 8.0 s) as 16-bit PCM

| Clients | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 20.8 | 21.2 | 22.0 | 0.0 | 0.9 | 18.9 | 0 |
| 2 | 80 | 25.1 | 26.1 | 26.4 | 0.0 | 1.0 | 22.6 | 0 |
| 4 | 121 | 33.1 | 35.1 | 35.8 | 0.0 | 1.3 | 30.2 | 0 |
| 8 | 123 | 64.9 | 68.8 | 69.6 | 30.5 | 1.5 | 30.6 | 0 |
| 16 | 122 | 131.7 | 136.7 | 138.8 | 96.9 | 1.5 | 31.2 | 0 |
| 32 | 124 | 259.9 | 267.9 | 270.0 | 225.3 | 1.4 | 30.7 | 0 |

Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). The load generator runs on the same machine.
