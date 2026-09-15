# Stress test: w1

- Machine: Darwin arm64, Apple M5, 10 cores
- Server: 1 worker(s) x 1 ORT thread(s), 1 in-flight inference per worker; model B (model.onnx), 1 ORT thread(s), 1 in-flight per worker
- Closed loop, 1000 requests per level; payloads: 500 real clips (median 8.0 s, max 8.0 s) as 16-bit PCM

| Clients | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48 | 20.8 | 21.1 | 21.4 | 0.0 | 0.9 | 18.9 | 0 |
| 2 | 50 | 40.4 | 40.6 | 40.9 | 19.3 | 1.0 | 19.0 | 0 |
| 4 | 50 | 80.8 | 81.3 | 81.6 | 59.8 | 1.0 | 19.0 | 0 |
| 8 | 50 | 161.5 | 162.4 | 163.2 | 140.4 | 1.0 | 19.0 | 0 |
| 16 | 50 | 323.3 | 324.2 | 324.7 | 302.3 | 1.0 | 19.0 | 0 |
| 32 | 50 | 647.3 | 649.7 | 651.8 | 626.1 | 1.0 | 19.0 | 0 |

Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). The load generator runs on the same machine.
