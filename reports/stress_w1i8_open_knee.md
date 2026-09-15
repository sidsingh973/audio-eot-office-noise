# Stress test: w1i8_open_knee

- Machine: Darwin arm64, Apple M5, 10 cores
- Server: 1 worker(s) x 1 ORT thread(s), 8 in-flight inference(s) per worker; model B (model.onnx), 1 ORT thread(s), 8 in-flight per worker
- Open loop, Poisson arrivals, 30 s per rate; payloads: 500 real clips (median 8.0 s, max 8.0 s) as 16-bit PCM

| Offered (req/s) | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 90 | 88 | 27.4 | 42.3 | 51.4 | 0.0 | 1.1 | 24.6 | 0 |
| 100 | 98 | 29.1 | 47.2 | 58.1 | 0.0 | 1.1 | 26.0 | 0 |
| 110 | 108 | 33.4 | 59.9 | 77.5 | 0.0 | 1.4 | 29.5 | 0 |
| 120 | 117 | 38.8 | 75.9 | 94.3 | 0.0 | 2.2 | 33.6 | 0 |

Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). The load generator runs on the same machine.
