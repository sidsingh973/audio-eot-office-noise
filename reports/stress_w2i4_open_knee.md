# Stress test: w2i4_open_knee

- Machine: Darwin arm64, Apple M5, 10 cores
- Server: 2 worker(s) x 1 ORT thread(s), 4 in-flight inference(s) per worker; model B (model.onnx), 1 ORT thread(s), 4 in-flight per worker
- Open loop, Poisson arrivals, 30 s per rate; payloads: 500 real clips (median 8.0 s, max 8.0 s) as 16-bit PCM

| Offered (req/s) | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 90 | 88 | 30.9 | 51.3 | 67.6 | 0.0 | 1.2 | 27.4 | 0 |
| 100 | 98 | 32.7 | 59.8 | 81.7 | 0.0 | 1.3 | 28.9 | 0 |
| 110 | 108 | 35.1 | 76.1 | 99.4 | 0.0 | 1.4 | 30.3 | 0 |
| 120 | 117 | 38.4 | 82.3 | 115.0 | 0.0 | 1.8 | 31.8 | 0 |

Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). The load generator runs on the same machine.
