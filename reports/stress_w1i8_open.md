# Stress test: w1i8_open

- Machine: Darwin arm64, Apple M5, 10 cores
- Server: 1 worker(s) x 1 ORT thread(s), 8 in-flight inference(s) per worker; model B (model.onnx), 1 ORT thread(s), 8 in-flight per worker
- Open loop, Poisson arrivals, 30 s per rate; payloads: 500 real clips (median 8.0 s, max 8.0 s) as 16-bit PCM

| Offered (req/s) | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 41 | 23.1 | 31.0 | 34.3 | 0.0 | 1.0 | 20.8 | 0 |
| 84 | 83 | 26.8 | 39.3 | 47.4 | 0.0 | 1.1 | 23.9 | 0 |
| 126 | 123 | 37.7 | 216.4 | 655.0 | 0.0 | 2.1 | 32.2 | 0 |
| 151 | 126 | 1121.0 | 8054.0 | 12401.9 | 0.0 | 2.0 | 31.7 | 0 |
| 168 | 127 | 2093.4 | 15520.8 | 20205.0 | 0.0 | 2.3 | 33.9 | 0 |

Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). The load generator runs on the same machine.
