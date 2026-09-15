# Serving the audio end-of-turn model

Model **B** is Whisper-tiny with LoRA, trained on English speech with office noise added. It was the best model in `reports/lora_eval.md` and is served here behind a FastAPI endpoint.

- **Format:** fp32 ONNX, 32.4 MB. Input is 8 s of 16 kHz audio; output is P(turn complete).
- **Decision threshold:** **0.774**, the lowest threshold that keeps interruptions at 5% or below on B's validation set.
- **Serving stack:** numpy and ONNX Runtime only, with no PyTorch. The log-mel code reproduces the training featurizer exactly (see parity below).

## Findings in brief

Measured on an Apple M5 laptop, CPU only.

- **Latency:** **20.8 ms p50** end to end for one request, of which **19 ms is the model**. HTTP, the 256 KB body and the log-mel features add about 1 ms. Giving one inference 4 ONNX Runtime threads brings this down to **11.7 ms**.
- **Throughput:** 167 req/s at peak. **About 120 req/s is sustainable with p99 under 100 ms**, using 1 uvicorn worker running 8 inferences at once (about 470 MB of memory). Past that, latency climbs sharply.
- **Server configuration matters more than the hardware.** Eight separate uvicorn workers reached only 139 req/s, with p99 around 300 ms, because uvicorn balances connections across workers, not individual requests. One process with 8 inference slots balances each request.
- **Serve fp32 without batching.** Dynamic int8 was slower (20.0 ms) and less accurate. Batching gave no speed-up per clip on CPU.
- **Sizing:** one such machine covers about 60 req/s at peak with 2× headroom. That's the estimate for 1M calls/month, assuming about 30 end-of-turn checks per call.
- **Docker:** the image works. It gives identical answers, uses 149 MB of memory at idle and is healthy 2 s after starting. On this Mac, though, the container is about 3.5× slower (47 req/s peak with 4 CPUs), because Docker Desktop's Linux VM can't use the M5's matrix instructions. Benchmark the image on the Linux machine it will run on.

## API

| Endpoint | |
|---|---|
| `POST /predict` | Body: the audio of the user's current turn, as raw **16-bit PCM, 16 kHz mono** (`Content-Type: application/octet-stream`) or a 16 kHz 16-bit **WAV** file (`audio/wav`). Only the last 8 s are used. Optional `?threshold=0.9` overrides the default. |
| `GET /health` | Model, threshold, and threading config |
| `GET /docs` | Interactive OpenAPI docs |

```bash
curl -s localhost:8000/predict -H 'content-type: audio/wav' --data-binary @clip.wav
# {"eot_probability":0.996,"is_end_of_turn":true,"threshold":0.774,"audio_seconds":3.73,
#  "queue_ms":0.002,"features_ms":0.95,"model_ms":19.7,"latency_ms":20.7}      (real response, rounded)
```

`latency_ms` is compute time inside the server (features + model). `queue_ms` is the time spent waiting for a free inference slot.

Errors:

| Code | Cause |
|---|---|
| 400 | Empty body |
| 413 | More than 30 s of audio |
| 415 | Other content types |
| 422 | Unreadable audio, or a WAV that isn't 16 kHz / 16-bit |

## Run it

```bash
EOT_MAX_INFLIGHT=8 uvicorn service.app:app --port 8000     # API, throughput config
python scripts/client.py --test-set test_office5_10 --n 20 # inference script: labelled clips -> decisions
python scripts/client.py clip.wav                          # or your own WAV files
python scripts/check_parity.py                             # served model == evaluated model
bash scripts/run_stress_grid.sh                            # model benchmark + stress grid -> reports/
```

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `EOT_MODEL_DIR` | `runs/B` | Model directory |
| `EOT_MODEL_FILE` | `model.onnx` | Model file |
| `EOT_THREADS` | 1 | ONNX Runtime threads per inference |
| `EOT_MAX_INFLIGHT` | 1 | Inferences running at once per process; set it to about the number of CPUs |

For the lowest single-request latency instead of throughput, use `EOT_THREADS=4 EOT_MAX_INFLIGHT=1`.

**Docker:**

```bash
docker build -t eot-audio .                               # bakes in runs/B; --build-arg MODEL_DIR=runs/C for another
docker run --rm -p 8000:8000 --cpus 4 -e EOT_MAX_INFLIGHT=4 eot-audio
```

The image is `python:3.12-slim` plus fastapi, uvicorn, numpy and onnxruntime (`requirements-serve.txt`). It runs as a non-root user with a health check. Match `EOT_MAX_INFLIGHT` to the CPUs you give the container; `os.cpu_count()` inside a container reports the host's cores, so the value has to be set explicitly.

## Design choices

- **Raw PCM over HTTP.** It's the simplest thing a voice pipeline can send, with no decoding dependencies on the server. The server uses only the last 8 s (up to 256 KB), so truncating on the client side saves bandwidth.
- **Single-threaded inferences, several per process.** ONNX Runtime releases Python's global lock, so one process runs `EOT_MAX_INFLIGHT` inferences in parallel. A semaphore caps them, so extra requests queue instead of fighting over cores. `OMP_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `OPENBLAS_NUM_THREADS` are all set to 1, so numpy doesn't start extra threads either.
- **Torch-free featurizer.** Whisper's log-mel is an STFT, a mel filterbank and a log clamp: about 20 lines of numpy (`eot_lora/serving.py`). This keeps the image small and starts in about 40 ms (15 ms to load the session, 22 ms for the first prediction).

## Parity: the served model matches the evaluated one

`scripts/check_parity.py`, run on all 7,820 office-5 10 dB test clips:

| Check | Result |
|---|---|
| numpy log-mel vs the training featurizer (`WhisperFeatureExtractor`) | max difference 4.1e-5 |
| Served probabilities vs the evaluation scores | max difference 5.0e-3, mean 5.6e-5 |
| Decisions that flip at the 0.774 threshold | 1 of 7,820 |
| PR-AUC, served vs evaluated | 0.9806 vs 0.9806 |

The remaining differences come from the evaluation having stored features in fp16. The inference script on 20 clips: 19 correct and 0 interruptions on the 10 mid-turn clips.

## Performance on this machine

Setup:
- **Hardware:** Apple M5 laptop, 10 cores (4 performance + 6 efficiency), 24 GB, macOS.
- **Software:** Python 3.12, ONNX Runtime 1.30 (CPU), fp32 model.
- **Payload:** 500 real test clips (office-5 noise at 10 dB), sent as 16-bit PCM. The median clip is 8.0 s, the full window, so a typical request is 256 KB.
- **Caveat:** the load generator runs on the same machine and uses about one core.

Raw reports: `reports/bench_model.md` and `reports/stress_*.md`.

### Model only, without HTTP (`scripts/bench_model.py`)

| Stage | p50 |
|---|---:|
| Decode PCM | 0.04 ms |
| Log-mel (numpy) | 0.82 ms |
| **ONNX model, 1 thread** | **19.0 ms** |
| ONNX model, 2 / 4 threads | 12.7 / 9.9 ms |

| Batch size | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| ms per clip | 19.3 | 19.0 | 19.3 | 19.8 | 20.2 |

| Variant | Size | Model p50 | PR-AUC (1,000 clips) |
|---|---:|---:|---:|
| **fp32 (served)** | 32.4 MB | **19.0 ms** | **0.9837** |
| int8, dynamic | 8.4 MB | 20.0 ms | 0.9819 (single probabilities shift by up to 0.42) |

### Through the API: peak throughput (`scripts/stress_test.py`, closed loop)

Each of *C* clients sends its next request as soon as the previous one returns. There are 1,000 requests per level.

| Server config | p50, 1 client | Peak throughput | p99 at peak | Memory |
|---|---:|---:|---:|---:|
| 1 worker, 1 inference at a time | 20.8 ms | 50 req/s | 41 ms (2 clients) | |
| 1 worker, 1 inference using 4 ORT threads | **11.7 ms** | 91 req/s | 24 ms (2 clients) | |
| 4 workers × 1 | 21.1 ms | 114 req/s | 350 ms (16 clients) | |
| 8 workers × 1 | 20.6 ms | 139 req/s | 291 ms (16 clients) | |
| 1 worker, 4 inferences in flight | 20.8 ms | 123 req/s | **36 ms** (4 clients) | 331 MB |
| **1 worker, 8 inferences in flight** | 20.6 ms | **167 req/s** | 58 ms (8 clients) | 468 MB |
| 2 workers, 4 inferences in flight each | 20.4 ms | 164 req/s | 331 ms (32 clients) | 649 MB |

### Sustained load: capacity under a 100 ms p99 (open loop)

Requests arrive at a fixed average rate (Poisson), whether or not the server keeps up. This is how independent calls behave, and it shows queueing that the closed-loop test hides. 30 s per rate:

| Arrival rate | 1 worker × 8 in flight: p50 / p95 / p99 | 2 workers × 4 in flight: p50 / p95 / p99 |
|---:|---:|---:|
| 42/s | 23 / 31 / 34 ms | |
| 84/s | 27 / 39 / 47 ms | |
| 90/s | 27 / 42 / 51 ms | 31 / 51 / 68 ms |
| 100/s | 29 / 47 / 58 ms | 33 / 60 / 82 ms |
| 110/s | 33 / 60 / 78 ms | 35 / 76 / 99 ms |
| 120/s | 39 / 76 / **94 ms** | 38 / 82 / 115 ms |
| 126/s | 38 / 216 / 655 ms | |
| 151/s | overloaded: serves 126/s, p99 12 s | |

- **Capacity:** about **120 req/s at p99 under 100 ms**. Past that, the server saturates abruptly, so plan for about 100 req/s per machine to keep headroom.
- **Where requests wait:** the in-flight queue stays empty at every rate. The waiting happens before the handler starts: one event loop reads 256 KB request bodies while competing for Python's global lock with 8 inference threads. A streaming API (below) would remove most of this.

### In Docker (Docker Desktop on this Mac)

- **Image:** `eot-audio` is 511 MB in total: the `python:3.12-slim` base, 179 MB of Python packages and the 32.5 MB model. With the base image cached it builds in about 10 s.
- **Startup and memory:** the container passes its health check 2 s after starting and uses 149 MB of memory at idle.
- **Same answers:** all 20 samples get exactly the same probabilities as the server running directly on the Mac.
- **About 3.5× slower per request:** the model takes 72.9 ms in the container against 21.1 ms directly on the Mac, and features 2.7 ms against 1.1 ms.
  - The cause is that Docker Desktop runs containers inside a Linux virtual machine.
  - That VM doesn't pass through the M5's SME/SME2 matrix instructions. They're present on the Mac but missing from `/proc/cpuinfo` inside the container, so ONNX Runtime falls back to slower general-purpose (NEON) kernels.
  - A Linux server has no VM in the way. **Use the Docker numbers below to confirm the image works, not to size production.**

Container run with `--cpus 4 -e EOT_MAX_INFLIGHT=4` (`reports/stress_docker_c4*.md`):

| Load | Throughput | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| 1 client (closed loop) | 14 req/s | 73 ms | 74 ms | 82 ms |
| 4 clients (closed loop) | 46 req/s | 86 ms | 94 ms | 119 ms |
| 16 clients (closed loop, saturated) | 47 req/s | 338 ms | 361 ms | 378 ms |
| 20 req/s arriving (open loop) | 19 req/s | 77 ms | 91 ms | 109 ms |
| 40 req/s arriving (open loop) | 39 req/s | 102 ms | 225 ms | 286 ms |
| 50 req/s arriving (open loop) | 45 req/s | 1.1 s | 2.4 s | 2.5 s (overloaded) |

In Docker on this Mac, the container peaks at about 47 req/s. A single request already takes about 73 ms, so p99 can't stay under 100 ms. That's the missing matrix instructions in the VM, not the image.

## Caveats

- **This is a laptop.** The load generator shares the CPU. The M5 has only 4 performance cores, and beyond about 4 concurrent inferences the model time rises from 19 ms to 30–45 ms as work spills onto the efficiency cores. A Linux x86 server will behave differently. Rerun `bash scripts/run_stress_grid.sh` on the target machine before sizing.
- **uvicorn's `--workers` balances per connection, not per request.** With a few keep-alive clients, some workers sit idle: 6 workers with 4 clients gave only 50 req/s. Behind a load balancer with many connections this evens out. A single process with several inference slots balances per request.
- **Each check uploads up to 256 KB.** That's fine inside a cluster, but across regions the upload time adds directly to latency.
- **Docker on a Mac is slower than the Mac itself.** Containers there can't use the M5's matrix instructions (see above). Benchmark the image on the machine it will run on.

## Next steps

- **Streaming API.** A WebSocket takes the call's 20 ms audio frames into a buffer on the server, and each check sends only a "pause" event instead of re-uploading 8 s of audio.
- **Run in-process** in the voice orchestrator: the ONNX model with no network hop, removing HTTP and queueing entirely.
- **More throughput per machine:** the GPU or CoreML execution providers, or static (calibrated) int8 instead of dynamic.
- **Autoscale** on p99 latency and busy inference slots, and alert when p99 goes above 100 ms.
