# Audio end-of-turn detection in office noise

**Deliverables:**
- `presentation.ipynb`: the walkthrough, with live API tests.
- `SERVING.md`: API and performance details.
- `reports/lora_eval.md`: full evaluation.

## 1. Problem

Every time a caller pauses, a voice agent must decide whether the caller has **finished** (reply now) or is **pausing mid-thought** (keep listening). The two mistakes don't cost the same:
- **Interrupting** a caller is the costly error.
- **Replying late** is only slow: the agent falls back to a silence timeout.

So the model is used as a **policy**, not a bare classifier. Its decision threshold is the lowest one that keeps interruptions at **5% or below** on validation.

**Question:** can an audio end-of-turn model stay reliable with office background noise? Specifically, does training on noise help, and does WebRTC noise suppression help? Pipecat's open **Smart Turn v3.2** is the baseline.

## 2. Approach

**Data.** Pipecat's Smart Turn v3.2 dataset, English clips only:
- **Size:** 59,177 train / 6,625 validation / 7,820 test clips (a 90/10 split by clip-ID hash), 49% labelled *complete*.
- **Speech:** 34% synthetic (TTS), 66% human recordings.
- **Download:** the language filter meant streaming all 93 shards (46 GB).

**Noise.** Two IT-office recordings that aren't in Pipecat's noise library:
- `office-3` is used only for training and validation.
- `office-5` is used only for testing, at 20 / 10 / 5 / 0 dB, to test generalization to a noise the models never heard.
- SNR is measured against **active speech**. Otherwise complete turns, which end in silence, would get quieter noise than mid-turn clips, leaking the label.
- 80% of training clips get noise at a random 5–20 dB, stratified by label and source. The draws are seeded per clip, so B and C see identical noise.

**Three training conditions, one recipe:**

| Model | Training audio |
|---|---|
| A | native speech |
| B | + office-3 noise |
| C | B's audio after WebRTC noise suppression (LiveKit's WebRTC module: noise suppression only, 1 s warm-up, 6 ms delay compensated) |

**Model.** OpenAI's `whisper-tiny` encoder, **frozen**, with **LoRA adapters** (rank 16) on all 24 attention and MLP matrices, plus an attention-pooling classifier head (the same head design as Smart Turn).
- **Input:** the last 8 s of audio as an 80 × 800 log-mel spectrogram. **Output:** P(turn complete).
- **Size:** 657k trainable of 8.66M parameters (7.6%).

**Training:**
- 2 epochs, batch 64, AdamW (LoRA 3e-4, head 1e-3), cosine schedule.
- Always the final epoch, never "best epoch", so the three models are directly comparable.
- A trained on an Apple M5 GPU in 26 min; B and C on an RTX 3090 over SSH in 8 min each.

## 3. Results

All four models were scored on the same 7,820 test clips, each with its deployment input (C gets noise-suppressed audio).

| PR-AUC | native | office-5 20 dB | 10 dB | 5 dB | 0 dB |
|---|---:|---:|---:|---:|---:|
| Smart Turn v3.2 | 0.980 | 0.979 | 0.973 | 0.962 | 0.918 |
| A: native | **0.986** | **0.985** | 0.978 | 0.966 | 0.930 |
| **B: + office noise** | **0.986** | **0.985** | **0.981** | **0.972** | **0.947** |
| C: + office noise → noise suppression | 0.985 | 0.984 | 0.978 | 0.970 | 0.946 |

At 0 dB, the loudest noise level:

| Model | Turn ends answered quickly (≤5% cut-in) | Interruptions at the deployed threshold |
|---|---:|---:|
| Smart Turn v3.2 | 62% | 20.3% |
| **B** | **73%** | **9.9%** |

1. **Training on office noise works, even on a noise the model never heard.** B is best at every noise level. At 10 dB its 95% confidence interval [0.978, 0.984] doesn't overlap Smart Turn's [0.967, 0.977].
2. **It costs nothing on clean audio:** B equals A at 0.986.
3. **WebRTC noise suppression adds nothing.** C is within about 0.003 of B everywhere, because WebRTC removes only 6–13 dB of office noise. Skip that step.
4. **LoRA is enough.** A matches or beats Smart Turn on English (0.986 vs 0.980), with 7.6% of the weights trained and 22% of the data.

## 4. Serving

**API** (`service/app.py`, FastAPI):
- `POST /predict` takes the caller's current turn as raw 16 kHz 16-bit PCM or WAV.
- It returns the probability, the decision at the tuned threshold (0.774), and timings.
- It validates input: 400, 413, 415 or 422 on bad requests.

**Inference:** numpy log-mel plus ONNX Runtime, with no PyTorch.
- The LoRA is merged into a 32 MB ONNX model with the same interface as Smart Turn.
- The served model matches the evaluated one: **1 decision in 7,820 differs**, and PR-AUC is identical.

**Performance on an M5 laptop, CPU only:**
- **20.8 ms per request.** The model is 19 ms; HTTP and features add about 1 ms.
- **About 120 req/s sustained with p99 under 100 ms**, using one process with 8 inference slots.
  - Several uvicorn workers do worse (139 req/s peak, p99 around 300 ms), because uvicorn balances connections, not requests.
  - int8 and batching gave no gain.
- One such machine covers the estimated peak for 1M calls/month (about 60 req/s) with 2× headroom.

**Docker:** a 511 MB `python:3.12-slim` image that runs as a non-root user, with a health check.
- It gives identical outputs.
- On a Mac it is about 3.5× slower, because Docker Desktop's Linux VM can't use the M5's SME matrix instructions. So it should be benchmarked on the target Linux host.

**Testing:**
- `scripts/client.py` is the inference script.
- `scripts/stress_test.py` and `scripts/run_stress_grid.sh` are the load tests.
- The notebook runs live correctness, error-handling, accuracy and latency tests against the API.

## 5. Limitations and next steps

**Limitations:**
- Only two office recordings were used.
- A third of the speech is synthetic, and all of it is 16 kHz, so 8 kHz phone audio is untested.
- Each model was trained once, with one seed.
- Performance was measured on a laptop, with the load generator on the same machine.

**Next steps:**
- Start from Smart Turn's trained weights and add the office-noise LoRA on top.
- Evaluate on real call audio with more noise types.
- Replace per-request uploads with a streaming WebSocket API, or run the model in-process in the voice orchestrator.
- Fuse the audio model with the text model from the first project, which is aware of the dialogue context.

**Code:**
- `eot_lora/`: data download and preparation, noise and noise suppression, features, model, training, evaluation, export, serving.
- `service/`: the API.
- `scripts/`: client, parity check, benchmarks.
- `runs/{A,B,C}/`: checkpoints and ONNX models.

**Credits:** see `CREDITS.md`. It covers Pipecat's Smart Turn (code design, baseline model and data; BSD 2-Clause, © Daily), OpenAI's Whisper, the office recordings by thellywellyn on Pixabay, and every library used.
