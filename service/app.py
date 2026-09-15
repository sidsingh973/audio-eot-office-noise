"""FastAPI service for the audio end-of-turn model.

Run:  EOT_MAX_INFLIGHT=8 uvicorn service.app:app --port 8000
Env:  EOT_MODEL_DIR (default runs/B), EOT_MODEL_FILE (default from serving.json),
      EOT_THREADS (ONNX Runtime threads per inference, default 1),
      EOT_MAX_INFLIGHT (inferences running at once per process, default 1; set to about the CPU count)

The voice pipeline calls POST /predict when the user goes silent, with the audio of the user's
current turn. Body: raw little-endian 16-bit PCM, 16 kHz mono (Content-Type: application/octet-stream),
or a 16 kHz 16-bit WAV file (Content-Type: audio/wav). Only the last 8 s are used.
"""
from __future__ import annotations

import os

# One single-threaded inference per core; stop numpy/BLAS from adding threads of their own.
for _var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import asyncio  # noqa: E402
import time  # noqa: E402
import wave  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from eot_lora.serving import SR, WINDOW, AudioEoTPredictor, pcm16_to_float, read_wav  # noqa: E402

MAX_BYTES = 30 * SR * 2  # 30 s of 16-bit audio; only the last 8 s reach the model
WAV_TYPES = {"audio/wav", "audio/x-wav", "audio/wave"}
PCM_TYPES = {"application/octet-stream", "audio/pcm", "audio/l16"}
THREADS = int(os.getenv("EOT_THREADS", "1"))
MAX_INFLIGHT = int(os.getenv("EOT_MAX_INFLIGHT", "1"))


class PredictResponse(BaseModel):
    eot_probability: float
    is_end_of_turn: bool
    threshold: float
    audio_seconds: float
    queue_ms: float = 0.0
    features_ms: float
    model_ms: float
    latency_ms: float


predictor: AudioEoTPredictor | None = None
slots: asyncio.Semaphore | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global predictor, slots
    predictor = AudioEoTPredictor(os.getenv("EOT_MODEL_DIR", "runs/B"),
                                  model_file=os.getenv("EOT_MODEL_FILE") or None, intra_op_threads=THREADS)
    predictor.predict(np.zeros(SR, np.float32))  # first call pays one-off allocation costs
    # Excess requests wait here instead of all running at once and fighting over cores.
    slots = asyncio.Semaphore(MAX_INFLIGHT)
    yield


app = FastAPI(title="Audio End-of-Turn Detector", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_dir": str(predictor.model_dir), "model_file": predictor.model_file,
            "condition": predictor.config.get("condition"), "threshold": predictor.threshold,
            "sample_rate": SR, "window_seconds": WINDOW // SR, "ort_threads": THREADS,
            "max_inflight_per_worker": MAX_INFLIGHT}


def infer(audio: np.ndarray, threshold: float | None) -> PredictResponse:
    t0 = time.perf_counter()
    feats = predictor.features(audio)[None]
    t1 = time.perf_counter()
    p = float(predictor.predict_features(feats)[0])
    t2 = time.perf_counter()
    thr = threshold if threshold is not None else predictor.threshold
    return PredictResponse(eot_probability=p, is_end_of_turn=p >= thr, threshold=thr,
                           audio_seconds=len(audio) / SR, features_ms=(t1 - t0) * 1000,
                           model_ms=(t2 - t1) * 1000, latency_ms=(t2 - t0) * 1000)


@app.post("/predict", response_model=PredictResponse)
async def predict(request: Request,
                  threshold: float | None = Query(None, ge=0, le=1, description="Override the default threshold")):
    body = await request.body()
    if not body:
        raise HTTPException(400, "empty body: send 16 kHz 16-bit PCM or WAV audio")
    if len(body) > MAX_BYTES:
        raise HTTPException(413, f"audio too long: max {MAX_BYTES // (2 * SR)} s")
    ctype = request.headers.get("content-type", "application/octet-stream").split(";")[0].strip().lower()
    if ctype not in WAV_TYPES | PCM_TYPES:
        raise HTTPException(415, f"unsupported content-type {ctype!r}: use audio/wav or application/octet-stream")
    try:
        audio = read_wav(body) if ctype in WAV_TYPES else pcm16_to_float(body)
    except (ValueError, EOFError, wave.Error) as e:
        raise HTTPException(422, f"could not read audio: {e}")
    if len(audio) == 0:
        raise HTTPException(422, "audio contains no samples")
    t_wait = time.perf_counter()
    async with slots:
        queue_ms = (time.perf_counter() - t_wait) * 1000
        result = await run_in_threadpool(infer, audio, threshold)
    result.queue_ms = queue_ms
    return result
