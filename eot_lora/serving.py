"""Torch-free inference for the audio end-of-turn model: numpy log-mel + ONNX Runtime.

Reproduces the training preprocessing exactly: keep the last 8 s, zero-pad at the start, then
WhisperFeatureExtractor(chunk_length=8, do_normalize=True). Only numpy and onnxruntime are needed,
which keeps the serving image small. The API, client, benchmarks and Docker image all use this module.

Model directory layout (e.g. runs/B): model.onnx, mel_filters.npy, serving.json (threshold etc.).
"""
from __future__ import annotations

import io
import json
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

SR = 16_000
WINDOW = 8 * SR
N_FFT, HOP = 400, 160
HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # periodic, as torch.hann_window


def pcm16_to_float(data: bytes) -> np.ndarray:
    """Little-endian 16-bit PCM bytes -> float32 in [-1, 1]."""
    if len(data) % 2:
        raise ValueError("raw PCM must be 16-bit samples (an even number of bytes)")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def read_wav(data: bytes) -> np.ndarray:
    """16 kHz 16-bit PCM WAV (mono, or multi-channel averaged to mono) -> float32."""
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"WAV must be 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        if w.getframerate() != SR:
            raise ValueError(f"WAV must be {SR} Hz, got {w.getframerate()} Hz")
        channels = w.getnchannels()
        x = pcm16_to_float(w.readframes(w.getnframes()))
    return x.reshape(-1, channels).mean(axis=1) if channels > 1 else x


def log_mel(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """float32 audio (any length) -> [80, 800] features, identical to the training pipeline."""
    x = np.asarray(audio, dtype=np.float32)[-WINDOW:]
    if len(x) < WINDOW:
        x = np.pad(x, (WINDOW - len(x), 0))
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    x = np.pad(x, N_FFT // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP]
    power = np.abs(np.fft.rfft(frames * HANN, axis=-1)) ** 2
    mel = power[:-1].astype(np.float32) @ mel_filters
    logs = np.log10(np.maximum(mel, 1e-10))
    logs = np.maximum(logs, logs.max() - 8.0)
    return ((logs + 4.0) / 4.0).T.astype(np.float32)


class AudioEoTPredictor:
    def __init__(self, model_dir: str | Path = "runs/B", model_file: str | None = None, intra_op_threads: int = 1):
        self.model_dir = Path(model_dir).resolve()
        self.config = json.loads((self.model_dir / "serving.json").read_text())
        self.threshold = float(self.config["threshold"])
        self.mel_filters = np.load(self.model_dir / "mel_filters.npy").astype(np.float32)
        self.model_file = model_file or self.config.get("model", "model.onnx")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = intra_op_threads
        opts.inter_op_num_threads = 1
        if intra_op_threads > 1:  # idle ORT threads otherwise spin and steal cores from other workers
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.model_dir / self.model_file), opts,
                                            providers=["CPUExecutionProvider"])

    def features(self, audio: np.ndarray) -> np.ndarray:
        return log_mel(audio, self.mel_filters)

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        """[B, 80, 800] features -> P(turn complete) per item."""
        return self.session.run(None, {"input_features": features.astype(np.float32, copy=False)})[0][:, 0]

    def predict_proba(self, audios: list[np.ndarray]) -> np.ndarray:
        return self.predict_features(np.stack([self.features(a) for a in audios]))

    def predict(self, audio: np.ndarray) -> float:
        return float(self.predict_proba([audio])[0])
