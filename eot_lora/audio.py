"""Audio helpers: decoding, the 8 s model window, office-noise overlay, WebRTC noise suppression.

All audio is 16 kHz mono float32 in [-1, 1].
Processing order for one clip: last 8 s -> mix noise -> (prepend warm-up -> NS -> trim) -> zero-pad
at the start to 8 s -> log-mel. Mixing happens before padding, as in live use where the
padding is digital silence.
"""
from __future__ import annotations

import hashlib
import io
from functools import lru_cache
from pathlib import Path

import av
import numpy as np

SR = 16_000
WINDOW = 8 * SR
FRAME = SR // 100  # 10 ms: the frame size the WebRTC audio processing module requires
WARMUP = SR  # 1 s of preceding noise lets NS settle before the clip starts, as on a live stream
ROOT = Path(__file__).resolve().parent.parent
NOISES = {
    "office3": ROOT / "thellywellyn-it-office-3-535970.mp3",
    "office5": ROOT / "thellywellyn-it-office-5-535971.mp3",
}
NS_DELAY = 96  # samples: WebRTC NS's 256-point FFT on 160-sample frames; measured 93-96 by eot_lora.check_audio


def decode(src) -> np.ndarray:
    """Decode a file path or encoded bytes (FLAC, MP3, ...) to 16 kHz mono float32. PyAV bundles ffmpeg."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=SR)
    chunks = []
    with av.open(src) as container:
        for frame in container.decode(audio=0):
            chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(frame)]
        chunks += [f.to_ndarray().reshape(-1) for f in resampler.resample(None)]
    return np.concatenate(chunks).astype(np.float32) / 32768.0


@lru_cache(maxsize=None)
def noise(name: str) -> np.ndarray:
    return decode(str(NOISES[name]))


def to_pcm16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16)


def from_pcm16(b) -> np.ndarray:
    return np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0


def last_window(x: np.ndarray) -> np.ndarray:
    return x[-WINDOW:]


def pad_start(x: np.ndarray) -> np.ndarray:
    """Same as smart-turn/audio_utils.py truncate_audio_to_last_n_seconds."""
    x = x[-WINDOW:]
    return np.pad(x, (WINDOW - len(x), 0)) if len(x) < WINDOW else x


def clip_rng(clip_id: str, salt: str) -> np.random.Generator:
    """Per-clip RNG that is identical across processes and runs (Python's hash() is not)."""
    seed = int.from_bytes(hashlib.sha1(f"{salt}:{clip_id}".encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def _frame_power(x: np.ndarray, frame: int = 320) -> np.ndarray:
    n = len(x) // frame
    if n == 0:
        return np.array([np.mean(x**2) if len(x) else 0.0])
    return (x[: n * frame].reshape(n, frame) ** 2).mean(axis=1)


def active_power(x: np.ndarray, floor_db: float = 40.0) -> float:
    """Mean power of 20 ms frames within floor_db of the loudest frame.

    Using whole-clip power would let trailing silence (longer on complete turns) lower the
    noise level for one label, handing the model a shortcut.
    """
    p = _frame_power(x)
    keep = p >= p.max() * 10 ** (-floor_db / 10)
    return float(p[keep].mean()) + 1e-12


def noise_floor_db(x: np.ndarray) -> float:
    """Background level of a clip: mean power of its quietest 10% of 20 ms frames, in dBFS."""
    p = np.sort(_frame_power(x))
    q = p[: max(1, len(p) // 10)]
    return float(10 * np.log10(q.mean() + 1e-12))


def overlay(speech: np.ndarray, noise_wave: np.ndarray, snr_db: float, offset: int,
            warmup: int = WARMUP) -> tuple[np.ndarray, np.ndarray]:
    """Mix noise into speech at snr_db (vs. active speech power).

    The noise starts at `offset`. Also returns the `warmup` samples of the same noise stream
    that come just before the clip, at the same gain. If the mix would clip, both are scaled
    down together, which keeps the SNR.
    """
    n = len(speech)
    seg = noise_wave[(offset - warmup + np.arange(warmup + n)) % len(noise_wave)]
    body = seg[warmup:]
    gain = np.sqrt(active_power(speech) / (np.mean(body**2) + 1e-12) / 10 ** (snr_db / 10))
    mix, pre = speech + gain * body, gain * seg[:warmup]
    peak = max(np.abs(mix).max(initial=0.0), np.abs(pre).max(initial=0.0))
    if peak > 0.99:
        mix, pre = mix * (0.99 / peak), pre * (0.99 / peak)
    return pre.astype(np.float32), mix.astype(np.float32)


def webrtc_ns(x: np.ndarray, pre: np.ndarray | None = None, delay: int = NS_DELAY) -> np.ndarray:
    """WebRTC noise suppression (LiveKit's WebRTC audio processing module); everything else off.

    A fresh module per clip keeps clips independent and results reproducible. `pre` (warm-up
    audio) is processed first and dropped; `delay` shifts the output back into alignment.
    """
    from livekit import rtc

    apm = rtc.AudioProcessingModule(noise_suppression=True, echo_cancellation=False,
                                    high_pass_filter=False, auto_gain_control=False)
    lead = len(pre) if pre is not None else 0
    full = np.concatenate([pre, x]) if pre is not None else x
    full = np.pad(full, (0, delay + (-(len(full) + delay)) % FRAME))
    pcm = to_pcm16(full)
    out = np.empty_like(pcm)
    for i in range(0, len(pcm), FRAME):
        frame = rtc.AudioFrame(pcm[i:i + FRAME].tobytes(), SR, 1, FRAME)
        apm.process_stream(frame)
        out[i:i + FRAME] = np.frombuffer(frame.data, dtype=np.int16)
    start = lead + delay
    return out[start:start + len(x)].astype(np.float32) / 32768.0
