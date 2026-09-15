"""Sanity checks for eot_lora.audio before building datasets.

Checks: noise decoding, SNR accuracy, NS determinism, NS delay, NS warm-up length, NS speed.
Writes a few WAVs per condition to --wav-dir for listening.

  python -m eot_lora.check_audio --wav-dir /tmp/eot_wavs
"""
from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from eot_lora import audio as A

LOCAL_TEST = A.ROOT / "data" / "english" / "test" / "00000.parquet"
FALLBACK = Path.home() / "Desktop" / "happyrobot" / "data" / "raw" / "pipecat-v3.1-test-0000.parquet"


def load_clips(n: int) -> list[dict]:
    src = LOCAL_TEST if LOCAL_TEST.exists() else FALLBACK
    t = pq.read_table(src, columns=["id", "endpoint_bool", "audio"], filters=[("language", "=", "eng")])
    rows = t.slice(0, n).to_pylist()
    for r in rows:
        r["wave"] = A.last_window(A.decode(r["audio"]["bytes"]))
    print(f"{len(rows)} clips from {src.name}")
    return rows


def write_wav(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(A.SR)
        w.writeframes(A.to_pcm16(x).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--wav-dir", type=Path)
    args = ap.parse_args()

    for name in A.NOISES:
        nz = A.noise(name)
        print(f"{name}: {len(nz) / A.SR:.1f} s, rms {20 * np.log10(np.sqrt(np.mean(nz**2))):.1f} dBFS")

    clips = load_clips(args.n)
    office3 = A.noise("office3")

    errs, clipped = [], 0
    for c in clips:
        for snr in (0, 5, 10, 20):
            _, mix = A.overlay(c["wave"], office3, snr, offset=A.SR * 2)
            added = mix - c["wave"]
            if np.abs(mix).max() >= 0.989:
                clipped += 1
                continue
            measured = 10 * np.log10(A.active_power(c["wave"]) / np.mean(added**2))
            errs.append(abs(measured - snr))
    print(f"SNR error: max {max(errs):.4f} dB over {len(errs)} mixes ({clipped} rescaled for clipping)")

    x = clips[0]["wave"]
    assert np.array_equal(A.webrtc_ns(x), A.webrtc_ns(x)), "NS is not deterministic"
    print("NS deterministic: yes")

    lags = np.arange(-800, 801)
    for label, delay in (("uncompensated", 0), (f"compensated ({A.NS_DELAY})", A.NS_DELAY)):
        peaks = []
        for c in clips[:10]:
            y = A.webrtc_ns(c["wave"], delay=delay)
            xc = [np.dot(c["wave"][max(0, -l):len(y) - max(0, l)], y[max(0, l):len(y) - max(0, -l)]) for l in lags]
            peaks.append(int(lags[int(np.argmax(xc))]))
        print(f"NS lag {label}, samples per clip: {peaks}")

    body = office3[3 * A.SR: 5 * A.SR]
    level = lambda y: 10 * np.log10(np.mean(y**2) + 1e-12)
    print(f"warm-up test on a fixed 2 s of office-3 (input {level(body):.1f} dBFS):")
    for w in (0, 1, 3):
        pre = office3[(3 - w) * A.SR: 3 * A.SR] if w else None
        y = A.webrtc_ns(body, pre=pre)
        print(f"  warm-up {w} s: residual first 0.5 s {level(y[: A.SR // 2]):.1f} dBFS, "
              f"last 0.5 s {level(y[-A.SR // 2:]):.1f} dBFS")

    t0 = time.time()
    for c in clips[:20]:
        pre, mix = A.overlay(c["wave"], office3, 10, offset=A.SR * 2)
        A.webrtc_ns(mix, pre=pre)
    dur = sum(len(c["wave"]) for c in clips[:20]) / A.SR
    print(f"NS speed: {(time.time() - t0) / 20 * 1000:.0f} ms per clip ({dur / (time.time() - t0):.0f}x real time)")

    if args.wav_dir:
        args.wav_dir.mkdir(parents=True, exist_ok=True)
        for c in clips[:5]:
            tag = f"{c['id'][:8]}_{'complete' if c['endpoint_bool'] else 'incomplete'}"
            pre, mix = A.overlay(c["wave"], office3, 10, offset=A.SR * 2)
            write_wav(args.wav_dir / f"{tag}_A_native.wav", c["wave"])
            write_wav(args.wav_dir / f"{tag}_B_office3_10dB.wav", mix)
            write_wav(args.wav_dir / f"{tag}_C_office3_10dB_ns.wav", A.webrtc_ns(mix, pre=pre))
        print(f"wrote WAVs to {args.wav_dir}")


if __name__ == "__main__":
    main()
