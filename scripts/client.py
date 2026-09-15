"""Try the audio end-of-turn API: send WAV files, or labelled test clips, and print the decisions.

  python scripts/client.py clip1.wav clip2.wav
  python scripts/client.py --test-set test_office5_10 --n 20              # labelled clips from data/built
  python scripts/client.py --test-set test_native --n 5 --save-wavs /tmp/clips   # also write WAVs for curl

Start the API first:  uvicorn service.app:app --port 8000
"""
from __future__ import annotations

import argparse
import io
import wave
from pathlib import Path

import httpx
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
SR = 16_000


def to_wav(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return buf.getvalue()


def load_clips(set_name: str, n: int, seed: int = 0, pool: int = 512) -> list[dict]:
    """n labelled clips (random, reproducible) from the first ~`pool` rows of a built test set."""
    pf = pq.ParquetFile(ROOT / "data" / "built" / f"{set_name}.parquet")
    rows, g = [], 0
    while len(rows) < pool and g < pf.num_row_groups:
        rows += pf.read_row_group(g, columns=["id", "endpoint_bool", "pcm"]).to_pylist()
        g += 1
    idx = np.random.default_rng(seed).choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i] for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path, help="16 kHz 16-bit WAV files")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--test-set", default="test_office5_10", help="data/built/<name>.parquet")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--save-wavs", type=Path)
    args = ap.parse_args()

    with httpx.Client(base_url=args.url, timeout=30) as client:
        h = client.get("/health").json()
        print(f"model {h['condition']} ({h['model_file']}), threshold {h['threshold']:.3f}\n")

        if args.files:
            for f in args.files:
                r = client.post("/predict", content=f.read_bytes(), headers={"content-type": "audio/wav"})
                if r.status_code != 200:
                    print(f"{f.name}: HTTP {r.status_code} {r.json().get('detail')}")
                    continue
                d = r.json()
                print(f"{f.name:40s} P(end)={d['eot_probability']:.3f}  "
                      f"{'END OF TURN' if d['is_end_of_turn'] else 'keep listening':14s}  "
                      f"{d['audio_seconds']:.1f}s audio, {d['latency_ms']:.1f} ms")
            return

        clips = load_clips(args.test_set, args.n)
        if args.save_wavs:
            args.save_wavs.mkdir(parents=True, exist_ok=True)
        print(f"{'clip':10s} {'label':10s} {'P(end)':>7s}  {'':20s} {'decision':14s} {'ok':3s} {'ms':>6s}")
        correct, cut_ins, negatives, latency = 0, 0, 0, []
        for c in clips:
            r = client.post("/predict", content=c["pcm"], headers={"content-type": "application/octet-stream"})
            r.raise_for_status()
            d = r.json()
            label = "complete" if c["endpoint_bool"] else "mid-turn"
            ok = d["is_end_of_turn"] == c["endpoint_bool"]
            correct += ok
            if not c["endpoint_bool"]:
                negatives += 1
                cut_ins += d["is_end_of_turn"]
            latency.append(d["latency_ms"])
            print(f"{c['id'][:8]:10s} {label:10s} {d['eot_probability']:7.3f}  "
                  f"{'█' * int(d['eot_probability'] * 20):20s} "
                  f"{'END OF TURN' if d['is_end_of_turn'] else 'keep listening':14s} "
                  f"{'yes' if ok else 'NO':3s} {d['latency_ms']:6.1f}")
            if args.save_wavs:
                (args.save_wavs / f"{c['id'][:8]}_{label}.wav").write_bytes(to_wav(c["pcm"]))

    print(f"\naccuracy {correct}/{len(clips)}, interruptions {cut_ins}/{negatives} mid-turn clips, "
          f"server latency p50 {np.median(latency):.1f} ms")
    if args.save_wavs:
        example = next(args.save_wavs.glob("*.wav"))
        print(f"\nWAVs in {args.save_wavs}. Try:\n  curl -s {args.url}/predict -H 'content-type: audio/wav' "
              f"--data-binary @{example}")


if __name__ == "__main__":
    main()
