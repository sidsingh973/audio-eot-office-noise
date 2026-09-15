"""Model-only benchmark (no HTTP): cold start, per-stage latency, threads, batch scaling, int8, memory.

  python scripts/bench_model.py            # writes reports/bench_model.md

Uses 200 real clips (office-5 noise, 10 dB). int8 = onnxruntime dynamic quantization of runs/B/model.onnx,
saved as runs/B/model.int8.onnx and checked for accuracy on 1,000 clips.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import platform  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from client import load_clips  # noqa: E402
from eot_lora.serving import AudioEoTPredictor, pcm16_to_float  # noqa: E402

MODEL_DIR = ROOT / "runs" / "B"


def pct(xs, q):
    return float(np.percentile(xs, q))


def timeit(fn, n):
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def main() -> None:
    lines = ["# Model benchmark (no HTTP)", ""]
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() \
        if platform.system() == "Darwin" else platform.processor()
    lines.append(f"- Machine: {platform.system()} {platform.machine()}, {cpu}, {os.cpu_count()} cores")

    t0 = time.perf_counter()
    pred = AudioEoTPredictor(MODEL_DIR, intra_op_threads=1)
    load_ms = (time.perf_counter() - t0) * 1000
    clips = load_clips("test_office5_10", 200)
    audios = [pcm16_to_float(c["pcm"]) for c in clips]
    t0 = time.perf_counter()
    pred.predict(audios[0])
    first_ms = (time.perf_counter() - t0) * 1000
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e6 if platform.system() == "Darwin" else 1e3)
    lines.append(f"- Cold start: session load {load_ms:.0f} ms, first prediction {first_ms:.0f} ms; "
                 f"peak RSS of this process {rss_mb:.0f} MB")

    feats = np.stack([pred.features(a) for a in audios])
    decode = timeit(lambda: pcm16_to_float(clips[0]["pcm"]), 200)
    fstage = [timeit(lambda a=a: pred.features(a), 1)[0] for a in audios]
    mstage = [timeit(lambda f=f: pred.predict_features(f[None]), 1)[0] for f in feats]
    lines += ["", "## Per-stage latency, 1 thread, batch 1 (200 clips)", "",
              "| Stage | p50 (ms) | p95 (ms) |", "|---|---:|---:|",
              f"| Decode PCM | {pct(decode, 50):.2f} | {pct(decode, 95):.2f} |",
              f"| Log-mel (numpy) | {pct(fstage, 50):.2f} | {pct(fstage, 95):.2f} |",
              f"| Model (ONNX fp32) | {pct(mstage, 50):.2f} | {pct(mstage, 95):.2f} |"]
    print("\n".join(lines[-6:]), flush=True)

    lines += ["", "## ONNX threads per inference (batch 1)", "", "| Threads | Model p50 (ms) |", "|---:|---:|"]
    for th in (1, 2, 4):
        p = AudioEoTPredictor(MODEL_DIR, intra_op_threads=th)
        p.predict_features(feats[:1])
        lines.append(f"| {th} | {pct([timeit(lambda f=f: p.predict_features(f[None]), 1)[0] for f in feats[:100]], 50):.2f} |")
    print("\n".join(lines[-3:]), flush=True)

    lines += ["", "## Batch size (1 thread)", "", "| Batch | ms per batch | ms per clip |", "|---:|---:|---:|"]
    for bs in (1, 2, 4, 8, 16):
        t = pct(timeit(lambda: pred.predict_features(feats[:bs]), 30), 50)
        lines.append(f"| {bs} | {t:.1f} | {t / bs:.1f} |")
    print("\n".join(lines[-5:]), flush=True)

    lines += ["", "## fp32 vs int8 (dynamic quantization)", ""]
    int8_path = MODEL_DIR / "model.int8.onnx"
    try:
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic
        if not int8_path.exists():
            m = onnx.load(str(MODEL_DIR / "model.onnx"))
            del m.graph.value_info[:]  # stale shape annotations from the exporter break the quantizer's shape inference
            tmp = MODEL_DIR / "model.noshapes.onnx"
            onnx.save(m, str(tmp))
            quantize_dynamic(str(tmp), str(int8_path), weight_type=QuantType.QInt8)
            tmp.unlink()
        q = AudioEoTPredictor(MODEL_DIR, model_file="model.int8.onnx", intra_op_threads=1)
        q.predict_features(feats[:1])
        q_ms = pct([timeit(lambda f=f: q.predict_features(f[None]), 1)[0] for f in feats[:100]], 50)
        acc = load_clips("test_office5_10", 1000, seed=1, pool=1024)
        af = np.stack([pred.features(pcm16_to_float(c["pcm"])) for c in acc])
        y = np.array([c["endpoint_bool"] for c in acc], dtype=int)
        pf = np.concatenate([pred.predict_features(af[i:i + 32]) for i in range(0, len(af), 32)])
        pq_ = np.concatenate([q.predict_features(af[i:i + 32]) for i in range(0, len(af), 32)])
        lines += ["| Variant | Size (MB) | Model p50 (ms) | PR-AUC (1,000 clips) |", "|---|---:|---:|---:|",
                  f"| fp32 | {(MODEL_DIR / 'model.onnx').stat().st_size / 1e6:.1f} | {pct(mstage, 50):.2f} | "
                  f"{average_precision_score(y, pf):.4f} |",
                  f"| int8 dynamic | {int8_path.stat().st_size / 1e6:.1f} | {q_ms:.2f} | "
                  f"{average_precision_score(y, pq_):.4f} |", "",
                  f"Max |fp32 - int8| probability difference: {np.abs(pf - pq_).max():.3f}"]
    except Exception as e:  # report instead of failing the whole benchmark
        lines.append(f"int8 quantization failed: {e!r}")
    print("\n".join(lines[-6:]), flush=True)

    out = ROOT / "reports" / "bench_model.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
