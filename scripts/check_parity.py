"""Parity gates for the torch-free serving path (eot_lora/serving.py).

1. Features: numpy log-mel vs WhisperFeatureExtractor (the training featurizer) on real clips.
2. Probabilities: served ONNX model vs the evaluation scores on all of test_office5_10
   (evaluation used fp16 features + the PyTorch model, so expect ~1e-3 differences, not zero).

  python scripts/check_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
from transformers import WhisperFeatureExtractor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from eot_lora.serving import WINDOW, AudioEoTPredictor, pcm16_to_float  # noqa: E402

SET = "test_office5_10"


def main() -> None:
    pred = AudioEoTPredictor(ROOT / "runs" / "B", intra_op_threads=8)
    fe = WhisperFeatureExtractor(chunk_length=8)
    pf = pq.ParquetFile(ROOT / "data" / "built" / f"{SET}.parquet")

    feat_diff, probs, labels = 0.0, [], []
    for i, b in enumerate(pf.iter_batches(batch_size=64, columns=["pcm", "endpoint_bool"])):
        audios = [pcm16_to_float(x) for x in b.column("pcm").to_pylist()]
        feats = np.stack([pred.features(a) for a in audios])
        if i < 4:  # gate 1 on the first 256 clips
            padded = [np.pad(a[-WINDOW:], (WINDOW - len(a[-WINDOW:]), 0)) for a in audios]
            ref = fe(padded, sampling_rate=16000, return_tensors="np", padding="max_length",
                     max_length=WINDOW, truncation=True, do_normalize=True).input_features
            feat_diff = max(feat_diff, float(np.abs(feats - ref).max()))
        probs.append(pred.predict_features(feats))
        labels += b.column("endpoint_bool").to_pylist()
    probs, labels = np.concatenate(probs), np.array(labels, dtype=int)

    ref_probs = np.load(ROOT / "reports" / "scores" / "full" / f"B__{SET}.npy")
    diff = np.abs(probs - ref_probs)
    t = pred.threshold
    flips = int(((probs >= t) != (ref_probs >= t)).sum())
    print(f"gate 1  features   max |numpy - WhisperFeatureExtractor| = {feat_diff:.2e}  "
          f"({'PASS' if feat_diff < 1e-3 else 'FAIL'}, limit 1e-3)")
    print(f"gate 2  probabilities on {len(probs)} clips: max diff {diff.max():.2e}, mean {diff.mean():.2e}, "
          f"decision flips at thr {t:.3f}: {flips}")
    print(f"        PR-AUC served {average_precision_score(labels, probs):.4f} vs evaluation "
          f"{average_precision_score(labels, ref_probs):.4f}  ({'PASS' if diff.max() < 2e-2 else 'CHECK'})")


if __name__ == "__main__":
    main()
