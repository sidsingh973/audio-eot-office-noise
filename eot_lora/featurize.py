"""Precompute Whisper log-mel features for every built set, so training does no audio work.

For each data/built/<name>.parquet writes data/features/<name>.npy (fp16 [N, 80, 800], memory-mapped
at training time) and <name>.meta.parquet (row-aligned metadata, no audio). Preprocessing matches
Smart Turn's inference exactly: zero-pad at the start to 8 s, then WhisperFeatureExtractor(chunk_length=8)
with do_normalize=True.

  python -m eot_lora.featurize                 # every built set; skips ones already done
  python -m eot_lora.featurize --sets test_native train_A
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pyarrow.parquet as pq
from transformers import WhisperFeatureExtractor

from eot_lora import audio as A

BUILT = A.ROOT / "data" / "built"
FEATS = A.ROOT / "data" / "features"
SHAPE = (80, 800)


def featurize(name: str, fe: WhisperFeatureExtractor, batch: int = 256) -> None:
    meta_path = FEATS / f"{name}.meta.parquet"
    if meta_path.exists():
        print(f"{name}: cached")
        return
    src = BUILT / f"{name}.parquet"
    pf = pq.ParquetFile(src)
    n = pf.metadata.num_rows
    tmp = FEATS / f"{name}.tmp.npy"
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, *SHAPE))
    t0, i = time.time(), 0
    for b in pf.iter_batches(batch_size=batch, columns=["pcm"]):
        waves = [A.pad_start(A.from_pcm16(x)) for x in b.column("pcm").to_pylist()]
        feats = fe(waves, sampling_rate=A.SR, return_tensors="np", padding="max_length",
                   max_length=A.WINDOW, truncation=True, do_normalize=True).input_features
        mm[i:i + len(waves)] = feats.astype(np.float16)
        i += len(waves)
    mm.flush()
    del mm
    tmp.rename(FEATS / f"{name}.npy")
    pq.write_table(pq.read_table(src, columns=[c for c in pf.schema_arrow.names if c != "pcm"]), meta_path)
    print(f"{name}: {n} clips in {time.time() - t0:.0f}s", flush=True)


def load(name: str) -> tuple[np.ndarray, "pa.Table"]:
    return np.load(FEATS / f"{name}.npy", mmap_mode="r"), pq.read_table(FEATS / f"{name}.meta.parquet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*")
    args = ap.parse_args()
    FEATS.mkdir(parents=True, exist_ok=True)
    fe = WhisperFeatureExtractor(chunk_length=8)
    for name in args.sets or sorted(p.stem for p in BUILT.glob("*.parquet")):
        featurize(name, fe)


if __name__ == "__main__":
    main()
