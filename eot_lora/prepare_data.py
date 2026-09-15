"""Build the audio for every training condition and test variant from the English shards.

Train English is split 90/10 into train/val by id hash. Sets written to data/built/<name>.parquet
(int16 PCM of the last 8 s, unpadded, plus metadata):

  {train,val}_A   Pipecat-native audio
  {train,val}_B   80% of clips (stratified by label x source) get office-3 at SNR ~ U(5,20) dB
  {train,val}_C   B's exact audio after WebRTC noise suppression (all clips, clean ones too)
  test_native, test_office5_{20,10,5,0}, test_office3_10, and each of those + "_ns"

Test variants keep the same clip order, so scores line up row by row across variants.

  python -m eot_lora.prepare_data            # both splits; skips a split whose outputs exist
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from eot_lora import audio as A

ENGLISH = A.ROOT / "data" / "english"
BUILT = A.ROOT / "data" / "built"
META = ["id", "endpoint_bool", "midfiller", "endfiller", "dataset", "synthetic"]
SCHEMA = pa.schema([
    ("id", pa.string()), ("endpoint_bool", pa.bool_()), ("midfiller", pa.bool_()),
    ("endfiller", pa.bool_()), ("dataset", pa.string()), ("synthetic", pa.bool_()),
    ("noise", pa.string()), ("snr_db", pa.float32()), ("ns", pa.bool_()),
    ("noise_floor_db", pa.float32()), ("pcm", pa.binary()),
])
TEST_SNRS = (20, 10, 5, 0)
TEST_NOISES = (("office5", TEST_SNRS), ("office3", (10,)))  # office-5 unseen; office-3 10 dB = seen-noise check
TEST_VARIANTS = ["native"] + [f"{nz}_{snr}" for nz, snrs in TEST_NOISES for snr in snrs]
SETS = {
    "train": [f"{s}_{c}" for s in ("train", "val") for c in "ABC"],
    "test": [f"test_{v}{ns}" for v in TEST_VARIANTS for ns in ("", "_ns")],
}
NOISY_FRACTION = 0.8
TRAIN_SNR = (5.0, 20.0)
MIN_SAMPLES = A.SR // 10


def h(s: str) -> int:
    return int.from_bytes(hashlib.sha1(s.encode()).digest()[:8], "little")


def assign_train(ids: list[str], labels: list[bool], sources: list[str]) -> dict[str, tuple[str, bool]]:
    """id -> (train|val, noisy). Noisy is exactly 80% within each (label, source) stratum."""
    strata = defaultdict(list)
    for i, lab, src in zip(ids, labels, sources):
        strata[(lab, src)].append(i)
    out = {}
    for members in strata.values():
        members.sort(key=lambda i: h(f"noisy:{i}"))
        k = round(NOISY_FRACTION * len(members))
        for rank, i in enumerate(members):
            out[i] = ("val" if h(f"split:{i}") % 10 == 0 else "train", rank < k)
    return out


def process(task: dict) -> list[tuple[str, dict]]:
    wave = A.last_window(A.decode(task["flac"]))
    if len(wave) < MIN_SAMPLES:
        return []
    base = {k: task[k] for k in META}
    base["noise_floor_db"] = A.noise_floor_db(wave)
    silence = np.zeros(A.WARMUP, np.float32)
    out = []

    def emit(name, x, noise=None, snr=None, ns=False):
        out.append((name, {**base, "noise": noise, "snr_db": snr, "ns": ns, "pcm": A.to_pcm16(x).tobytes()}))

    if task["split"] in ("train", "val"):
        s = task["split"]
        emit(f"{s}_A", wave)
        if task["noisy"]:
            nz = A.noise("office3")
            rng = A.clip_rng(task["id"], "office3")
            snr = float(rng.uniform(*TRAIN_SNR))
            offset = int(rng.integers(A.WARMUP, len(nz) - len(wave) + 1))
            pre, mix = A.overlay(wave, nz, snr, offset)
            emit(f"{s}_B", mix, "office3", snr)
            emit(f"{s}_C", A.webrtc_ns(mix, pre=pre), "office3", snr, ns=True)
        else:
            emit(f"{s}_B", wave)
            emit(f"{s}_C", A.webrtc_ns(wave, pre=silence), ns=True)
    else:
        emit("test_native", wave)
        emit("test_native_ns", A.webrtc_ns(wave, pre=silence), ns=True)
        for nz_name, snrs in TEST_NOISES:
            nz = A.noise(nz_name)
            offset = int(A.clip_rng(task["id"], f"test-{nz_name}").integers(A.WARMUP, len(nz) - len(wave) + 1))
            for snr in snrs:
                pre, mix = A.overlay(wave, nz, snr, offset)
                emit(f"test_{nz_name}_{snr}", mix, nz_name, snr)
                emit(f"test_{nz_name}_{snr}_ns", A.webrtc_ns(mix, pre=pre), nz_name, snr, ns=True)
    return out


def read_meta(split: str) -> pa.Table:
    return pq.read_table(ENGLISH / split, columns=["id", "endpoint_bool", "dataset"])


def shards(split: str, assignment: dict | None):
    """One list of tasks per shard, so at most one shard of FLAC bytes is in memory."""
    seen = set()
    for f in sorted((ENGLISH / split).glob("*.parquet")):
        batch = []
        for r in pq.read_table(f, columns=META + ["audio"]).to_pylist():
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            t = {k: r[k] for k in META}
            t["flac"] = r["audio"]["bytes"]
            if assignment is None:
                t["split"] = "test"
            else:
                t["split"], t["noisy"] = assignment[r["id"]]
            batch.append(t)
        yield batch


def build(split: str, workers: int) -> None:
    names = SETS[split]
    if all((BUILT / f"{n}.parquet").exists() for n in names):
        print(f"{split}: already built")
        return
    assignment = None
    if split == "train":
        meta = read_meta("train")
        test_ids = set(read_meta("test").column("id").to_pylist())
        ids = meta.column("id").to_pylist()
        overlap = test_ids.intersection(ids)
        assert not overlap, f"{len(overlap)} clip ids appear in both train and test"
        assignment = assign_train(ids, meta.column("endpoint_bool").to_pylist(), meta.column("dataset").to_pylist())

    BUILT.mkdir(parents=True, exist_ok=True)
    writers = {n: pq.ParquetWriter(BUILT / f"{n}.parquet.tmp", SCHEMA) for n in names}
    buffers, counts = defaultdict(list), Counter()

    def flush(n):
        if buffers[n]:
            writers[n].write_table(pa.Table.from_pylist(buffers[n], schema=SCHEMA))
            buffers[n].clear()

    done = 0
    with mp.get_context("spawn").Pool(workers) as pool:
        for shard in shards(split, assignment):
            for rows in pool.imap(process, shard, chunksize=16):
                for n, row in rows:
                    buffers[n].append(row)
                    counts[n] += 1
                    if len(buffers[n]) >= 256:
                        flush(n)
            done += len(shard)
            print(f"{split}: {done} clips processed", flush=True)
    for n in names:
        flush(n)
        writers[n].close()
        (BUILT / f"{n}.parquet.tmp").rename(BUILT / f"{n}.parquet")
    for n in names:
        print(f"  {n}: {counts[n]} clips")


def summary() -> None:
    for f in sorted(BUILT.glob("*.parquet")):
        t = pq.read_table(f, columns=["endpoint_bool", "noise", "snr_db"]).to_pandas()
        noisy = t["noise"].notna()
        snr = t[noisy].astype({"snr_db": float}).groupby("endpoint_bool")["snr_db"].mean()
        by_label = {k: round(v, 3) for k, v in snr.items()}
        print(f"{f.stem:22s} n={len(t):6d}  complete={t['endpoint_bool'].mean():.3f}  "
              f"noisy={noisy.mean():.3f}  mean SNR by label={by_label}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test", "all"], default="all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    for split in (["test", "train"] if args.split == "all" else [args.split]):
        build(split, args.workers)
    summary()


if __name__ == "__main__":
    main()
