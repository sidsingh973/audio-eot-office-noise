"""Download Pipecat Smart Turn v3.2 shards and keep only the English rows.

Every shard mixes all 23 languages (~25% English per row group), so each one is downloaded,
filtered to English, written to data/english/<split>/, and the raw shard deleted.
Re-running skips shards that are already done.

  python -m eot_lora.download                 # test (10 shards) then train (83 shards)
  python -m eot_lora.download --split test
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import socket
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REPOS = {
    "test": ("pipecat-ai/smart-turn-data-v3.2-test", 10),
    "train": ("pipecat-ai/smart-turn-data-v3.2-train", 83),
}
URL = "https://huggingface.co/datasets/{repo}/resolve/main/data/train-{i:05d}-of-{n:05d}.parquet"
COLS = ["id", "language", "endpoint_bool", "midfiller", "endfiller", "synthetic", "dataset", "audio"]

socket.setdefaulttimeout(120)


def fetch(split: str, i: int) -> str:
    repo, n = REPOS[split]
    out = DATA / "english" / split / f"{i:05d}.parquet"
    if out.exists():
        return f"{split} {i:02d}: cached"
    raw = DATA / "raw_shards" / f"{split}-{i:05d}.parquet"
    raw.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for attempt in range(5):
        try:
            urllib.request.urlretrieve(URL.format(repo=repo, i=i, n=n), raw)
            break
        except Exception as e:  # network hiccups / HF rate limits
            if attempt == 4:
                raise
            wait = 30 * (attempt + 1)
            print(f"{split} {i:02d}: {e!r}, retrying in {wait}s", flush=True)
            time.sleep(wait)
    table = pq.read_table(raw, columns=COLS, filters=[("language", "=", "eng")])
    tmp = out.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.rename(out)
    raw.unlink()
    return f"{split} {i:02d}: {table.num_rows} English rows ({time.time() - t0:.0f}s)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "train", "all"], default="all")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    splits = ["test", "train"] if args.split == "all" else [args.split]
    jobs = [(s, i) for s in splits for i in range(REPOS[s][1])]
    with cf.ThreadPoolExecutor(args.workers) as pool:
        for fut in cf.as_completed([pool.submit(fetch, s, i) for s, i in jobs]):
            print(fut.result(), flush=True)

    for s in splits:
        files = sorted((DATA / "english" / s).glob("*.parquet"))
        rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        print(f"{s}: {len(files)}/{REPOS[s][1]} shards, {rows} English clips")


if __name__ == "__main__":
    main()
