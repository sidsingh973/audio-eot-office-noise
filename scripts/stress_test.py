"""Stress-test the audio EoT API: throughput and latency, closed loop and open loop.

  python scripts/stress_test.py --tag w4 --note "4 workers x 1 thread"            # closed-loop sweep
  python scripts/stress_test.py --tag w4_open --open-loop 20 40 60 80 --duration 30

Closed loop: C clients each send their next request as soon as the previous one returns. This finds
peak throughput but hides queueing, because clients slow down when the server does.
Open loop: requests arrive at a fixed average rate (Poisson), whether or not the server keeps up,
which is how real calls behave. The useful number is the highest rate whose p99 stays under 100 ms.

Payloads are real test clips (office-5 noise, 10 dB) sent as raw 16-bit PCM. Writes
reports/stress_<tag>.md and appends every level to reports/stress_results.jsonl.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from client import load_clips  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ("latency_ms", "queue_ms", "features_ms", "model_ms")


class Recorder:
    def __init__(self):
        self.client_ms, self.errors = [], 0
        self.server = {k: [] for k in FIELDS}

    async def send(self, client: httpx.AsyncClient, payload: bytes):
        t0 = time.perf_counter()
        try:
            r = await client.post("/predict", content=payload, headers={"content-type": "application/octet-stream"})
            r.raise_for_status()
            self.client_ms.append((time.perf_counter() - t0) * 1000)
            d = r.json()
            for k in FIELDS:
                self.server[k].append(d[k])
        except Exception:
            self.errors += 1

    def summary(self, elapsed: float, **extra) -> dict:
        lat = np.array(self.client_ms) if self.client_ms else np.array([np.nan])
        med = {f"{k.split('_')[0]}_p50": float(np.median(v)) if v else float("nan") for k, v in self.server.items()}
        return {**extra, "requests": len(self.client_ms) + self.errors, "errors": self.errors,
                "rps": len(self.client_ms) / elapsed, "p50": float(np.percentile(lat, 50)),
                "p95": float(np.percentile(lat, 95)), "p99": float(np.percentile(lat, 99)), **med}


async def closed_loop(client, payloads, concurrency: int, total: int) -> dict:
    rec, jobs = Recorder(), iter(range(total))

    async def worker():
        for i in jobs:
            await rec.send(client, payloads[i % len(payloads)])

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return rec.summary(time.perf_counter() - t0, mode="closed", concurrency=concurrency)


async def open_loop(client, payloads, rate: float, duration: float, seed: int = 0) -> dict:
    rec, rng, tasks = Recorder(), np.random.default_rng(seed), []
    t0 = time.perf_counter()
    t_next, i = t0, 0
    while t_next - t0 < duration:
        delay = t_next - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(rec.send(client, payloads[i % len(payloads)])))
        i += 1
        t_next += rng.exponential(1.0 / rate)
    await asyncio.gather(*tasks)
    return rec.summary(time.perf_counter() - t0, mode="open", offered_rps=rate)


def machine_info() -> str:
    cpu = platform.processor()
    if platform.system() == "Darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return f"{platform.system()} {platform.machine()}, {cpu}, {os.cpu_count()} cores"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--requests", type=int, default=1000, help="closed loop: requests per level")
    ap.add_argument("--open-loop", type=float, nargs="*", help="open loop: offered request rates (req/s)")
    ap.add_argument("--duration", type=float, default=30, help="open loop: seconds per rate")
    ap.add_argument("--tag", default="local")
    ap.add_argument("--note", default="", help="server configuration, for the report")
    args = ap.parse_args()

    clips = load_clips("test_office5_10", 500)
    payloads = [c["pcm"] for c in clips]
    secs = np.array([len(p) / 2 / 16000 for p in payloads])
    limits = httpx.Limits(max_connections=512, max_keepalive_connections=512)
    async with httpx.AsyncClient(base_url=args.url, limits=limits, timeout=120) as client:
        health = (await client.get("/health")).json()
        await closed_loop(client, payloads, 4, 100)  # warm-up
        results = []
        levels = [("open", r) for r in args.open_loop] if args.open_loop else [("closed", c) for c in args.concurrency]
        for mode, x in levels:
            r = await (open_loop(client, payloads, x, args.duration) if mode == "open"
                       else closed_loop(client, payloads, x, args.requests))
            r |= {"tag": args.tag, "note": args.note}
            results.append(r)
            head = f"offered={x:5.0f}/s" if mode == "open" else f"clients={x:3d}"
            print(f"{head}  rps={r['rps']:6.1f}  p50={r['p50']:7.1f}ms  p95={r['p95']:7.1f}ms  p99={r['p99']:7.1f}ms  "
                  f"queue={r['queue_p50']:.1f}  features={r['features_p50']:.1f}  model={r['model_p50']:.1f}  "
                  f"errors={r['errors']}", flush=True)

    first = "Offered (req/s)" if args.open_loop else "Clients"
    lines = [f"# Stress test: {args.tag}", "",
             f"- Machine: {machine_info()}",
             f"- Server: {args.note or 'as started'}; model {health['condition']} ({health['model_file']}), "
             f"{health['ort_threads']} ORT thread(s), {health['max_inflight_per_worker']} in-flight per worker",
             f"- {'Open loop, Poisson arrivals, ' + str(int(args.duration)) + ' s per rate' if args.open_loop else 'Closed loop, ' + str(args.requests) + ' requests per level'}; "
             f"payloads: 500 real clips (median {np.median(secs):.1f} s, max {secs.max():.1f} s) as 16-bit PCM", "",
             f"| {first} | Throughput (req/s) | p50 (ms) | p95 (ms) | p99 (ms) | Queue p50 | Features p50 | Model p50 | Errors |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        x = f"{r['offered_rps']:.0f}" if r["mode"] == "open" else r["concurrency"]
        lines.append(f"| {x} | {r['rps']:.0f} | {r['p50']:.1f} | {r['p95']:.1f} | {r['p99']:.1f} | "
                     f"{r['queue_p50']:.1f} | {r['features_p50']:.1f} | {r['model_p50']:.1f} | {r['errors']} |")
    lines += ["", "Latencies are end to end as seen by the client. Queue/Features/Model are server-side medians (ms). "
              "The load generator runs on the same machine."]
    out = ROOT / "reports" / f"stress_{args.tag}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    with open(ROOT / "reports" / "stress_results.jsonl", "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
