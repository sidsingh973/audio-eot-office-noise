"""Train one condition (A, B or C) with the identical recipe: LoRA + head on precomputed log-mels.

The checkpoint is always the final epoch (no best-epoch picking), so the three conditions differ
only in their input audio. Writes runs/<condition>/{model.pt, val_probs.npz, log.jsonl}.

  python -m eot_lora.train --condition A
  python -m eot_lora.train --condition A --limit 500 --epochs 1 --out runs/smoke_A   # smoke test
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# Cap MPS memory: once GPU buffers were swapped out, steps went from 1 s to >10 s. Now it OOMs instead.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from eot_lora import audio as A
from eot_lora.featurize import load as load_features
from eot_lora.metrics import classification_metrics
from eot_lora.model import build, save

RUNS = A.ROOT / "runs"


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def batch_tensor(X: np.ndarray, idx: np.ndarray, dev: str) -> torch.Tensor:
    idx = np.sort(idx)  # sequential-ish memmap reads
    return torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).to(dev), idx


@torch.no_grad()
def predict(model, X: np.ndarray, dev: str, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(np.asarray(X[i:i + batch], dtype=np.float32)).to(dev)
        out.append(torch.sigmoid(model(x)).float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["A", "B", "C"])
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4, help="LoRA learning rate")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="use only the first N train/val clips (smoke test)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    out = args.out or RUNS / args.condition
    out.mkdir(parents=True, exist_ok=True)
    dev = device()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    Xtr, mtr = load_features(f"train_{args.condition}")
    Xva, mva = load_features(f"val_{args.condition}")
    ytr = mtr.column("endpoint_bool").to_numpy(zero_copy_only=False).astype(np.float32)
    yva = mva.column("endpoint_bool").to_numpy(zero_copy_only=False).astype(int)
    n_tr = min(len(ytr), args.limit or len(ytr))
    n_va = min(len(yva), args.limit or len(yva))
    Xva, yva = Xva[:n_va], yva[:n_va]
    print(f"condition {args.condition}: {n_tr} train / {n_va} val clips on {dev}", flush=True)

    model = build().to(dev)
    lora = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    head = [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n]
    opt = torch.optim.AdamW([{"params": lora, "lr": args.lr}, {"params": head, "lr": args.head_lr}],
                            weight_decay=0.01)
    steps_per_epoch = n_tr // args.batch
    total = steps_per_epoch * args.epochs
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    log = open(out / "log.jsonl", "w")
    step, t0 = 0, time.time()
    t_mark = t0
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = rng.permutation(n_tr)
        running = []
        for b in range(steps_per_epoch):
            x, idx = batch_tensor(Xtr, perm[b * args.batch:(b + 1) * args.batch], dev)
            y = torch.from_numpy(ytr[idx]).to(dev)
            pos_weight = ((y == 0).sum() / (y == 1).sum().clamp(min=1)).clamp(0.1, 10.0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            running.append(loss.item())
            if step == 50:
                eta = (time.time() - t0) / 50 * total / 60
                print(f"50 steps in {time.time() - t0:.0f}s -> ETA {eta:.0f} min for {total} steps", flush=True)
            if step % 200 == 0:
                now = time.time()
                mem = f", mps {torch.mps.driver_allocated_memory() / 1e9:.1f} GB" if dev == "mps" else ""
                print(f"{time.strftime('%H:%M:%S')} epoch {epoch} step {step}/{total} "
                      f"loss {np.mean(running[-200:]):.4f} ({(now - t0) / 60:.1f} min, "
                      f"{(now - t_mark) / 200:.2f} s/step{mem})", flush=True)
                t_mark = now

        probs = predict(model, Xva, dev, batch=64)
        if dev == "mps":
            torch.mps.empty_cache()
        m = classification_metrics(yva, probs, 0.5)
        rec = {"epoch": epoch, "step": step, "train_loss": float(np.mean(running)),
               "val_pr_auc": m["pr_auc"], "val_roc_auc": m["roc_auc"], "val_acc": m["accuracy"],
               "minutes": (time.time() - t0) / 60}
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    save(model, out / "model.pt")
    ids = mva.column("id").to_pylist()[:n_va]
    np.savez(out / "val_probs.npz", ids=np.array(ids), labels=yva, probs=probs)
    json.dump(vars(args) | {"out": str(out), "n_train": n_tr, "n_val": n_va}, open(out / "config.json", "w"), indent=2)
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
