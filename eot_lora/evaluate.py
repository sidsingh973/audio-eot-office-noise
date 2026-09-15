"""Score models A, B, C and Smart Turn v3.2 on every test variant; write reports/lora_eval.{md,json}.

Each model is paired with the input it would get in deployment: C gets noise-suppressed audio,
the others raw audio. The full model x variant matrix is reported too.

Threshold-free metrics (PR-AUC with paired bootstrap CI, ROC-AUC, recall at <=5% cut-in with the
threshold picked on that variant) compare ranking quality. The deployable metric uses each model's
threshold picked on its own val set at a 5% cut-in budget; Smart Turn's comes from B's val clips.

  python -m eot_lora.evaluate
  python -m eot_lora.evaluate --models ST --limit 500      # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import average_precision_score

from eot_lora import audio as A
from eot_lora.featurize import load as load_features
from eot_lora.metrics import bootstrap_ci, classification_metrics, recall_at_cut_in, threshold_for_cut_in_rate
from eot_lora.prepare_data import TEST_VARIANTS
from eot_lora.train import RUNS, device, predict

SMART_TURN = Path(os.getenv("SMART_TURN_ONNX", A.ROOT / "models" / "smart-turn" / "smart-turn-v3.2-gpu.onnx"))
SMART_TURN_URL = "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-gpu.onnx"  # BSD-2-Clause
REPORTS = A.ROOT / "reports"
BUDGET = 0.05
NAMES = {"A": "A: Pipecat-native", "B": "B: + office-3", "C": "C: + office-3 -> NS", "ST": "Smart Turn v3.2 (fp32)"}


def deploy_set(model: str, variant: str) -> str:
    return f"test_{variant}_ns" if model == "C" else f"test_{variant}"


class Scorer:
    def __init__(self, limit: int | None):
        self.limit = limit
        self.models = {}
        self.dir = REPORTS / "scores" / (f"n{limit}" if limit else "full")
        self.dir.mkdir(parents=True, exist_ok=True)

    def labels(self, set_name: str) -> np.ndarray:
        _, meta = load_features(set_name)
        return meta.column("endpoint_bool").to_numpy(zero_copy_only=False).astype(int)[: self.limit]

    def ids(self, set_name: str) -> list[str]:
        return load_features(set_name)[1].column("id").to_pylist()[: self.limit]

    def _model(self, name: str):
        if name not in self.models:
            if name == "ST":
                if not SMART_TURN.exists():
                    SMART_TURN.parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(SMART_TURN_URL, SMART_TURN)
                so = ort.SessionOptions()
                so.intra_op_num_threads = os.cpu_count() or 4
                self.models[name] = ort.InferenceSession(str(SMART_TURN), so, providers=["CPUExecutionProvider"])
            else:
                from eot_lora.model import load
                self.models[name] = load(RUNS / name / "model.pt", device())
        return self.models[name]

    def scores(self, name: str, set_name: str) -> np.ndarray:
        path = self.dir / f"{name}__{set_name}.npy"
        if path.exists():
            return np.load(path)
        X, _ = load_features(set_name)
        X = X[: self.limit or len(X)]
        m = self._model(name)
        if name == "ST":
            p = np.concatenate([m.run(None, {"input_features": np.asarray(X[i:i + 64], np.float32)})[0][:, 0]
                                for i in range(0, len(X), 64)])
        else:
            p = predict(m, X, device())
        np.save(path, p)
        print(f"  scored {name} on {set_name}", flush=True)
        return p

    def threshold(self, name: str) -> float:
        if name == "ST":
            return threshold_for_cut_in_rate(self.labels("val_B"), self.scores("ST", "val_B"), BUDGET)
        z = np.load(RUNS / name / "val_probs.npz")
        return threshold_for_cut_in_rate(z["labels"], z["probs"], BUDGET)


def metrics(labels: np.ndarray, probs: np.ndarray, thr: float) -> dict:
    at_thr = classification_metrics(labels, probs, thr)
    lo, hi = bootstrap_ci(labels, probs, average_precision_score)
    return {
        "pr_auc": at_thr["pr_auc"], "pr_auc_ci": [lo, hi], "roc_auc": at_thr["roc_auc"],
        "acc_at_0.5": classification_metrics(labels, probs, 0.5)["accuracy"],
        "recall_at_5pct_cut_in": recall_at_cut_in(labels, probs, BUDGET),
        "val_threshold": thr, "recall_at_val_thr": at_thr["recall"], "cut_in_at_val_thr": at_thr["cut_in_rate"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["A", "B", "C", "ST"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    models = [m for m in args.models if m == "ST" or (RUNS / m / "model.pt").exists()]
    if missing := set(args.models) - set(models):
        print(f"skipping models without a trained run: {sorted(missing)}")
    sc = Scorer(args.limit)

    ref_ids = sc.ids("test_native")
    for v in TEST_VARIANTS:
        for s in (f"test_{v}", f"test_{v}_ns"):
            assert sc.ids(s) == ref_ids, f"{s} rows are not aligned with test_native"
    labels = sc.labels("test_native")

    results = {"n_test": len(labels), "budget": BUDGET, "deploy": {}, "matrix": {}, "noise_floor": {}}
    for m in models:
        thr = sc.threshold(m)
        results["deploy"][m] = {v: metrics(labels, sc.scores(m, deploy_set(m, v)), thr) for v in TEST_VARIANTS}
        results["matrix"][m] = {s: float(average_precision_score(labels, sc.scores(m, s)))
                                for v in TEST_VARIANTS for s in (f"test_{v}", f"test_{v}_ns")}

    floor = np.asarray(load_features("test_native")[1].column("noise_floor_db").to_pylist(), dtype=float)[: args.limit]
    edges = np.percentile(floor, [100 / 3, 200 / 3])
    bins = np.digitize(floor, edges)
    for m in models:
        p = sc.scores(m, deploy_set(m, "native"))
        results["noise_floor"][m] = [float(average_precision_score(labels[bins == b], p[bins == b])) for b in range(3)]
    results["noise_floor_edges_db"] = edges.tolist()

    REPORTS.mkdir(exist_ok=True)
    suffix = f"_n{args.limit}" if args.limit else ""
    json.dump(results, open(REPORTS / f"lora_eval{suffix}.json", "w"), indent=2)
    (REPORTS / f"lora_eval{suffix}.md").write_text(render(results, models))
    print(render(results, models))


def render(r: dict, models: list[str]) -> str:
    f3 = lambda x: f"{x:.3f}"
    L = [f"# Office-noise end-of-turn evaluation\n",
         f"{r['n_test']} English clips from smart-turn-data-v3.2-test. Office-5 noise is unseen by every model. "
         f"Each model gets its deployment input (C: noise-suppressed; others: raw). "
         f"Thresholds from each model's own val set at a {r['budget']:.0%} cut-in budget.\n"]

    head = "office5_10"
    L += [f"## Headline: office-5 at 10 dB\n",
          "| Model | PR-AUC [95% CI] | ROC-AUC | Recall @ ≤5% cut-in (best thr) | Val thr → recall | Val thr → cut-in |",
          "|---|---:|---:|---:|---:|---:|"]
    for m in models:
        d = r["deploy"][m][head]
        L.append(f"| {NAMES[m]} | {f3(d['pr_auc'])} [{f3(d['pr_auc_ci'][0])}, {f3(d['pr_auc_ci'][1])}] | "
                 f"{f3(d['roc_auc'])} | {f3(d['recall_at_5pct_cut_in'])} | {f3(d['recall_at_val_thr'])} | "
                 f"{f3(d['cut_in_at_val_thr'])} |")

    cols = ["native"] + [f"office5_{s}" for s in (20, 10, 5, 0)] + ["office3_10"]
    titles = ["native", "office-5 20 dB", "10 dB", "5 dB", "0 dB*", "office-3 10 dB (seen)"]
    for key, title in (("pr_auc", "PR-AUC by noise level"), ("recall_at_5pct_cut_in", "Recall at ≤5% cut-in by noise level"),
                       ("cut_in_at_val_thr", "Cut-in rate at the val threshold (drift under noise)")):
        L += [f"\n## {title}\n", "| Model | " + " | ".join(titles) + " |", "|---|" + "---:|" * len(cols)]
        for m in models:
            L.append(f"| {NAMES[m]} | " + " | ".join(f3(r["deploy"][m][c][key]) for c in cols) + " |")
    L.append("\n*0 dB is louder than anything in training (5-20 dB).")

    L += ["\n## Full matrix: PR-AUC for every model on every input\n",
          "| Model | " + " | ".join(f"{t} raw / NS" for t in titles) + " |", "|---|" + "---:|" * len(cols)]
    for m in models:
        L.append(f"| {NAMES[m]} | " + " | ".join(
            f"{f3(r['matrix'][m][f'test_{c}'])} / {f3(r['matrix'][m][f'test_{c}_ns'])}" for c in cols) + " |")

    e = r["noise_floor_edges_db"]
    L += [f"\n## Native test split by the clip's own background level (Pipecat already mixed some noise in)\n",
          f"| Model | quiet (< {e[0]:.0f} dBFS) | middle | noisy (> {e[1]:.0f} dBFS) |", "|---|---:|---:|---:|"]
    for m in models:
        L.append(f"| {NAMES[m]} | " + " | ".join(f3(x) for x in r["noise_floor"][m]) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
