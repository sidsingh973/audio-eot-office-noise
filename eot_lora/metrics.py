"""End-of-turn metrics, copied from ~/Desktop/happyrobot/eot/metrics.py, plus bootstrap CIs.

Label 1 = turn complete (agent should speak), 0 = mid-turn pause.
cut_in_rate = share of mid-turn pauses called complete (each one is an interruption).
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def classification_metrics(labels, probs, threshold: float) -> dict:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    both_classes = 0 < labels.sum() < len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(len(labels)),
        "pr_auc": float(average_precision_score(labels, probs)) if both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(labels, probs)) if both_classes else float("nan"),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "cut_in_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "accuracy": float((tp + tn) / len(labels)) if len(labels) else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def threshold_for_cut_in_rate(labels, probs, max_cut_in_rate: float) -> float:
    """Lowest threshold whose cut-in rate stays <= max_cut_in_rate."""
    labels = np.asarray(labels).astype(int)
    neg = np.sort(np.asarray(probs, dtype=float)[labels == 0])[::-1]
    if len(neg) == 0:
        return 0.5
    allowed = int(math.floor(max_cut_in_rate * len(neg)))
    if allowed >= len(neg):
        return 1e-6
    return float(min(np.nextafter(neg[allowed], 1.0), 1.0))


def recall_at_cut_in(labels, probs, max_cut_in_rate: float = 0.05) -> float:
    """Recall at the best threshold for this data (threshold-free ranking quality at the operating point)."""
    t = threshold_for_cut_in_rate(labels, probs, max_cut_in_rate)
    return classification_metrics(labels, probs, t)["recall"]


def bootstrap_ci(labels, probs, stat, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of stat(labels, probs) by resampling clips. Reuse the seed across models so CIs are paired."""
    labels, probs = np.asarray(labels), np.asarray(probs)
    rng = np.random.default_rng(seed)
    vals = [stat(labels[i], probs[i]) for i in (rng.integers(0, len(labels), len(labels)) for _ in range(n))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
