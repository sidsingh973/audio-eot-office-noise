"""Export a trained condition to ONNX with Smart Turn's exact interface.

input_features float32 [B, 80, 800] -> logits float32 [B, 1], which (like Smart Turn's) are already
sigmoid probabilities. So runs/<condition>/model.onnx is a drop-in for happyrobot's eot/smart_turn.py.

  python -m eot_lora.export --condition B
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import onnxruntime as ort
import torch
from torch.export import Dim

from eot_lora.featurize import load as load_features
from eot_lora.model import load
from eot_lora.train import RUNS


class Probabilities(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.inner(input_features)).unsqueeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, choices=["A", "B", "C"])
    args = ap.parse_args()

    model = load(RUNS / args.condition / "model.pt")
    x = torch.from_numpy(np.asarray(load_features("test_native")[0][:32], dtype=np.float32))
    with torch.no_grad():
        ref = torch.sigmoid(model(x))
    merged = Probabilities(model.merge_and_unload()).eval()
    with torch.no_grad():
        got = merged(x)[:, 0]
    merge_diff = float((ref - got).abs().max())
    assert merge_diff < 1e-5, f"merged LoRA differs from adapter model by {merge_diff}"

    path = RUNS / args.condition / "model.onnx"
    torch.onnx.export(merged, (x[:2],), str(path), input_names=["input_features"], output_names=["logits"],
                      opset_version=18, dynamic_shapes={"input_features": {0: Dim.DYNAMIC}}, external_data=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"input_features": x.numpy()})[0]
    assert out.shape == (32, 1), out.shape
    onnx_diff = float(np.abs(out[:, 0] - ref.numpy()).max())
    assert onnx_diff < 1e-4, f"ONNX differs from PyTorch by {onnx_diff}"
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB); merge diff {merge_diff:.2e}, ONNX diff {onnx_diff:.2e}")
    write_serving_files(args.condition)


DESCRIPTIONS = {"A": "Pipecat-native English speech",
                "B": "English speech + office-3 noise (80% of clips, 5-20 dB SNR)",
                "C": "B's audio after WebRTC noise suppression"}


def write_serving_files(condition: str) -> None:
    """mel_filters.npy + serving.json next to model.onnx: everything eot_lora/serving.py needs to load the model."""
    from transformers import WhisperFeatureExtractor

    from eot_lora.metrics import threshold_for_cut_in_rate

    out = RUNS / condition
    np.save(out / "mel_filters.npy", WhisperFeatureExtractor(chunk_length=8).mel_filters.astype(np.float32))
    z = np.load(out / "val_probs.npz")
    cfg = {"condition": condition, "model": "model.onnx",
           "description": f"Whisper-tiny encoder + LoRA (merged) + classifier head; trained on {DESCRIPTIONS[condition]}",
           "threshold": threshold_for_cut_in_rate(z["labels"], z["probs"], 0.05),
           "threshold_rule": "lowest threshold with <=5% cut-in on this model's validation set",
           "sample_rate": 16000, "window_seconds": 8}
    (out / "serving.json").write_text(json.dumps(cfg, indent=2))
    print(f"wrote {out / 'serving.json'} (threshold {cfg['threshold']:.4f}) and mel_filters.npy")


if __name__ == "__main__":
    main()
