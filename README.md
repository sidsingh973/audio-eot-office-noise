# Audio end-of-turn detection in office noise

A voice agent has to decide at every pause whether the caller has **finished** (reply now) or is only **pausing** (keep listening). This project trains a small audio end-of-turn model three ways:
- on native speech (A)
- on speech with office noise added (B)
- on B's audio after WebRTC noise suppression (C)

The model is a Whisper-tiny encoder with LoRA adapters and a classifier head. All three versions are tested on an office noise none of them heard in training and compared with Pipecat's open [Smart Turn v3.2](https://github.com/pipecat-ai/smart-turn). The best model is served behind a FastAPI endpoint, with a Docker image.

| Read | |
|---|---|
| [`SOLUTION.md`](SOLUTION.md) | Short write-up of the whole solution |
| [`presentation.ipynb`](presentation.ipynb) | Walkthrough with charts, audio and live API tests, all outputs saved |
| [`SERVING.md`](SERVING.md) | API contract, load tests and Docker |
| [`reports/lora_eval.md`](reports/lora_eval.md) | Full evaluation tables |
| [`CREDITS.md`](CREDITS.md) | Credits and licenses |

## Results

All four models were scored on the same 7,820 English test clips, with an unseen office noise ("office-5").

| Model | PR-AUC, 10 dB | PR-AUC, 0 dB | Turn ends answered quickly at ≤5% cut-in, 0 dB | Interruptions at the deployed threshold, 0 dB |
|---|---:|---:|---:|---:|
| Smart Turn v3.2 | 0.973 | 0.918 | 62% | 20.3% |
| A: native speech | 0.978 | 0.930 | 65% | 14.2% |
| **B: + office noise** | **0.981** | **0.947** | **73%** | 9.9% |
| C: + office noise → noise suppression | 0.978 | 0.946 | 72% | **9.0%** |

- **Training on office noise (B)** makes the model robust to a different office noise, at no cost on clean audio.
- **Noise suppression (C) adds nothing** over B.
- **Serving:** 20.8 ms per request, and about 120 requests/s at p99 under 100 ms on an Apple M5 laptop CPU.

## Quick start: run the API

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

EOT_MAX_INFLIGHT=8 uvicorn service.app:app --port 8000
# in a second terminal:
python scripts/client.py samples/*.wav
curl -s localhost:8000/predict -H 'content-type: audio/wav' --data-binary @samples/07c85f8b_complete.wav
```

**Docker:**

```bash
docker build -t eot-audio .
docker run --rm -p 8000:8000 --cpus 4 -e EOT_MAX_INFLIGHT=4 eot-audio
```

The trained models are in `runs/A`, `runs/B` and `runs/C`. The API serves `runs/B`; set `EOT_MODEL_DIR` to serve another.

## Reproduce from scratch

The data (about 11 GB after filtering) and the two office-noise recordings are **not** in this repository.

1. **Get the noise.** Download "IT office 3" and "IT office 5" by thellywellyn from Pixabay. Save them in the repo root as `thellywellyn-it-office-3-535970.mp3` and `thellywellyn-it-office-5-535971.mp3`.
2. **Get the data:** `python -m eot_lora.download`. This streams Pipecat's Smart Turn v3.2 dataset (46 GB) and keeps only the English clips.
3. **Build the training and test audio:** `python -m eot_lora.prepare_data`. This splits train/validation, mixes in the office noise, runs WebRTC noise suppression, and builds 12 test variants.
4. **Compute features:** `python -m eot_lora.featurize`
5. **Train:** `python -m eot_lora.train --condition A --epochs 2`, then the same for `B` and `C`.
6. **Evaluate:** `python -m eot_lora.evaluate`. This scores A, B, C and Smart Turn v3.2 (downloaded automatically) and writes `reports/lora_eval.md`.
7. **Export:** `python -m eot_lora.export --condition B` writes the ONNX model and serving config.
8. **Serving checks:**
   - `python scripts/check_parity.py`
   - `bash scripts/run_stress_grid.sh`
   - `python scripts/build_notebook.py`, which rebuilds the notebook and needs the API running on port 8000

## Layout

| Path | |
|---|---|
| `eot_lora/` | Data download and preparation, noise and noise suppression, features, model, training, evaluation, export, and the torch-free predictor (`serving.py`) |
| `service/app.py` | FastAPI service |
| `scripts/` | Inference client, stress test, model benchmark, parity check, notebook builder |
| `runs/{A,B,C}/` | Trained models: LoRA checkpoints, ONNX exports, logs, serving config |
| `samples/` | 20 labelled test clips (office-5 noise at 10 dB) for trying the API |
| `reports/` | Evaluation, benchmark and stress-test results |

## Credits and license

This project uses Pipecat's Smart Turn code design, model and data (BSD 2-Clause, © Daily), OpenAI's Whisper, office recordings by thellywellyn on Pixabay, and many open-source libraries. See [`CREDITS.md`](CREDITS.md) for the full list with licenses.

No license has been chosen yet for this repository's own code. Third-party components keep their own licenses.
