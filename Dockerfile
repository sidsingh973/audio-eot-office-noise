# Serving image for the audio end-of-turn model: FastAPI + ONNX Runtime + numpy (no torch).
#   docker build -t eot-audio .                          # bakes in runs/B (--build-arg MODEL_DIR=runs/C for another)
#   docker run --rm -p 8000:8000 --cpus 4 -e EOT_MAX_INFLIGHT=4 eot-audio
# One process running EOT_MAX_INFLIGHT single-threaded inferences balances per request; set it to the CPU count.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EOT_MODEL_DIR=/app/model \
    EOT_THREADS=1 \
    EOT_MAX_INFLIGHT=4 \
    WORKERS=1 \
    OMP_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt && useradd --create-home --uid 1000 app

COPY eot_lora/serving.py eot_lora/
COPY service/app.py service/
ARG MODEL_DIR=runs/B
COPY ${MODEL_DIR}/model.onnx ${MODEL_DIR}/mel_filters.npy ${MODEL_DIR}/serving.json model/

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["sh", "-c", "exec uvicorn service.app:app --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]
