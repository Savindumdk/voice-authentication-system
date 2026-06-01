# GPU-ready image for the Voice Authentication API.
# Base matches the project's torch build (2.5.1 / CUDA 12.1).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch

# System deps: ffmpeg (pydub), libsndfile (soundfile), git (HF downloads).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn==23.0.0

# App code.
COPY . .

# Model weights are NOT baked in (large + some are gated). They download on
# first run into the mounted /models and ./pretrained_models volumes.
RUN mkdir -p /models/hf /models/torch pretrained_models

EXPOSE 8000

# Basic container healthcheck hitting the liveness endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Gunicorn manages uvicorn workers; tuning lives in gunicorn_conf.py.
CMD ["gunicorn", "-c", "gunicorn_conf.py", "main:app"]
