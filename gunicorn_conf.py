"""
Gunicorn configuration for the Voice Authentication API.

GPU note: each worker loads a full copy of every model into GPU memory. Run
ONE worker per GPU unless you have headroom — set WEB_CONCURRENCY accordingly.
Concurrency within a worker is bounded by the GPU anyway, so scale out with
more pods/GPUs rather than more workers per GPU.
"""

import os

bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

# Default to a single worker (one GPU). Override with WEB_CONCURRENCY.
workers = int(os.getenv("WEB_CONCURRENCY", "1"))
worker_class = "uvicorn.workers.UvicornWorker"

# Model inference can be slow; don't let gunicorn kill long requests.
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
graceful_timeout = 30
keepalive = 5

# Recycle workers periodically to bound any slow memory growth.
max_requests = int(os.getenv("GUNICORN_MAX_REQUESTS", "1000"))
max_requests_jitter = 100

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()
