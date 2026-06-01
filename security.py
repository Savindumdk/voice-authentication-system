"""
Security primitives for the Voice Authentication API.

Phase 0 hardening:
  * API-key authentication (constant-time comparison)
  * Simple in-memory, per-IP rate limiting

These are deliberately dependency-free so they can be dropped in without
changing the runtime requirements.

PRODUCTION NOTES
----------------
* The rate limiter stores hit timestamps in process memory. With multiple
  uvicorn/gunicorn workers each worker has its own counter, so the effective
  limit is `RATE_LIMIT_PER_MIN * num_workers`. For real multi-worker / multi-pod
  deployments move this to a shared store (e.g. Redis via `slowapi`).
* API keys are coarse-grained service credentials. Browser/end-user flows should
  use short-lived per-user tokens (JWT/session) issued after login rather than a
  static key embedded in the page.
"""

import os
import time
import secrets
import logging
from collections import defaultdict, deque
from typing import Optional

from dotenv import load_dotenv
from fastapi import Request, HTTPException, status, Header

# Ensure env vars are available even if this module is imported very early.
load_dotenv()

logger = logging.getLogger("voice_auth.security")


# ----------------------------
# API-key authentication
# ----------------------------

def _load_api_keys() -> set:
    """Read accepted API keys from API_KEYS (comma-separated) or API_KEY."""
    raw = os.getenv("API_KEYS") or os.getenv("API_KEY") or ""
    return {k.strip() for k in raw.split(",") if k.strip()}


API_KEYS = _load_api_keys()
AUTH_ENABLED = len(API_KEYS) > 0

if not AUTH_ENABLED:
    logger.warning(
        "⚠️  No API_KEY/API_KEYS configured — API authentication is DISABLED. "
        "This is acceptable for local development only. Set API_KEY in the "
        "environment before deploying."
    )


async def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """FastAPI dependency that enforces a valid X-API-Key header.

    When no keys are configured the check is skipped (dev mode) so the local
    web UI keeps working. As soon as API_KEY is set in the environment the
    endpoints are protected.
    """
    if not AUTH_ENABLED:
        return

    if not x_api_key or not any(
        secrets.compare_digest(x_api_key, key) for key in API_KEYS
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "API-Key"},
        )


# ----------------------------
# In-memory per-IP rate limiting
# ----------------------------

RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
_WINDOW_SECONDS = 60.0
_hits: "defaultdict[str, deque]" = defaultdict(deque)


def _client_id(request: Request) -> str:
    """Best-effort client identifier, honoring a single proxy hop."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def rate_limit(request: Request) -> None:
    """FastAPI dependency that applies a sliding-window per-IP rate limit."""
    if RATE_LIMIT_PER_MIN <= 0:
        return

    client = _client_id(request)
    now = time.monotonic()
    bucket = _hits[client]

    # Drop timestamps older than the window.
    cutoff = now - _WINDOW_SECONDS
    while bucket and bucket[0] <= cutoff:
        bucket.popleft()

    if len(bucket) >= RATE_LIMIT_PER_MIN:
        retry_after = max(1, int(_WINDOW_SECONDS - (now - bucket[0])))
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please slow down and try again.",
            headers={"Retry-After": str(retry_after)},
        )

    bucket.append(now)
