"""Unit tests for auth + rate limiting. No GPU/models/DB required."""

import types

import pytest
from fastapi import HTTPException

import security


def _fake_request(host="1.2.3.4", forwarded=None):
    headers = {}
    if forwarded:
        headers["x-forwarded-for"] = forwarded
    return types.SimpleNamespace(
        client=types.SimpleNamespace(host=host),
        headers=types.SimpleNamespace(get=lambda k, d=None: headers.get(k, d)),
    )


# ---- API key ----

async def test_api_key_skipped_when_auth_disabled(monkeypatch):
    monkeypatch.setattr(security, "AUTH_ENABLED", False)
    # Should not raise even with no key.
    await security.require_api_key(None)


async def test_api_key_rejects_missing_key(monkeypatch):
    monkeypatch.setattr(security, "AUTH_ENABLED", True)
    monkeypatch.setattr(security, "API_KEYS", {"secret-key"})
    with pytest.raises(HTTPException) as exc:
        await security.require_api_key(None)
    assert exc.value.status_code == 401


async def test_api_key_accepts_valid_key(monkeypatch):
    monkeypatch.setattr(security, "AUTH_ENABLED", True)
    monkeypatch.setattr(security, "API_KEYS", {"secret-key"})
    await security.require_api_key("secret-key")


async def test_api_key_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(security, "AUTH_ENABLED", True)
    monkeypatch.setattr(security, "API_KEYS", {"secret-key"})
    with pytest.raises(HTTPException) as exc:
        await security.require_api_key("wrong-key")
    assert exc.value.status_code == 401


# ---- Rate limiting ----

async def test_rate_limit_allows_then_blocks(monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT_PER_MIN", 3)
    security._hits.clear()
    req = _fake_request(host="10.0.0.1")

    for _ in range(3):
        await security.rate_limit(req)  # first 3 allowed

    with pytest.raises(HTTPException) as exc:
        await security.rate_limit(req)  # 4th blocked
    assert exc.value.status_code == 429


async def test_rate_limit_isolated_per_client(monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT_PER_MIN", 1)
    security._hits.clear()

    await security.rate_limit(_fake_request(host="10.0.0.2"))
    # Different client should still be allowed.
    await security.rate_limit(_fake_request(host="10.0.0.3"))


async def test_rate_limit_disabled_when_zero(monkeypatch):
    monkeypatch.setattr(security, "RATE_LIMIT_PER_MIN", 0)
    security._hits.clear()
    req = _fake_request(host="10.0.0.4")
    for _ in range(100):
        await security.rate_limit(req)  # never blocks when disabled
