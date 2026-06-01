"""Tests for the pluggable embedding backend. No GPU/real models required."""

import sys
import types

import pytest
import torch

import embeddings


@pytest.fixture(autouse=True)
def _reset_backend():
    embeddings._backend = None
    yield
    embeddings._backend = None


def test_default_backend_is_ecapa(monkeypatch):
    monkeypatch.setattr(embeddings.settings, "EMBEDDING_BACKEND", "ecapa")
    assert embeddings.get_backend().name == "ecapa"


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setattr(embeddings.settings, "EMBEDDING_BACKEND", "nope")
    with pytest.raises(ValueError):
        embeddings.get_backend()


def test_campplus_backend_selected(monkeypatch):
    monkeypatch.setattr(embeddings.settings, "EMBEDDING_BACKEND", "campplus")
    monkeypatch.setattr(embeddings.settings, "CAMPLUS_MODEL", "english")
    backend = embeddings.get_backend()
    assert backend.name == "campplus"
    assert backend.model_id == "english"


def test_ecapa_extract_uses_global_model(monkeypatch):
    """ECAPA path delegates to models.speaker_encoder.encode_batch without
    importing the real (heavy) models module."""
    monkeypatch.setattr(embeddings.settings, "EMBEDDING_BACKEND", "ecapa")

    class FakeEncoder:
        def encode_batch(self, signal):
            return torch.ones(1, 1, 192)

    fake_models = types.ModuleType("models")
    fake_models.speaker_encoder = FakeEncoder()
    monkeypatch.setitem(sys.modules, "models", fake_models)

    out = embeddings.extract_embedding(torch.randn(1, 16000))
    assert out.shape == (1, 1, 192)
    assert out.dtype == torch.float32


def test_ecapa_extract_raises_when_model_missing(monkeypatch):
    monkeypatch.setattr(embeddings.settings, "EMBEDDING_BACKEND", "ecapa")
    fake_models = types.ModuleType("models")
    fake_models.speaker_encoder = None
    monkeypatch.setitem(sys.modules, "models", fake_models)
    with pytest.raises(RuntimeError):
        embeddings.extract_embedding(torch.randn(1, 16000))


def test_amp_context_is_noop_on_cpu(monkeypatch):
    # On a CPU-only host the autocast context must be a harmless no-op.
    monkeypatch.setattr(embeddings.settings, "USE_AMP", True)
    with embeddings._amp_context():
        x = torch.ones(3) + 1
    assert torch.equal(x, torch.full((3,), 2.0))


def test_as_batched_shapes():
    assert embeddings._as_batched(torch.randn(192)).shape == (1, 1, 192)
    assert embeddings._as_batched(torch.randn(1, 192)).shape == (1, 1, 192)
    assert embeddings._as_batched(torch.randn(1, 1, 192)).shape == (1, 1, 192)
