"""Tests for the anti-spoof decision + label-resolution + gate logic.

These cover everything that does NOT require model weights; the model
inference path is validated on the GPU host with a real checkpoint.
"""

import torch

import antispoof
from antispoof import is_spoof, resolve_spoof_index, gate


# ---- pure decision ----

def test_is_spoof_threshold():
    assert is_spoof(0.9, 0.5) is True
    assert is_spoof(0.5, 0.5) is True  # boundary rejects
    assert is_spoof(0.49, 0.5) is False


# ---- spoof-index resolution ----

def test_resolve_index_by_label_name():
    assert resolve_spoof_index({0: "bonafide", 1: "spoof"}) == 1
    assert resolve_spoof_index({0: "fake", 1: "real"}) == 0


def test_resolve_index_override_wins():
    assert resolve_spoof_index({0: "spoof", 1: "bonafide"}, override=1) == 1


def test_resolve_index_binary_infers_complement():
    # Only the bonafide class is named; spoof must be the other index.
    assert resolve_spoof_index({0: "genuine", 1: "LABEL_1"}) == 1


def test_resolve_index_defaults_to_one_for_unknown_binary():
    assert resolve_spoof_index({0: "LABEL_0", 1: "LABEL_1"}) == 1


# ---- gate behaviour (model mocked / disabled) ----

def test_gate_disabled_allows(monkeypatch):
    monkeypatch.setattr(antispoof.settings, "ANTISPOOF_ENABLED", False)
    allowed, _ = gate(torch.randn(16000), 16000)
    assert allowed is True


def test_gate_enabled_without_model_fails_closed(monkeypatch):
    monkeypatch.setattr(antispoof.settings, "ANTISPOOF_ENABLED", True)
    monkeypatch.setattr(antispoof.settings, "ANTISPOOF_MODEL", "")
    monkeypatch.setattr(antispoof.settings, "ANTISPOOF_FAIL_CLOSED", True)
    allowed, detail = gate(torch.randn(16000), 16000)
    assert allowed is False


def test_gate_enabled_without_model_can_fail_open(monkeypatch):
    monkeypatch.setattr(antispoof.settings, "ANTISPOOF_ENABLED", True)
    monkeypatch.setattr(antispoof.settings, "ANTISPOOF_MODEL", "")
    monkeypatch.setattr(antispoof.settings, "ANTISPOOF_FAIL_CLOSED", False)
    allowed, _ = gate(torch.randn(16000), 16000)
    assert allowed is True


def test_gate_rejects_detected_spoof(monkeypatch):
    monkeypatch.setattr(antispoof.settings, "ANTISPOOF_ENABLED", True)
    monkeypatch.setattr(antispoof.settings, "ANTISPOOF_MODEL", "dummy/model")
    monkeypatch.setattr(antispoof.settings, "ANTISPOOF_THRESHOLD", 0.5)

    class FakeDetector:
        def score(self, signal, sr):
            return antispoof.AntiSpoofResult(0.92, True, "spoof")

    monkeypatch.setattr(antispoof, "_get_detector", lambda: FakeDetector())
    allowed, detail = gate(torch.randn(16000), 16000)
    assert allowed is False
    assert "spoof" in detail.lower()


def test_gate_allows_genuine(monkeypatch):
    monkeypatch.setattr(antispoof.settings, "ANTISPOOF_ENABLED", True)
    monkeypatch.setattr(antispoof.settings, "ANTISPOOF_MODEL", "dummy/model")

    class FakeDetector:
        def score(self, signal, sr):
            return antispoof.AntiSpoofResult(0.04, False, "bonafide")

    monkeypatch.setattr(antispoof, "_get_detector", lambda: FakeDetector())
    allowed, _ = gate(torch.randn(16000), 16000)
    assert allowed is True
