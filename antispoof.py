"""
Anti-spoofing / liveness gate (Phase 3).

Voice verification alone does not stop replay attacks or TTS/voice-cloning —
a stolen recording or a cloned voice will match the enrolled embedding. This
module adds a presentation-attack-detection (PAD) gate that runs BEFORE
verification and rejects synthetic/replayed audio.

Design
------
* Pluggable: points at any HuggingFace audio-classification checkpoint
  (e.g. an AASIST / wav2vec2-based spoof detector) via ANTISPOOF_MODEL.
* Lazy: the model is loaded on first use, not at import, so the rest of the
  app starts even when anti-spoofing is disabled or the weights are absent.
* Fail-closed by default: if the detector can't load and ANTISPOOF_FAIL_CLOSED
  is true, the gate rejects (safest for banking-grade deployments).
* The decision logic is pure and unit-tested independently of any model.

The concrete checkpoint and its threshold MUST be validated on real
attack/bonafide data (e.g. ASVspoof 5) on the GPU host before enabling.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from config import settings

logger = logging.getLogger("voice_auth.antispoof")

_SPOOF_KEYWORDS = ("spoof", "fake", "deepfake", "synthetic", "attack")
_BONAFIDE_KEYWORDS = ("bona", "real", "genuine", "human")


@dataclass
class AntiSpoofResult:
    spoof_probability: float
    is_spoof: bool
    label: str


def is_spoof(spoof_probability: float, threshold: float) -> bool:
    """Pure decision: reject when P(spoof) meets/exceeds the threshold."""
    return spoof_probability >= threshold


def resolve_spoof_index(id2label: dict, override: int = -1) -> int:
    """Decide which model output index corresponds to the 'spoof' class.

    Priority: explicit override -> label-name match -> assume binary index 1.
    """
    if override is not None and override >= 0:
        return override

    labels = {int(k): str(v).lower() for k, v in id2label.items()}

    for idx, name in labels.items():
        if any(kw in name for kw in _SPOOF_KEYWORDS):
            return idx
    # If we can identify the bonafide class in a binary head, spoof is the other.
    if len(labels) == 2:
        for idx, name in labels.items():
            if any(kw in name for kw in _BONAFIDE_KEYWORDS):
                return 1 - idx
        return 1  # conventional: index 1 = spoof
    # Multi-class with no obvious spoof label: default to last index.
    return max(labels) if labels else 0


class AntiSpoofDetector:
    """Lazy wrapper around a HF audio-classification spoof detector."""

    def __init__(self, model_id: str, device: Optional[torch.device] = None):
        self.model_id = model_id
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = None
        self._extractor = None
        self._spoof_index = None
        self._load_failed = False

    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        if self._load_failed:
            return False
        try:
            from transformers import (
                AutoFeatureExtractor,
                AutoModelForAudioClassification,
            )

            logger.info("Loading anti-spoof model: %s", self.model_id)
            self._extractor = AutoFeatureExtractor.from_pretrained(self.model_id)
            self._model = AutoModelForAudioClassification.from_pretrained(
                self.model_id
            ).to(self.device)
            self._model.eval()
            id2label = getattr(self._model.config, "id2label", {0: "0", 1: "1"})
            self._spoof_index = resolve_spoof_index(
                id2label, settings.ANTISPOOF_SPOOF_INDEX
            )
            logger.info(
                "Anti-spoof model ready (spoof_index=%s, labels=%s)",
                self._spoof_index, id2label,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - we want to degrade gracefully
            self._load_failed = True
            logger.error("Failed to load anti-spoof model '%s': %s", self.model_id, exc)
            return False

    @torch.inference_mode()
    def score(self, signal: torch.Tensor, sample_rate: int) -> Optional[AntiSpoofResult]:
        """Return the spoof result, or None if the model is unavailable."""
        if not self._ensure_loaded():
            return None

        # Collapse to mono 1-D float waveform on CPU for the feature extractor.
        wav = signal.detach().float()
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        wav = wav.cpu().numpy()

        inputs = self._extractor(
            wav, sampling_rate=sample_rate, return_tensors="pt"
        ).to(self.device)
        logits = self._model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        spoof_prob = float(probs[self._spoof_index].item())
        label = self._model.config.id2label.get(self._spoof_index, "spoof")
        return AntiSpoofResult(
            spoof_probability=spoof_prob,
            is_spoof=is_spoof(spoof_prob, settings.ANTISPOOF_THRESHOLD),
            label=str(label),
        )


_detector: Optional[AntiSpoofDetector] = None


def _get_detector() -> Optional[AntiSpoofDetector]:
    global _detector
    if not settings.ANTISPOOF_MODEL:
        return None
    if _detector is None:
        _detector = AntiSpoofDetector(settings.ANTISPOOF_MODEL)
    return _detector


def gate(signal: torch.Tensor, sample_rate: int) -> Tuple[bool, str]:
    """Anti-spoofing gate. Returns (allowed, detail).

    * Disabled       -> (True, "disabled")
    * Genuine        -> (True, "<prob>")
    * Spoof detected -> (False, reason)
    * Model missing  -> fail-closed (reject) or fail-open per config.
    """
    if not settings.ANTISPOOF_ENABLED:
        return True, "anti-spoof disabled"

    detector = _get_detector()
    if detector is None:
        msg = "ANTISPOOF_ENABLED but ANTISPOOF_MODEL is not set"
        logger.error(msg)
        return (not settings.ANTISPOOF_FAIL_CLOSED), msg

    result = detector.score(signal, sample_rate)
    if result is None:
        # Model failed to load.
        if settings.ANTISPOOF_FAIL_CLOSED:
            return False, "anti-spoof model unavailable (failing closed)"
        return True, "anti-spoof model unavailable (failing open)"

    if result.is_spoof:
        return False, (
            f"Spoofed/synthetic audio detected "
            f"(P(spoof)={result.spoof_probability:.3f} >= {settings.ANTISPOOF_THRESHOLD})"
        )
    return True, f"genuine (P(spoof)={result.spoof_probability:.3f})"
