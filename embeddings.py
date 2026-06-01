"""
Pluggable speaker-embedding extraction (Phase 3).

Lets the embedding model be swapped via config without touching the endpoints:

    EMBEDDING_BACKEND=ecapa      # default — current SpeechBrain ECAPA-TDNN (192-d)
    EMBEDDING_BACKEND=campplus   # CAM++ (faster, lower EER) via wespeaker/modelscope

The ECAPA path is behaviour-preserving: it wraps the already-loaded global
`speaker_encoder.encode_batch` under no_grad (+ optional fp16 autocast). The
CAM++ path is a validated-on-GPU integration point.

⚠️  IMPORTANT — backends are NOT interchangeable at runtime for existing data:
different models produce different embedding spaces and dimensions (ECAPA=192,
CAM++≈512). Switching EMBEDDING_BACKEND requires RE-ENROLLING all users; old
embeddings won't compare meaningfully against new ones. `matching.py` already
skips dimension-mismatched embeddings rather than crashing.
"""

import contextlib
import logging
from typing import Optional

import torch

from config import settings

logger = logging.getLogger("voice_auth.embeddings")

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _amp_context():
    """fp16 autocast on CUDA when USE_AMP is on; otherwise a no-op."""
    if settings.USE_AMP and _DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _as_batched(emb: torch.Tensor) -> torch.Tensor:
    """Normalize output to ECAPA's [batch, 1, dim] convention."""
    if emb.dim() == 1:
        return emb.view(1, 1, -1)
    if emb.dim() == 2:
        return emb.unsqueeze(1)
    return emb


class _EcapaBackend:
    name = "ecapa"

    def extract(self, signal: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        # Lazy import avoids an import cycle and premature model load.
        import models

        model = models.speaker_encoder
        if model is None:
            raise RuntimeError("ECAPA speaker_encoder is not loaded")
        with torch.no_grad(), _amp_context():
            emb = model.encode_batch(signal)
        # Cast back to fp32 so downstream cosine/EWMA math is stable.
        return emb.float()


class _CampPlusBackend:
    """CAM++ via wespeaker (preferred) or modelscope. Validate on the GPU host.

    Set CAMPLUS_MODEL to a wespeaker language tag / local dir, or a modelscope
    model id (e.g. 'iic/speech_campplus_sv_en_voxceleb_16k').
    """

    name = "campplus"

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._backend = None  # "wespeaker" | "modelscope"

    def _ensure_loaded(self):
        if self._model is not None:
            return
        # Try wespeaker first.
        try:
            import wespeaker

            try:
                self._model = wespeaker.load_model_local(self.model_id)
            except Exception:
                self._model = wespeaker.load_model(self.model_id or "english")
            if _DEVICE.type == "cuda":
                self._model.set_device("cuda")
            self._backend = "wespeaker"
            logger.info("CAM++ loaded via wespeaker (%s)", self.model_id)
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("wespeaker unavailable (%s); trying modelscope", exc)

        # Fall back to modelscope 3D-Speaker pipeline.
        from modelscope.pipelines import pipeline as ms_pipeline

        self._model = ms_pipeline(task="speaker-verification", model=self.model_id)
        self._backend = "modelscope"
        logger.info("CAM++ loaded via modelscope (%s)", self.model_id)

    def extract(self, signal: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        self._ensure_loaded()
        wav = signal.detach().float()
        if wav.dim() > 1:
            wav = wav.mean(dim=0)

        with torch.no_grad(), _amp_context():
            if self._backend == "wespeaker":
                # wespeaker expects a 2-D [1, samples] CPU tensor.
                emb = self._model.extract_embedding_from_pcm(
                    wav.unsqueeze(0).cpu(), sample_rate
                )
                emb = torch.as_tensor(emb)
            else:  # modelscope
                out = self._model([wav.cpu().numpy()], output_emb=True)
                emb = torch.as_tensor(out["embs"][0])
        return _as_batched(emb.float().to(_DEVICE))


_backend = None


def get_backend():
    """Return the configured embedding backend singleton."""
    global _backend
    if _backend is not None:
        return _backend
    name = settings.EMBEDDING_BACKEND
    if name == "ecapa":
        _backend = _EcapaBackend()
    elif name == "campplus":
        _backend = _CampPlusBackend(settings.CAMPLUS_MODEL)
    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {name!r}")
    return _backend


def extract_embedding(signal: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """Extract a speaker embedding using the configured backend.

    Returns a [batch, 1, dim] fp32 tensor on the active device — drop-in
    compatible with the previous `encode_batch` outputs.
    """
    return get_backend().extract(signal, sample_rate)


def backend_name() -> str:
    return settings.EMBEDDING_BACKEND
