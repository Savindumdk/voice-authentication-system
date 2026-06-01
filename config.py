"""
Centralized application configuration (12-factor).

Single source of truth for environment-driven settings. Existing modules keep
their own `AppConfig`/`DatabaseConfig` for backward compatibility; this module
is the canonical place new code should read settings from.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _get_list(name: str, default: str) -> list:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


class Settings:
    """Immutable-ish settings snapshot read once at import."""

    # --- Database ---
    MONGODB_URI: str = os.getenv("MONGODB_URI", "")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "voice_auth")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "user_data")

    # --- Models ---
    SPEAKER_VERIFIER_MODEL: str = os.getenv(
        "SPEAKER_VERIFIER_MODEL", "speechbrain/spkrec-ecapa-voxceleb"
    )
    HF_AUTH_TOKEN: str = os.getenv("HF_AUTH_TOKEN", "")
    DEVICE: str = os.getenv("DEVICE", "cuda")

    # Speaker-embedding backend: "ecapa" (default) | "campplus".
    # Switching requires re-enrolling users (different embedding space/dim).
    EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "ecapa").strip().lower()
    # wespeaker tag / local dir or modelscope id for CAM++.
    CAMPLUS_MODEL: str = os.getenv("CAMPLUS_MODEL", "")

    # --- Thresholds / EWMA ---
    VERIFICATION_THRESHOLD: float = _get_float("VERIFICATION_THRESHOLD", 0.50)
    EWMA_ENABLED: bool = _get_bool("EWMA_ENABLED", True)
    EWMA_ADAPTATION_THRESHOLD: float = _get_float("EWMA_ADAPTATION_THRESHOLD", 0.70)
    EWMA_LEARNING_RATE: float = _get_float("EWMA_LEARNING_RATE", 0.1)

    # --- Security ---
    API_KEYS: list = _get_list("API_KEYS", os.getenv("API_KEY", ""))
    ALLOWED_ORIGINS: list = _get_list(
        "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    )
    RATE_LIMIT_PER_MIN: int = _get_int("RATE_LIMIT_PER_MIN", 60)

    # --- Limits ---
    MAX_UPLOAD_MB: float = _get_float("MAX_UPLOAD_MB", 2.0)

    # --- Anti-spoofing (Phase 3) ---
    # Disabled by default: enable once a checkpoint is configured + validated.
    ANTISPOOF_ENABLED: bool = _get_bool("ANTISPOOF_ENABLED", False)
    # HF audio-classification model id (e.g. an AASIST / wav2vec2 spoof detector).
    ANTISPOOF_MODEL: str = os.getenv("ANTISPOOF_MODEL", "")
    # Reject when P(spoof) >= threshold.
    ANTISPOOF_THRESHOLD: float = _get_float("ANTISPOOF_THRESHOLD", 0.5)
    # If the detector can't load, fail CLOSED (reject) — safest for high-assurance.
    ANTISPOOF_FAIL_CLOSED: bool = _get_bool("ANTISPOOF_FAIL_CLOSED", True)
    # Optional manual override of which output index = "spoof" (else auto-resolved).
    ANTISPOOF_SPOOF_INDEX: int = _get_int("ANTISPOOF_SPOOF_INDEX", -1)

    # --- Pipeline / performance (Phase 2) ---
    # "always" = current behaviour (VAD+diarization+separation every request).
    # "off"    = skip the heavy multi-speaker pipeline (fastest; for cooperative
    #            single-speaker 1:1 auth). Benchmark EER impact before enabling.
    HEAVY_PIPELINE_MODE: str = os.getenv("HEAVY_PIPELINE_MODE", "always").strip().lower()
    # CUDA mixed-precision (fp16) inference. No-op on CPU. Off by default because
    # it slightly shifts embedding numerics — enable + recalibrate the threshold.
    USE_AMP: bool = _get_bool("USE_AMP", False)
    # Max concurrent model inferences per worker (bounds GPU oversubscription).
    MAX_CONCURRENT_INFERENCE: int = _get_int("MAX_CONCURRENT_INFERENCE", 2)

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_JSON: bool = _get_bool("LOG_JSON", False)

    @property
    def auth_enabled(self) -> bool:
        return len(self.API_KEYS) > 0


settings = Settings()
