"""
Structured logging setup.

Replaces scattered print() calls with a configurable logger. Call
`configure_logging()` once at startup. Supports plain or JSON output
(LOG_JSON=true) so logs are ingestible by Loki/CloudWatch/ELK in production.
"""

import json
import logging
import sys
from datetime import datetime, timezone

from config import settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Attach any structured extras.
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord("", 0, "", 0, "", (), None).__dict__ and key != "message":
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging() -> logging.Logger:
    """Configure the root logger once; return the app logger."""
    root = logging.getLogger()
    root.setLevel(settings.LOG_LEVEL)

    # Clear any pre-existing handlers (uvicorn/lightning may have added some).
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    if settings.LOG_JSON:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
    root.addHandler(handler)

    # Tame noisy third-party loggers (kept from the original suppression intent).
    for noisy in (
        "pytorch_lightning",
        "lightning.pytorch",
        "transformers",
        "speechbrain",
        "pyannote",
        "huggingface_hub",
    ):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    return logging.getLogger("voice_auth")


logger = logging.getLogger("voice_auth")
