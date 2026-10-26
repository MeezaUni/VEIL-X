"""Structured (JSON-lines) logging.

Usage::

    from veilx.logging_utils import configure_logging, get_logger
    configure_logging(level="INFO")
    log = get_logger(__name__)
    log.info("split assigned", extra={"fields": {"dataset": "digiface", "n_ids": 2000}})

Privacy rule: never log image pixels, embeddings, absolute user paths or personal names.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

_RESERVED = set(logging.LogRecord("x", 0, "x", 0, "x", None, None).__dict__) | {"message"}


class JsonFormatter(logging.Formatter):
    """Format records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload["fields"] = fields
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key != "fields" and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, sort_keys=True)


def configure_logging(
    level: str | int = "INFO",
    *,
    log_file: Path | None = None,
    stream: TextIO | None = None,
) -> logging.Logger:
    """Configure the ``veilx`` logger hierarchy with JSON output. Idempotent."""
    root = logging.getLogger("veilx")
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    formatter = JsonFormatter()
    stream_handler = logging.StreamHandler(stream or sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    root.propagate = False
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a logger inside the ``veilx`` namespace."""
    return logging.getLogger(name if name.startswith("veilx") else f"veilx.{name}")
