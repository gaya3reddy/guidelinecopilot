from __future__ import annotations

import json
import logging
import os
import sys


class _JsonFormatter(logging.Formatter):
    """Emits one JSON line per log record — Cloud Logging parses these automatically."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "severity": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        # Merge any extra fields passed via extra={...}
        for key, val in record.__dict__.items():
            if key not in {
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "name",
                "message",
            }:
                payload[key] = val

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger that emits:
    - JSON to stdout when LOG_FORMAT=json (or when running on Cloud Run / GCP)
    - Human-readable to stdout locally (default)
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured, don't add duplicate handlers

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "text").lower()

    # Cloud Run sets this env var automatically
    is_gcp = os.getenv("K_SERVICE") is not None
    use_json = log_format == "json" or is_gcp

    handler = logging.StreamHandler(sys.stdout)

    if use_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
        )

    logger.addHandler(handler)
    logger.setLevel(getattr(logging, log_level, logging.INFO))
    logger.propagate = False

    return logger
