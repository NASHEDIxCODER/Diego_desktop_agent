"""
Structured logging for Leo Desktop Assistant.

Provides JSON-formatted logging with context fields.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "extra"):
            log_entry["extra"] = record.extra

        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
               Defaults to settings.LOG_LEVEL or INFO.
    """
    if level is None:
        from config.settings import settings
        level = settings.LOG_LEVEL

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    logging.getLogger("__main__").setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("core").setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("nlp").setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("plugins").setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("voice").setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("memory").setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("ai").setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("telemetry").setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Get a structured logger for a module."""
    return logging.getLogger(name)