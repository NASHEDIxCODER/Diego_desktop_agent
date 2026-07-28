"""
Structured logging for Leo Desktop Assistant.

Provides JSON-formatted logging with:
- Correlation IDs for request tracing
- Subsystem IDs for filtering
- Latency metrics
- Memory metrics
- Context fields
"""

import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# Thread-local correlation ID
_correlation_id: Optional[str] = None
_subsystem_id: Optional[str] = None


def set_correlation_id(cid: Optional[str] = None) -> str:
    """
    Set or generate a correlation ID for the current request.

    Args:
        cid: Optional correlation ID. If None, generates a new one.

    Returns:
        The correlation ID.
    """
    global _correlation_id
    if cid is None:
        cid = str(uuid.uuid4())[:8]
    _correlation_id = cid
    return cid


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID."""
    return _correlation_id


def set_subsystem_id(sid: str) -> None:
    """Set the current subsystem ID."""
    global _subsystem_id
    _subsystem_id = sid


def get_subsystem_id() -> Optional[str]:
    """Get the current subsystem ID."""
    return _subsystem_id


class StructuredFormatter(logging.Formatter):
    """JSON-structured log formatter with correlation IDs and metrics."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Correlation ID
        cid = get_correlation_id()
        if cid:
            log_entry["correlation_id"] = cid

        # Subsystem ID
        sid = get_subsystem_id()
        if sid:
            log_entry["subsystem"] = sid

        # Extra fields
        if hasattr(record, "extra"):
            log_entry["extra"] = record.extra

        # Duration if available
        if hasattr(record, "duration"):
            log_entry["duration_ms"] = round(record.duration * 1000, 2)

        # Exception info
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

    # Set levels for all Leo subsystems
    for subsystem in ["__main__", "core", "nlp", "plugins", "voice",
                      "memory", "ai", "telemetry", "auth", "config"]:
        logging.getLogger(subsystem).setLevel(getattr(logging, level.upper(), logging.INFO))

    # Suppress noisy third-party loggers
    logging.getLogger("telethon").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("TTS").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a structured logger for a module."""
    return logging.getLogger(name)


class TimerContext:
    """
    Context manager for timing operations.

    Usage:
        with TimerContext(logger, "operation_name"):
            do_something()
    """

    def __init__(self, logger: logging.Logger, operation: str,
                 extra: Optional[Dict] = None):
        self._logger = logger
        self._operation = operation
        self._extra = extra or {}
        self._start: Optional[float] = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self._start
        self._logger.info("Timer [%s]: %.2fms", self._operation, duration * 1000,
                         extra={"duration": duration, **self._extra})