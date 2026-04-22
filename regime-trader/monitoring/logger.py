"""Structured logger for the engine (Phase A8).

- JSON-friendly line format so logs can be shipped into any log aggregator.
- Rotating main/trades/alerts/regime files so the host disk doesn't fill up.
- Thin ``emit_event`` helper used by the orchestrator to broadcast structured
  operational events (reused by the FastAPI SSE stream).
"""
from __future__ import annotations

import json
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Iterable, Mapping


_STD_RECORD_KEYS = {
    "name",
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
    "message",
    "asctime",
    "taskName",
}


class JsonLineFormatter(logging.Formatter):
    """Serialize log records as one-line JSON for easy ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STD_RECORD_KEYS:
                continue
            if key.startswith("_"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    *,
    log_dir: Path | str,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 30,
) -> None:
    """Install rotating file handlers for main/trades/alerts/regime channels."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = JsonLineFormatter()

    # Remove previous handlers we installed (idempotent startup).
    for handler in list(root.handlers):
        if getattr(handler, "_regime_trader_managed", False):
            root.removeHandler(handler)

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(root.level)
    console._regime_trader_managed = True  # type: ignore[attr-defined]
    root.addHandler(console)

    for name in ("main", "trades", "alerts", "regime"):
        path = log_dir / f"{name}.log"
        handler = logging.handlers.RotatingFileHandler(
            filename=str(path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        handler._regime_trader_managed = True  # type: ignore[attr-defined]
        logger = logging.getLogger(f"regime_trader.{name}")
        logger.setLevel(root.level)
        logger.addHandler(handler)
        logger.propagate = True


def emit_event(channel: str, event: str, **extra: Any) -> None:
    """Emit a structured event to the given channel logger."""
    logger = logging.getLogger(f"regime_trader.{channel}")
    logger.info(event, extra=extra)


def summarize(events: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Helper used by the dashboard to aggregate event counts per channel."""
    counts: dict[str, int] = {}
    for event in events:
        channel = str(event.get("channel", "main"))
        counts[channel] = counts.get(channel, 0) + 1
    return counts
