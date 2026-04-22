"""Alert dispatcher for the engine (Phase A8).

- Rate-limited: one alert per event type per ``rate_limit_minutes`` window.
- Pluggable sinks: console + logger by default; callers can register webhooks
  (used by the API/dashboard) without touching engine internals.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping

LOG = logging.getLogger("regime_trader.alerts")


AlertSink = Callable[[str, Mapping[str, Any]], None]


@dataclass
class AlertDispatcher:
    """Rate-limited alert emitter."""

    rate_limit_minutes: int = 15
    sinks: list[AlertSink] = field(default_factory=list)
    _last_emission: dict[str, datetime] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def register_sink(self, sink: AlertSink) -> None:
        self.sinks.append(sink)

    def emit_alert(
        self,
        event_type: str,
        message: str,
        *,
        severity: str = "info",
        context: Mapping[str, Any] | None = None,
    ) -> bool:
        with self._lock:
            now = datetime.now(timezone.utc)
            last = self._last_emission.get(event_type)
            if last and now - last < timedelta(minutes=self.rate_limit_minutes):
                return False
            self._last_emission[event_type] = now
        payload = {
            "event_type": event_type,
            "severity": severity,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            "context": dict(context or {}),
        }
        LOG.log(_severity_level(severity), message, extra=payload)
        for sink in self.sinks:
            try:
                sink(event_type, payload)
            except Exception as exc:  # pragma: no cover - sink isolation
                LOG.error("Alert sink failed for %s: %s", event_type, exc)
        return True

    def clear(self) -> None:
        with self._lock:
            self._last_emission.clear()


def _severity_level(severity: str) -> int:
    return {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }.get(severity.lower(), logging.INFO)


def console_sink(event_type: str, payload: Mapping[str, Any]) -> None:
    """Convenience sink that prints a short banner to stdout."""
    print(f"[ALERT:{payload.get('severity', 'info').upper()}] {event_type} - {payload.get('message', '')}")
