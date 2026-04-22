"""Idempotency store: guarantees repeated intents do not double-execute.

Backed by an in-memory dictionary with optional JSON snapshot support so the
store survives restart. The API middleware and the OpenClaw adapter both rely
on this to deduplicate retries from flaky clients.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

LOG = logging.getLogger(__name__)


@dataclass
class IdempotencyRecord:
    idempotency_key: str
    intent_id: str
    actor: str
    resource_type: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "pending"
    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None


@dataclass
class IdempotencyStore:
    """Thread-safe keyed store for idempotency records."""

    snapshot_path: Path | None = None
    _records: dict[str, IdempotencyRecord] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.snapshot_path and Path(self.snapshot_path).exists():
            self._load()

    def register_intent(
        self,
        key: str,
        *,
        intent_id: str,
        actor: str,
        resource_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> IdempotencyRecord:
        with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                return existing
            record = IdempotencyRecord(
                idempotency_key=key,
                intent_id=intent_id,
                actor=actor,
                resource_type=resource_type,
                payload=dict(payload or {}),
            )
            self._records[key] = record
            self._save_unlocked()
            return record

    def check_idempotency(self, key: str) -> IdempotencyRecord | None:
        with self._lock:
            return self._records.get(key)

    def mark_status(
        self,
        key: str,
        *,
        status: str,
        result: Mapping[str, Any] | None = None,
    ) -> IdempotencyRecord | None:
        with self._lock:
            record = self._records.get(key)
            if record is None:
                return None
            record.status = status
            if result is not None:
                record.result = dict(result)
            self._save_unlocked()
            return record

    def remove(self, key: str) -> None:
        with self._lock:
            self._records.pop(key, None)
            self._save_unlocked()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {key: asdict(record) for key, record in self._records.items()}

    def restore(self, snapshot: Mapping[str, Mapping[str, Any]]) -> None:
        with self._lock:
            self._records = {
                key: IdempotencyRecord(**payload)
                for key, payload in snapshot.items()
            }
            self._save_unlocked()

    # ---------------------------------------------------------- persistence
    def _load(self) -> None:
        try:
            payload = json.loads(Path(self.snapshot_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
            LOG.warning("Unable to load idempotency snapshot: %s", exc)
            return
        self.restore(payload.get("records", {}))

    def _save_unlocked(self) -> None:
        if self.snapshot_path is None:
            return
        path = Path(self.snapshot_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"records": self.snapshot()}, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)


def build_idempotency_key(*parts: Any) -> str:
    """Build a stable idempotency key from structured parts."""
    import hashlib

    payload = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
