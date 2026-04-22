"""Per-resource locking used by the execution coordinator (Phase A9).

Ensures that the engine has a *single-writer* path for broker mutations and that
two conflicting requests against the same symbol or account serialize cleanly.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterator

LOG = logging.getLogger(__name__)


class LockUnavailable(RuntimeError):
    """Raised when a caller cannot acquire a resource lock in the timeout."""


@dataclass
class _LockEntry:
    lock: threading.RLock
    owner: str | None = None
    acquired_at: datetime | None = None


@dataclass
class LockManager:
    """Small wrapper around keyed RLocks for deterministic serialization."""

    default_timeout: float = 5.0
    _locks: dict[str, _LockEntry] = field(default_factory=dict, init=False)
    _registry_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _entry(self, key: str) -> _LockEntry:
        with self._registry_lock:
            entry = self._locks.get(key)
            if entry is None:
                entry = _LockEntry(lock=threading.RLock())
                self._locks[key] = entry
            return entry

    def acquire_order_lock(
        self,
        key: str,
        *,
        owner: str,
        timeout: float | None = None,
    ) -> bool:
        entry = self._entry(key)
        timeout = self.default_timeout if timeout is None else timeout
        # Busy-wait while a different owner currently holds the lock so callers
        # on the same thread (RLock re-entry) still observe ownership
        # semantics. This guarantees that concurrent submissions from distinct
        # intent_ids serialize cleanly even if they share a thread.
        if entry.owner is not None and entry.owner != owner:
            deadline = time.monotonic() + timeout
            while entry.owner is not None and entry.owner != owner:
                if time.monotonic() >= deadline:
                    LOG.warning(
                        "LockManager ownership wait timed out: key=%s current_owner=%s requested=%s",
                        key,
                        entry.owner,
                        owner,
                    )
                    return False
                time.sleep(0.005)
        acquired = entry.lock.acquire(timeout=timeout)
        if not acquired:
            LOG.warning("LockManager timeout: key=%s owner=%s", key, owner)
            return False
        entry.owner = owner
        entry.acquired_at = datetime.now(timezone.utc)
        return True

    def release_order_lock(self, key: str) -> None:
        entry = self._locks.get(key)
        if entry is None:
            return
        entry.owner = None
        entry.acquired_at = None
        try:
            entry.lock.release()
        except RuntimeError:  # pragma: no cover - defensive
            LOG.debug("LockManager: release called without ownership key=%s", key)

    @contextmanager
    def guard(self, key: str, *, owner: str, timeout: float | None = None) -> Iterator[None]:
        acquired = self.acquire_order_lock(key, owner=owner, timeout=timeout)
        if not acquired:
            raise LockUnavailable(f"Could not acquire lock for {key}")
        try:
            yield
        finally:
            self.release_order_lock(key)

    def status(self) -> dict[str, dict[str, str | None]]:
        return {
            key: {
                "owner": entry.owner,
                "acquired_at": entry.acquired_at.isoformat() if entry.acquired_at else None,
            }
            for key, entry in self._locks.items()
        }

    def held_too_long(self, *, threshold: timedelta) -> list[str]:
        now = datetime.now(timezone.utc)
        return [
            key
            for key, entry in self._locks.items()
            if entry.acquired_at and now - entry.acquired_at > threshold
        ]
