"""Phase A9 - concurrency, locking, and single-writer guarantees.

Complements ``test_idempotency.py`` by focusing on *multi-symbol* ordering
and stress patterns:

- conflicting writes against the same symbol serialize (single-writer)
- writes against *different* symbols proceed in parallel (no false sharing)
- ``guard`` context manager propagates exceptions and still releases
- ``held_too_long`` surfaces stale lock owners for monitoring
- ``status`` is a JSON-friendly snapshot for the admin/API surface
- high-fanout submissions produce exactly one non-duplicate outcome per
  idempotency key
"""
from __future__ import annotations

import threading
import time
from datetime import timedelta
from pathlib import Path

import pytest

from core.execution_coordinator import ExecutionCoordinator
from core.idempotency import IdempotencyStore, build_idempotency_key
from core.lock_manager import LockManager, LockUnavailable
from core.order_state_machine import OrderStateMachine
from core.risk_manager import RiskLimits, RiskManager
from core.types import Direction, PortfolioState, TradeIntent


def _coordinator(tmp_path: Path) -> ExecutionCoordinator:
    return ExecutionCoordinator(
        risk_manager=RiskManager(limits=RiskLimits()),
        state_machine=OrderStateMachine(),
        idempotency=IdempotencyStore(snapshot_path=tmp_path / "idem.json"),
        lock_manager=LockManager(default_timeout=1.0),
        portfolio_provider=lambda: PortfolioState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0),
        market_price_provider=lambda symbol: 100.0,
    )


# ---------------------------------------------------------------- LockManager


def test_same_key_serializes_across_threads() -> None:
    manager = LockManager(default_timeout=1.0)
    order: list[str] = []
    barrier = threading.Barrier(2)

    def _hold(name: str, hold_ms: float) -> None:
        barrier.wait()
        acquired = manager.acquire_order_lock("SPY", owner=name, timeout=2.0)
        assert acquired, f"{name} could not acquire lock"
        order.append(f"{name}:start")
        time.sleep(hold_ms / 1000.0)
        order.append(f"{name}:end")
        manager.release_order_lock("SPY")

    t1 = threading.Thread(target=_hold, args=("A", 80))
    t2 = threading.Thread(target=_hold, args=("B", 20))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # Whichever thread wins the race, the two critical sections must not interleave.
    assert order.index("A:end") == order.index("A:start") + 1 or order.index("B:end") == order.index("B:start") + 1


def test_different_keys_do_not_block_each_other() -> None:
    manager = LockManager(default_timeout=1.0)
    events: dict[str, float] = {}

    def _hold(symbol: str) -> None:
        assert manager.acquire_order_lock(symbol, owner=symbol, timeout=1.0)
        events[f"{symbol}:start"] = time.monotonic()
        time.sleep(0.05)
        events[f"{symbol}:end"] = time.monotonic()
        manager.release_order_lock(symbol)

    t1 = threading.Thread(target=_hold, args=("SPY",))
    t2 = threading.Thread(target=_hold, args=("QQQ",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # Both critical sections must have overlapped - neither can wait for the other.
    overlap = min(events["SPY:end"], events["QQQ:end"]) - max(events["SPY:start"], events["QQQ:start"])
    assert overlap > 0, "different-key locks serialized unexpectedly"


def test_guard_releases_lock_on_exception() -> None:
    manager = LockManager(default_timeout=0.5)
    with pytest.raises(ValueError):
        with manager.guard("SPY", owner="caller", timeout=0.5):
            raise ValueError("boom")
    # After the exception the lock must be free again.
    assert manager.acquire_order_lock("SPY", owner="next", timeout=0.2)
    manager.release_order_lock("SPY")


def test_guard_raises_lock_unavailable_on_timeout() -> None:
    manager = LockManager(default_timeout=0.2)
    manager.acquire_order_lock("SPY", owner="A")
    with pytest.raises(LockUnavailable):
        with manager.guard("SPY", owner="B", timeout=0.05):
            pass  # pragma: no cover - guard must fail before entering body
    manager.release_order_lock("SPY")


def test_status_reports_active_owner() -> None:
    manager = LockManager(default_timeout=0.5)
    manager.acquire_order_lock("SPY", owner="owner-1")
    snapshot = manager.status()
    assert snapshot["SPY"]["owner"] == "owner-1"
    assert snapshot["SPY"]["acquired_at"] is not None
    manager.release_order_lock("SPY")
    assert manager.status()["SPY"]["owner"] is None


def test_held_too_long_surfaces_stale_locks() -> None:
    manager = LockManager(default_timeout=0.5)
    manager.acquire_order_lock("SPY", owner="stale")
    time.sleep(0.05)
    stale = manager.held_too_long(threshold=timedelta(milliseconds=10))
    assert "SPY" in stale
    manager.release_order_lock("SPY")


# ---------------------------------------------------------------- Coordinator


def test_conflicting_submissions_same_key_produce_single_execution(tmp_path: Path) -> None:
    coord = _coordinator(tmp_path)
    key = build_idempotency_key("openclaw", "SPY", "LONG", 0.1)
    intents = [
        TradeIntent(
            symbol="SPY",
            direction=Direction.LONG,
            allocation_pct=0.1,
            source="openclaw",
            idempotency_key=key,
        )
        for _ in range(10)
    ]
    results: list = []

    def _submit(intent: TradeIntent) -> None:
        results.append(coord.submit_intent(intent))

    threads = [threading.Thread(target=_submit, args=(i,)) for i in intents]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    non_duplicate = [r for r in results if not r.duplicate]
    assert len(non_duplicate) == 1
    # All other responses must flag duplicate.
    assert all(r.duplicate for r in results if r is not non_duplicate[0])


def test_submissions_for_distinct_symbols_are_independent(tmp_path: Path) -> None:
    coord = _coordinator(tmp_path)
    intents = [
        TradeIntent(symbol="SPY", direction=Direction.LONG, allocation_pct=0.05, source="openclaw"),
        TradeIntent(symbol="QQQ", direction=Direction.LONG, allocation_pct=0.05, source="openclaw"),
        TradeIntent(symbol="IWM", direction=Direction.LONG, allocation_pct=0.05, source="openclaw"),
    ]
    results = [coord.submit_intent(i) for i in intents]
    # Each symbol submission is non-duplicate and successful.
    assert all(r.order is not None for r in results)
    assert len({r.order.order_id for r in results}) == 3
