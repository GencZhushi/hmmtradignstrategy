"""Phase A9 - idempotency store, lock manager, execution coordinator."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from core.execution_coordinator import ExecutionCoordinator
from core.idempotency import IdempotencyStore, build_idempotency_key
from core.lock_manager import LockManager, LockUnavailable
from core.order_state_machine import OrderStateMachine
from core.risk_manager import RiskLimits, RiskManager
from core.types import Direction, PortfolioState, TradeIntent


def test_idempotency_registration_is_deduplicated(tmp_path: Path) -> None:
    store = IdempotencyStore(snapshot_path=tmp_path / "idem.json")
    key = build_idempotency_key("openclaw", "SPY", "LONG", 0.1)
    first = store.register_intent(key, intent_id="intent-1", actor="openclaw", resource_type="trade")
    second = store.register_intent(key, intent_id="intent-2", actor="openclaw", resource_type="trade")
    assert first.intent_id == "intent-1"
    assert second.intent_id == "intent-1"
    assert store.check_idempotency(key).intent_id == "intent-1"


def test_idempotency_snapshot_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "idem.json"
    store = IdempotencyStore(snapshot_path=path)
    key = build_idempotency_key("a", "b")
    store.register_intent(key, intent_id="id-1", actor="x", resource_type="r")
    store.mark_status(key, status="previewed", result={"approved": True})
    reloaded = IdempotencyStore(snapshot_path=path)
    record = reloaded.check_idempotency(key)
    assert record is not None
    assert record.status == "previewed"
    assert record.result == {"approved": True}


def test_lock_manager_serializes_owners() -> None:
    manager = LockManager(default_timeout=0.5)
    assert manager.acquire_order_lock("SPY", owner="a")
    assert not manager.acquire_order_lock("SPY", owner="b", timeout=0.1)
    manager.release_order_lock("SPY")
    assert manager.acquire_order_lock("SPY", owner="b", timeout=0.1)
    manager.release_order_lock("SPY")


def test_lock_manager_guard_releases_on_exception() -> None:
    manager = LockManager(default_timeout=0.5)
    with pytest.raises(RuntimeError):
        with manager.guard("QQQ", owner="owner-1", timeout=0.1):
            raise RuntimeError("boom")
    # Lock must be released despite the exception.
    assert manager.acquire_order_lock("QQQ", owner="owner-2", timeout=0.1)
    manager.release_order_lock("QQQ")


def _build_coordinator(tmp_path: Path) -> ExecutionCoordinator:
    store = IdempotencyStore(snapshot_path=tmp_path / "idem.json")
    coord = ExecutionCoordinator(
        risk_manager=RiskManager(limits=RiskLimits()),
        state_machine=OrderStateMachine(),
        idempotency=store,
        lock_manager=LockManager(default_timeout=0.5),
        portfolio_provider=lambda: PortfolioState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0),
        market_price_provider=lambda symbol: 100.0,
    )
    return coord


def test_coordinator_returns_duplicate_on_retry(tmp_path: Path) -> None:
    coord = _build_coordinator(tmp_path)
    intent = TradeIntent(symbol="SPY", direction=Direction.LONG, allocation_pct=0.1, source="openclaw")
    first = coord.submit_intent(intent)
    retry = TradeIntent(
        symbol="SPY",
        direction=Direction.LONG,
        allocation_pct=0.1,
        source="openclaw",
        idempotency_key=first.intent.idempotency_key,
    )
    second = coord.submit_intent(retry)
    assert second.duplicate is True


def test_coordinator_prevents_double_execution_under_threads(tmp_path: Path) -> None:
    coord = _build_coordinator(tmp_path)
    intent = TradeIntent(symbol="AAPL", direction=Direction.LONG, allocation_pct=0.05, source="openclaw")
    results: list = []

    def _submit() -> None:
        copy = TradeIntent(
            symbol=intent.symbol,
            direction=intent.direction,
            allocation_pct=intent.allocation_pct,
            source=intent.source,
            idempotency_key=intent.idempotency_key or build_idempotency_key(intent.source, intent.symbol, intent.direction.value, intent.allocation_pct, intent.intent_type),
        )
        results.append(coord.submit_intent(copy))

    threads = [threading.Thread(target=_submit) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    non_duplicate = [r for r in results if not r.duplicate]
    assert len(non_duplicate) == 1, "Only one of the concurrent retries should execute"
