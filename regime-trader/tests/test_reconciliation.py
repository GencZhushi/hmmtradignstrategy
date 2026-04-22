"""Phase A7 / A9 - position tracker sync + coordinator reconciliation."""
from __future__ import annotations

from pathlib import Path

import pytest

from broker.alpaca_client import SimulatedBroker
from broker.position_tracker import PositionTracker
from core.execution_coordinator import ExecutionCoordinator
from core.idempotency import IdempotencyStore
from core.lock_manager import LockManager
from core.order_state_machine import OrderStateMachine
from core.risk_manager import RiskLimits, RiskManager
from core.types import Direction, OrderStatus, PortfolioState, TradeIntent


def test_position_tracker_applies_partial_fill_and_updates_metrics() -> None:
    tracker = PositionTracker(broker=None, initial_equity=100_000.0)
    tracker.apply_fill(symbol="SPY", side="BUY", qty=10, price=100.0, stop_price=95.0, regime_name="BULL")
    state = tracker.snapshot()
    assert state.positions["SPY"].quantity == pytest.approx(10)
    tracker.apply_fill(symbol="SPY", side="BUY", qty=5, price=102.0)
    state = tracker.snapshot()
    pos = state.positions["SPY"]
    assert pos.quantity == pytest.approx(15)
    # Weighted average entry price
    assert pos.avg_entry_price == pytest.approx((100.0 * 10 + 102.0 * 5) / 15)


def test_position_tracker_removes_closed_position() -> None:
    tracker = PositionTracker(broker=None, initial_equity=100_000.0)
    tracker.apply_fill(symbol="SPY", side="BUY", qty=10, price=100.0, stop_price=95.0)
    tracker.apply_fill(symbol="SPY", side="SELL", qty=10, price=105.0)
    assert "SPY" not in tracker.snapshot().positions


def test_position_tracker_sync_picks_up_broker_positions() -> None:
    broker = SimulatedBroker()
    broker.positions["SPY"] = {"symbol": "SPY", "qty": 7.0, "avg_entry_price": 100.0, "current_price": 105.0}
    tracker = PositionTracker(broker=broker, initial_equity=100_000.0)
    tracker.sync_positions()
    state = tracker.snapshot()
    assert state.positions["SPY"].quantity == pytest.approx(7.0)
    assert state.positions["SPY"].current_price == pytest.approx(105.0)


def test_tracker_load_state_is_idempotent(tmp_path: Path) -> None:
    tracker = PositionTracker(broker=None, initial_equity=100_000.0)
    tracker.apply_fill(symbol="SPY", side="BUY", qty=3, price=100.0, stop_price=95.0)
    payload = tracker.dump_state()
    other = PositionTracker(broker=None, initial_equity=50_000.0)
    other.load_state(payload)
    assert other.snapshot().positions["SPY"].quantity == pytest.approx(3)


class _StubExecutor:
    """Minimal executor that simulates broker acceptance for reconciliation tests."""

    def __init__(self) -> None:
        self._counter = 0

    def submit_order(self, order) -> str:  # type: ignore[no-untyped-def]
        self._counter += 1
        return f"broker-{self._counter}"

    def modify_stop(self, order_id: str, new_stop: float) -> None:  # pragma: no cover
        return None

    def cancel_order(self, order_id: str) -> None:  # pragma: no cover
        return None


def test_coordinator_reconcile_after_reconnect_syncs_state() -> None:
    state_machine = OrderStateMachine()
    coordinator = ExecutionCoordinator(
        risk_manager=RiskManager(limits=RiskLimits()),
        state_machine=state_machine,
        idempotency=IdempotencyStore(),
        lock_manager=LockManager(default_timeout=0.5),
        executor=_StubExecutor(),
        portfolio_provider=lambda: PortfolioState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0),
        market_price_provider=lambda symbol: 100.0,
    )
    intent = TradeIntent(symbol="SPY", direction=Direction.LONG, allocation_pct=0.1, source="test")
    outcome = coordinator.submit_intent(intent)
    assert outcome.order is not None
    order = outcome.order
    # Simulate the broker's view reflecting a full fill we missed.
    broker_orders = [
        {
            "broker_order_id": order.attempts[-1].broker_order_id,
            "status": "filled",
            "filled_qty": order.quantity,
            "avg_fill_price": 100.0,
        }
    ]
    updated = coordinator.reconcile_after_reconnect(broker_orders=broker_orders)
    refreshed = state_machine.orders[order.order_id]
    assert refreshed.status == OrderStatus.FILLED
    assert refreshed.filled_qty == pytest.approx(order.quantity)
    assert order.order_id in [u.order_id for u in updated]


def test_coordinator_marks_missing_broker_order_dead() -> None:
    state_machine = OrderStateMachine()
    coordinator = ExecutionCoordinator(
        risk_manager=RiskManager(limits=RiskLimits()),
        state_machine=state_machine,
        idempotency=IdempotencyStore(),
        lock_manager=LockManager(default_timeout=0.5),
        executor=_StubExecutor(),
        portfolio_provider=lambda: PortfolioState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0),
        market_price_provider=lambda symbol: 100.0,
    )
    intent = TradeIntent(symbol="QQQ", direction=Direction.LONG, allocation_pct=0.05, source="test")
    outcome = coordinator.submit_intent(intent)
    assert outcome.order is not None
    coordinator.reconcile_after_reconnect(broker_orders=[])
    assert state_machine.orders[outcome.order.order_id].status == OrderStatus.DEAD
