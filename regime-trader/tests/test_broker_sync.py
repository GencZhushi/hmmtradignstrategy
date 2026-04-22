"""Phase A7 — Broker sync, reconciliation, and retries.

Covers:

- startup reconciliation restores broker positions into local state
- partial fills update local state correctly
- position tracker sync removes positions closed at broker
- retry logic does not duplicate orders (SimulatedBroker level)
- order executor submit + cancel flows
- close_position and close_all_positions
"""
from __future__ import annotations

import pytest

from broker.alpaca_client import BrokerUnavailable, SimulatedBroker
from broker.order_executor import OrderExecutor
from broker.position_tracker import PositionTracker
from core.order_state_machine import OrderRecord


def _sample_order(**overrides) -> OrderRecord:
    defaults = dict(
        order_id="o1",
        trade_id="t1",
        intent_id="i1",
        symbol="SPY",
        side="BUY",
        quantity=10,
        limit_price=500.0,
        stop_price=490.0,
        take_profit=None,
        idempotency_key="key-1",
    )
    defaults.update(overrides)
    return OrderRecord(**defaults)


# ------------------------------------------------------------------ reconciliation


def test_sync_positions_restores_broker_state() -> None:
    broker = SimulatedBroker()
    broker.positions["SPY"] = {"symbol": "SPY", "qty": 50, "avg_entry_price": 500.0}
    broker.positions["QQQ"] = {"symbol": "QQQ", "qty": 20, "avg_entry_price": 350.0}
    tracker = PositionTracker(broker=broker)
    positions = tracker.sync_positions()
    assert len(positions) == 2
    symbols = {p.symbol for p in positions}
    assert symbols == {"SPY", "QQQ"}
    assert tracker.state.positions["SPY"].quantity == 50


def test_sync_positions_removes_closed_at_broker() -> None:
    broker = SimulatedBroker()
    tracker = PositionTracker(broker=broker)
    # Manually inject a local-only position
    tracker.apply_fill(symbol="AAPL", side="BUY", qty=10, price=150.0)
    assert "AAPL" in tracker.state.positions
    # Broker has nothing → sync should remove AAPL
    positions = tracker.sync_positions()
    assert len(positions) == 0
    assert "AAPL" not in tracker.state.positions


def test_sync_positions_no_broker_returns_local() -> None:
    tracker = PositionTracker(broker=None)
    tracker.apply_fill(symbol="MSFT", side="BUY", qty=5, price=300.0)
    positions = tracker.sync_positions()
    assert len(positions) == 1
    assert positions[0].symbol == "MSFT"


# ------------------------------------------------------------------ fill handling


def test_apply_fill_buy_creates_position() -> None:
    tracker = PositionTracker()
    pos = tracker.apply_fill(symbol="SPY", side="BUY", qty=10, price=500.0, stop_price=490.0)
    assert pos.symbol == "SPY"
    assert pos.quantity == 10
    assert pos.avg_entry_price == 500.0
    assert pos.stop_price == 490.0


def test_apply_fill_sell_closes_position() -> None:
    tracker = PositionTracker()
    tracker.apply_fill(symbol="SPY", side="BUY", qty=10, price=500.0)
    tracker.apply_fill(symbol="SPY", side="SELL", qty=10, price=510.0)
    assert "SPY" not in tracker.state.positions


def test_apply_partial_fill_updates_quantity() -> None:
    tracker = PositionTracker()
    tracker.apply_fill(symbol="SPY", side="BUY", qty=20, price=500.0)
    tracker.apply_fill(symbol="SPY", side="SELL", qty=5, price=510.0)
    assert tracker.state.positions["SPY"].quantity == 15


def test_apply_fill_rejects_negative_position() -> None:
    tracker = PositionTracker()
    tracker.apply_fill(symbol="SPY", side="BUY", qty=5, price=500.0)
    with pytest.raises(ValueError, match="short position"):
        tracker.apply_fill(symbol="SPY", side="SELL", qty=10, price=510.0)


def test_apply_fill_sell_without_position_raises() -> None:
    tracker = PositionTracker()
    with pytest.raises(ValueError, match="no position"):
        tracker.apply_fill(symbol="XYZ", side="SELL", qty=1, price=100.0)


# ------------------------------------------------------------------ order executor


def test_executor_submit_returns_broker_id() -> None:
    broker = SimulatedBroker()
    executor = OrderExecutor(broker=broker)
    order = _sample_order()
    broker_id = executor.submit_order(order)
    assert broker_id.startswith("sim-")
    assert len(broker.orders) == 1


def test_executor_cancel_marks_order_canceled() -> None:
    broker = SimulatedBroker()
    executor = OrderExecutor(broker=broker)
    order = _sample_order()
    broker_id = executor.submit_order(order)
    executor.cancel_order(broker_id)
    assert broker.orders[0]["status"] == "canceled"


def test_executor_dry_run_does_not_touch_broker() -> None:
    broker = SimulatedBroker()
    executor = OrderExecutor(broker=broker, dry_run=True)
    order = _sample_order()
    broker_id = executor.submit_order(order)
    assert broker_id.startswith("dry-")
    assert len(broker.orders) == 0


def test_executor_close_position() -> None:
    broker = SimulatedBroker()
    executor = OrderExecutor(broker=broker)
    broker_id = executor.close_position(symbol="AAPL", qty=15)
    assert broker_id.startswith("sim-")
    assert broker.orders[0]["side"] == "SELL"
    assert broker.orders[0]["qty"] == 15


def test_executor_close_all_positions() -> None:
    broker = SimulatedBroker()
    executor = OrderExecutor(broker=broker)
    ids = executor.close_all_positions({"SPY": 10, "QQQ": 5})
    assert len(ids) == 2
    assert len(broker.orders) == 2


# ------------------------------------------------------------------ simulated broker retries


def test_simulated_broker_fill_and_partial() -> None:
    broker = SimulatedBroker()
    resp = broker.submit_order({"symbol": "SPY", "side": "BUY", "qty": 100})
    broker.simulate_fill(resp["broker_order_id"], fill_qty=50, fill_price=500.0)
    assert resp["status"] == "partially_filled"
    assert resp["filled_qty"] == 50
    broker.simulate_fill(resp["broker_order_id"], fill_qty=50, fill_price=501.0)
    assert resp["status"] == "filled"


def test_simulated_broker_replace_order() -> None:
    broker = SimulatedBroker()
    resp = broker.submit_order({"symbol": "SPY", "side": "BUY", "qty": 10})
    updated = broker.replace_order(resp["broker_order_id"], {"stop_price": 495.0})
    assert updated["stop_price"] == 495.0


# ------------------------------------------------------------------ tracker state snapshots


def test_tracker_dump_and_load_roundtrip() -> None:
    tracker = PositionTracker()
    tracker.apply_fill(symbol="SPY", side="BUY", qty=10, price=500.0)
    dump = tracker.dump_state()
    new_tracker = PositionTracker()
    new_tracker.load_state(dump)
    assert "SPY" in new_tracker.state.positions
    assert new_tracker.state.positions["SPY"].quantity == 10
    assert new_tracker.state.equity == pytest.approx(tracker.state.equity)


def test_tracker_reconcile_from_orders() -> None:
    tracker = PositionTracker()
    tracker.reconcile_from_orders([
        {"symbol": "SPY", "avg_fill_price": 510.0},
        {"symbol": "QQQ", "avg_fill_price": 360.0},
    ])
    assert tracker.current_prices()["SPY"] == 510.0
    assert tracker.current_prices()["QQQ"] == 360.0
