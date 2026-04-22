"""Phase A7 + A10 - order executor, state machine, partial fills, stop failures."""
from __future__ import annotations

import pytest

from broker.alpaca_client import SimulatedBroker
from broker.order_executor import OrderExecutor
from core.order_state_machine import (
    InvalidTransition,
    OrderStateMachine,
)
from core.types import OrderStatus


def _make_order(state: OrderStateMachine, symbol: str = "SPY", qty: float = 100.0):
    return state.create_order(
        intent_id="intent-1",
        symbol=symbol,
        side="BUY",
        quantity=qty,
        limit_price=100.0,
        stop_price=95.0,
        take_profit=None,
        idempotency_key=f"idem-{symbol}",
    )


def test_order_submission_via_simulated_broker() -> None:
    broker = SimulatedBroker()
    executor = OrderExecutor(broker=broker)
    state = OrderStateMachine()
    order = _make_order(state)
    broker_id = executor.submit_order(order)
    assert broker_id.startswith("sim-")
    assert broker.orders[0]["symbol"] == "SPY"


def test_state_machine_rejects_illegal_transitions() -> None:
    state = OrderStateMachine()
    order = _make_order(state)
    state.advance_order_state(order.order_id, to=OrderStatus.SUBMITTED)
    state.advance_order_state(order.order_id, to=OrderStatus.ACCEPTED)
    state.advance_order_state(order.order_id, to=OrderStatus.FILLED)
    with pytest.raises(InvalidTransition):
        state.advance_order_state(order.order_id, to=OrderStatus.PARTIALLY_FILLED)


def test_partial_fill_updates_quantity_and_status() -> None:
    state = OrderStateMachine()
    order = _make_order(state, qty=100.0)
    state.advance_order_state(order.order_id, to=OrderStatus.SUBMITTED)
    state.advance_order_state(order.order_id, to=OrderStatus.ACCEPTED)

    partial = state.handle_partial_fill(order.order_id, filled_qty=40.0, fill_price=100.25)
    assert partial.status == OrderStatus.PARTIALLY_FILLED
    assert partial.filled_qty == pytest.approx(40.0)
    assert partial.avg_fill_price == pytest.approx(100.25)

    final = state.handle_partial_fill(order.order_id, filled_qty=60.0, fill_price=100.50)
    assert final.status == OrderStatus.FILLED
    assert final.filled_qty == pytest.approx(100.0)
    assert final.avg_fill_price == pytest.approx((40 * 100.25 + 60 * 100.50) / 100)


def test_retry_reuses_trade_id_and_intent_id() -> None:
    state = OrderStateMachine()
    order = _make_order(state)
    state.advance_order_state(order.order_id, to=OrderStatus.FAILED)
    state.advance_order_state(order.order_id, to=OrderStatus.DEAD, reason="broker_reject")
    # A retry must build a fresh order with the same trade_id and intent_id.
    retry = state.create_order(
        intent_id=order.intent_id,
        symbol=order.symbol,
        side=order.side,
        quantity=order.quantity,
        limit_price=order.limit_price,
        stop_price=order.stop_price,
        take_profit=order.take_profit,
        idempotency_key=order.idempotency_key,
        trade_id=order.trade_id,
    )
    assert retry.order_id != order.order_id
    assert retry.trade_id == order.trade_id
    assert retry.intent_id == order.intent_id


def test_stop_failure_marks_protective_status() -> None:
    state = OrderStateMachine()
    order = _make_order(state)
    state.handle_stop_failure(order.order_id, reason="alpaca_rejected_stop")
    assert order.protective_stop_status == "failed"
    assert order.rejection_reason == "alpaca_rejected_stop"


def test_bracket_desync_flagged() -> None:
    state = OrderStateMachine()
    order = _make_order(state)
    updated = state.handle_bracket_desync(order.order_id, missing_child_ids=["stop-abc"])
    assert updated.protective_stop_status == "missing"
    assert "bracket_desync" in (updated.rejection_reason or "")


def test_dead_order_detection() -> None:
    state = OrderStateMachine()
    order = _make_order(state)
    state.mark_order_dead(order.order_id, reason="broker_unreachable")
    assert order.status == OrderStatus.DEAD


def test_trailing_stop_updates_only_upwards() -> None:
    state = OrderStateMachine()
    order = _make_order(state)
    state.update_trailing_stop_after_partial_exit(order.order_id, new_stop=98.0)
    state.update_trailing_stop_after_partial_exit(order.order_id, new_stop=99.5)
    assert order.stop_price == pytest.approx(99.5)
    state.update_trailing_stop_after_partial_exit(order.order_id, new_stop=90.0)
    assert order.stop_price == pytest.approx(99.5)


def test_simulated_broker_can_fill_and_remove_position() -> None:
    broker = SimulatedBroker()
    executor = OrderExecutor(broker=broker)
    state = OrderStateMachine()
    order = _make_order(state, qty=10.0)
    broker_id = executor.submit_order(order)
    broker.simulate_fill(broker_id, fill_qty=10.0, fill_price=100.0)
    assert broker.positions["SPY"]["qty"] == pytest.approx(10.0)
    closing = state.create_order(
        intent_id="close-1",
        symbol="SPY",
        side="SELL",
        quantity=10.0,
        limit_price=None,
        stop_price=None,
        take_profit=None,
        idempotency_key="close",
    )
    close_id = executor.submit_order(closing)
    broker.simulate_fill(close_id, fill_qty=10.0, fill_price=101.0)
    assert "SPY" not in broker.positions
