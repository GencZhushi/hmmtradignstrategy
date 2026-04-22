"""Phase A10 - partial fill handling in the order state machine.

Verifies that the ``OrderStateMachine.handle_partial_fill`` path:

- updates ``filled_qty`` and ``avg_fill_price`` atomically
- transitions ACCEPTED -> PARTIALLY_FILLED on the first partial
- stays in PARTIALLY_FILLED on subsequent partials (idempotent transition)
- finalizes to FILLED once cumulative quantity reaches the order size
- refuses to over-fill past the order quantity
- lets ``update_trailing_stop_after_partial_exit`` only ratchet the stop upward

Plus the position-tracker integration side: a partial fill flows into a
``PositionTracker`` with the right quantity and average entry price.
"""
from __future__ import annotations

import pytest

from broker.position_tracker import PositionTracker
from core.order_state_machine import OrderStateMachine
from core.types import OrderStatus


def _new_accepted_order(sm: OrderStateMachine, *, quantity: float = 100.0):
    order = sm.create_order(
        intent_id="intent-1",
        symbol="SPY",
        side="BUY",
        quantity=quantity,
        limit_price=None,
        stop_price=95.0,
        take_profit=None,
        idempotency_key="idem-1",
    )
    sm.register_attempt(order.order_id, attempt_id="att-1", broker_order_id="broker-1")
    sm.advance_order_state(order.order_id, to=OrderStatus.ACCEPTED)
    return order


def test_first_partial_fill_transitions_to_partially_filled() -> None:
    sm = OrderStateMachine()
    order = _new_accepted_order(sm)
    updated = sm.handle_partial_fill(order.order_id, filled_qty=30.0, fill_price=100.0)
    assert updated.status == OrderStatus.PARTIALLY_FILLED
    assert updated.filled_qty == pytest.approx(30.0)
    assert updated.avg_fill_price == pytest.approx(100.0)
    assert updated.remaining_qty() == pytest.approx(70.0)


def test_second_partial_fill_updates_vwap_and_stays_partial() -> None:
    sm = OrderStateMachine()
    order = _new_accepted_order(sm)
    sm.handle_partial_fill(order.order_id, filled_qty=30.0, fill_price=100.0)
    updated = sm.handle_partial_fill(order.order_id, filled_qty=20.0, fill_price=102.0)
    assert updated.status == OrderStatus.PARTIALLY_FILLED
    assert updated.filled_qty == pytest.approx(50.0)
    # Volume-weighted average price of two partials.
    assert updated.avg_fill_price == pytest.approx((30 * 100 + 20 * 102) / 50)


def test_final_partial_fill_completes_the_order() -> None:
    sm = OrderStateMachine()
    order = _new_accepted_order(sm, quantity=100.0)
    sm.handle_partial_fill(order.order_id, filled_qty=60.0, fill_price=100.0)
    final = sm.handle_partial_fill(order.order_id, filled_qty=40.0, fill_price=101.0)
    assert final.status == OrderStatus.FILLED
    assert final.filled_qty == pytest.approx(100.0)
    assert final.remaining_qty() == pytest.approx(0.0)


def test_overfill_is_rejected() -> None:
    sm = OrderStateMachine()
    order = _new_accepted_order(sm, quantity=50.0)
    sm.handle_partial_fill(order.order_id, filled_qty=40.0, fill_price=100.0)
    with pytest.raises(ValueError, match="partial fill exceeds"):
        sm.handle_partial_fill(order.order_id, filled_qty=20.0, fill_price=101.0)


def test_non_positive_fill_rejected() -> None:
    sm = OrderStateMachine()
    order = _new_accepted_order(sm)
    with pytest.raises(ValueError, match="filled_qty must be positive"):
        sm.handle_partial_fill(order.order_id, filled_qty=0.0, fill_price=100.0)


def test_trailing_stop_only_ratchets_upward() -> None:
    sm = OrderStateMachine()
    order = _new_accepted_order(sm)
    sm.update_trailing_stop_after_partial_exit(order.order_id, new_stop=98.0)
    assert sm.orders[order.order_id].stop_price == pytest.approx(98.0)
    # A lower proposed stop must not loosen protection.
    sm.update_trailing_stop_after_partial_exit(order.order_id, new_stop=90.0)
    assert sm.orders[order.order_id].stop_price == pytest.approx(98.0)


def test_position_tracker_reflects_partial_fill_quantities() -> None:
    tracker = PositionTracker(broker=None, initial_equity=100_000.0)
    # Two partial fills at different prices should weight-average the entry.
    tracker.apply_fill(symbol="SPY", side="BUY", qty=30.0, price=100.0, stop_price=95.0)
    tracker.apply_fill(symbol="SPY", side="BUY", qty=20.0, price=102.0)
    position = tracker.snapshot().positions["SPY"]
    assert position.quantity == pytest.approx(50.0)
    assert position.avg_entry_price == pytest.approx((30 * 100 + 20 * 102) / 50)
