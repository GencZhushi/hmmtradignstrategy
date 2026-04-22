"""Phase A10 - protective stop-order failure handling.

When a protective stop cannot be placed (or is rejected by the broker after a
parent fills), the state machine must:

- mark the order's ``protective_stop_status`` as ``failed``
- capture a structured rejection reason for operators and the audit log
- leave the parent order status alone so the coordinator can decide
  whether to flatten or retry the protective leg
"""
from __future__ import annotations

from core.order_state_machine import OrderStateMachine
from core.types import OrderStatus


def _build_accepted_order(sm: OrderStateMachine):
    order = sm.create_order(
        intent_id="intent-stop",
        symbol="SPY",
        side="BUY",
        quantity=10.0,
        limit_price=None,
        stop_price=95.0,
        take_profit=None,
        idempotency_key="idem-stop",
    )
    sm.register_attempt(order.order_id, attempt_id="att-1", broker_order_id="b-1")
    sm.advance_order_state(order.order_id, to=OrderStatus.ACCEPTED)
    return order


def test_handle_stop_failure_sets_protective_status_and_reason() -> None:
    sm = OrderStateMachine()
    order = _build_accepted_order(sm)
    updated = sm.handle_stop_failure(order.order_id, reason="broker_rejected:stop_below_market")
    assert updated.protective_stop_status == "failed"
    assert updated.rejection_reason == "broker_rejected:stop_below_market"


def test_handle_stop_failure_does_not_change_parent_status() -> None:
    sm = OrderStateMachine()
    order = _build_accepted_order(sm)
    sm.handle_stop_failure(order.order_id, reason="stop_rejected")
    assert sm.orders[order.order_id].status == OrderStatus.ACCEPTED


def test_stop_failure_is_visible_in_summary_payload() -> None:
    sm = OrderStateMachine()
    order = _build_accepted_order(sm)
    sm.handle_stop_failure(order.order_id, reason="child_stop_rejected")
    payload = sm.summary(order.order_id)
    assert payload["protective_stop_status"] == "failed"
    assert payload["rejection_reason"] == "child_stop_rejected"


def test_stop_failure_allows_subsequent_mark_dead() -> None:
    sm = OrderStateMachine()
    order = _build_accepted_order(sm)
    sm.handle_stop_failure(order.order_id, reason="stop_missing")
    # Operator decides to flatten the unprotected position - mark_order_dead
    # must still succeed from ACCEPTED.
    dead = sm.mark_order_dead(order.order_id, reason="flatten_due_to_stop_failure")
    assert dead.status == OrderStatus.DEAD
    assert dead.protective_stop_status == "failed"
