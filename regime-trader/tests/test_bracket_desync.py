"""Phase A10 - bracket-order child desync reconciliation.

A bracket order (parent + stop + take-profit child) can lose its protective
child legs if the broker drops or never accepts them. The state machine must:

- record which bracket children are missing on the order record
- flag ``protective_stop_status`` as ``missing`` so the coordinator can
  repair the bracket or close the exposure
- preserve the parent order's current lifecycle status so downstream logic
  still sees it as live
- let ``mark_order_dead`` drive the terminal transition if recovery fails
"""
from __future__ import annotations

from core.order_state_machine import OrderStateMachine
from core.types import OrderStatus


def _bracketed_order(sm: OrderStateMachine):
    order = sm.create_order(
        intent_id="intent-bracket",
        symbol="SPY",
        side="BUY",
        quantity=10.0,
        limit_price=None,
        stop_price=95.0,
        take_profit=110.0,
        idempotency_key="idem-bracket",
    )
    order.bracket_child_ids = ["stop-1", "tp-1"]
    sm.register_attempt(order.order_id, attempt_id="att-1", broker_order_id="b-parent")
    sm.advance_order_state(order.order_id, to=OrderStatus.ACCEPTED)
    return order


def test_missing_stop_child_is_recorded_with_reason() -> None:
    sm = OrderStateMachine()
    order = _bracketed_order(sm)
    updated = sm.handle_bracket_desync(order.order_id, missing_child_ids=["stop-1"])
    assert updated.protective_stop_status == "missing"
    assert "bracket_desync" in (updated.rejection_reason or "")
    assert "stop-1" in (updated.rejection_reason or "")


def test_multiple_missing_children_are_listed_in_reason() -> None:
    sm = OrderStateMachine()
    order = _bracketed_order(sm)
    updated = sm.handle_bracket_desync(
        order.order_id,
        missing_child_ids=["stop-1", "tp-1"],
    )
    reason = updated.rejection_reason or ""
    assert "stop-1" in reason and "tp-1" in reason


def test_bracket_desync_leaves_parent_status_alone() -> None:
    sm = OrderStateMachine()
    order = _bracketed_order(sm)
    sm.handle_bracket_desync(order.order_id, missing_child_ids=["stop-1"])
    # Parent is still live - coordinator decides whether to repair or flatten.
    assert sm.orders[order.order_id].status == OrderStatus.ACCEPTED


def test_bracket_desync_payload_surfaces_in_summary() -> None:
    sm = OrderStateMachine()
    order = _bracketed_order(sm)
    sm.handle_bracket_desync(order.order_id, missing_child_ids=["tp-1"])
    payload = sm.summary(order.order_id)
    assert payload["protective_stop_status"] == "missing"
    assert "bracket_desync" in (payload["rejection_reason"] or "")


def test_desync_recovery_can_be_escalated_to_dead() -> None:
    sm = OrderStateMachine()
    order = _bracketed_order(sm)
    sm.handle_bracket_desync(order.order_id, missing_child_ids=["stop-1"])
    # Repair failed - coordinator escalates by killing the order outright.
    killed = sm.mark_order_dead(order.order_id, reason="bracket_repair_failed")
    assert killed.status == OrderStatus.DEAD
    assert killed.rejection_reason == "bracket_repair_failed"
    assert killed.protective_stop_status == "missing"
