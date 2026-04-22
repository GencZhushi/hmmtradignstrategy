"""Phase A10 - explicit order lifecycle transitions and retry semantics.

The state machine must:

- reject illegal transitions (FILLED -> anything, REJECTED -> anything, etc.)
- preserve ``trade_id`` and ``intent_id`` across retry attempts so the audit
  log can reconstruct the full retry chain
- mark dead orders cleanly from any live state
- expose a ``summary()`` payload containing every attempt for API consumers
"""
from __future__ import annotations

import pytest

from core.order_state_machine import InvalidTransition, OrderStateMachine
from core.types import OrderStatus


def _create(sm: OrderStateMachine, *, trade_id: str | None = None, intent_id: str = "intent-1"):
    return sm.create_order(
        intent_id=intent_id,
        symbol="SPY",
        side="BUY",
        quantity=10.0,
        limit_price=None,
        stop_price=95.0,
        take_profit=None,
        idempotency_key=f"idem-{intent_id}",
        trade_id=trade_id,
    )


def test_create_order_populates_new_status_and_unique_ids() -> None:
    sm = OrderStateMachine()
    order = _create(sm)
    assert order.status == OrderStatus.NEW
    assert order.order_id
    assert order.trade_id
    assert order.intent_id == "intent-1"


def test_legal_transition_chain_accept_fill() -> None:
    sm = OrderStateMachine()
    order = _create(sm)
    sm.advance_order_state(order.order_id, to=OrderStatus.SUBMITTED)
    sm.advance_order_state(order.order_id, to=OrderStatus.ACCEPTED)
    sm.advance_order_state(order.order_id, to=OrderStatus.FILLED)
    assert sm.orders[order.order_id].status == OrderStatus.FILLED


def test_illegal_transition_from_new_to_filled_is_rejected() -> None:
    sm = OrderStateMachine()
    order = _create(sm)
    with pytest.raises(InvalidTransition):
        sm.advance_order_state(order.order_id, to=OrderStatus.FILLED)


def test_illegal_transition_from_filled_is_rejected() -> None:
    sm = OrderStateMachine()
    order = _create(sm)
    sm.advance_order_state(order.order_id, to=OrderStatus.SUBMITTED)
    sm.advance_order_state(order.order_id, to=OrderStatus.ACCEPTED)
    sm.advance_order_state(order.order_id, to=OrderStatus.FILLED)
    with pytest.raises(InvalidTransition):
        sm.advance_order_state(order.order_id, to=OrderStatus.CANCELLED)


def test_retries_reuse_trade_id_across_attempts() -> None:
    sm = OrderStateMachine()
    first = _create(sm, intent_id="intent-1")
    sm.register_attempt(first.order_id, attempt_id="att-1", broker_order_id="broker-1")
    # Second attempt against the same intent should reuse the trade_id so the
    # audit log can reconstruct the retry chain.
    second = sm.create_order(
        intent_id="intent-1",
        symbol="SPY",
        side="BUY",
        quantity=10.0,
        limit_price=None,
        stop_price=95.0,
        take_profit=None,
        idempotency_key="idem-intent-1",
        trade_id=first.trade_id,
    )
    sm.register_attempt(second.order_id, attempt_id="att-2", broker_order_id="broker-2")

    attempts = sm.attempts_for_trade(first.trade_id)
    assert {a.attempt_id for a in attempts} == {"att-1", "att-2"}
    assert first.intent_id == second.intent_id


def test_register_attempt_advances_new_to_submitted() -> None:
    sm = OrderStateMachine()
    order = _create(sm)
    attempt = sm.register_attempt(order.order_id, attempt_id="att-1", broker_order_id="b-1")
    assert attempt.status == OrderStatus.SUBMITTED
    assert sm.orders[order.order_id].status == OrderStatus.SUBMITTED


def test_mark_order_dead_from_accepted() -> None:
    sm = OrderStateMachine()
    order = _create(sm)
    sm.advance_order_state(order.order_id, to=OrderStatus.SUBMITTED)
    sm.advance_order_state(order.order_id, to=OrderStatus.ACCEPTED)
    killed = sm.mark_order_dead(order.order_id, reason="broker_unresponsive")
    assert killed.status == OrderStatus.DEAD
    assert killed.rejection_reason == "broker_unresponsive"


def test_mark_order_dead_is_idempotent_for_terminal_states() -> None:
    sm = OrderStateMachine()
    order = _create(sm)
    sm.advance_order_state(order.order_id, to=OrderStatus.SUBMITTED)
    sm.advance_order_state(order.order_id, to=OrderStatus.ACCEPTED)
    sm.advance_order_state(order.order_id, to=OrderStatus.FILLED)
    # A filled order cannot legally transition to DEAD but the helper must
    # complete without crashing - it records the rejection reason and leaves
    # the terminal status alone.
    result = sm.mark_order_dead(order.order_id, reason="post_fill_cleanup")
    assert result.status == OrderStatus.FILLED
    assert result.rejection_reason == "post_fill_cleanup"


def test_summary_payload_exposes_every_attempt() -> None:
    sm = OrderStateMachine()
    order = _create(sm)
    sm.register_attempt(order.order_id, attempt_id="att-1", broker_order_id="b-1")
    sm.register_attempt(order.order_id, attempt_id="att-2", broker_order_id="b-2")
    payload = sm.summary(order.order_id)
    assert payload["order_id"] == order.order_id
    assert payload["trade_id"] == order.trade_id
    attempts = payload["attempts"]
    assert [a["attempt_id"] for a in attempts] == ["att-1", "att-2"]
    assert [a["broker_order_id"] for a in attempts] == ["b-1", "b-2"]
