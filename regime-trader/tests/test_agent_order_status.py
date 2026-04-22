"""Phase C6 - agent interpretation of order lifecycle (partial fills, retries)."""
from __future__ import annotations

from integrations.openclaw.interpreters import (
    decide_retry_or_wait,
    escalate_protection_failure,
    interpret_order_state,
)


def test_partial_fill_is_not_terminal() -> None:
    order = {
        "status": "partially_filled",
        "filled_qty": 5.0,
        "quantity": 10.0,
        "protective_stop_status": "active",
        "attempts": [{"attempt_id": "a1"}],
    }
    interp = interpret_order_state(order)
    assert interp.terminal is False
    assert interp.partially_filled is True
    assert interp.should_wait is True


def test_filled_order_is_terminal_and_done() -> None:
    order = {
        "status": "filled",
        "filled_qty": 10.0,
        "quantity": 10.0,
        "protective_stop_status": "active",
        "attempts": [{"attempt_id": "a1"}],
    }
    interp = interpret_order_state(order)
    assert interp.terminal is True
    assert interp.should_retry is False
    assert decide_retry_or_wait(order) == "done"


def test_rejected_order_triggers_escalation() -> None:
    order = {
        "status": "rejected",
        "filled_qty": 0.0,
        "quantity": 10.0,
        "protective_stop_status": "failed",
        "attempts": [{"attempt_id": "a1"}],
    }
    interp = interpret_order_state(order)
    assert interp.terminal is True
    assert interp.escalate is True
    assert decide_retry_or_wait(order) == "escalate"


def test_failed_order_retries_up_to_limit() -> None:
    order = {
        "status": "failed",
        "filled_qty": 0.0,
        "quantity": 10.0,
        "protective_stop_status": "active",
        "attempts": [{"attempt_id": "a1"}],
    }
    assert decide_retry_or_wait(order, max_attempts=3) == "retry"
    order_three = {**order, "attempts": [{"attempt_id": "a1"}, {"attempt_id": "a2"}, {"attempt_id": "a3"}]}
    # Three attempts already -> no further retry; interp says should_retry=False
    assert interpret_order_state(order_three).should_retry is False


def test_stop_failure_produces_escalation_payload() -> None:
    order = {
        "order_id": "ord-1",
        "symbol": "SPY",
        "protective_stop_status": "failed",
        "rejection_reason": "stop order rejected by broker",
    }
    payload = escalate_protection_failure(order)
    assert payload["escalate"] is True
    assert payload["severity"] == "critical"
    assert payload["order_id"] == "ord-1"


def test_accepted_order_waits_for_broker() -> None:
    order = {"status": "accepted", "filled_qty": 0.0, "quantity": 10.0, "protective_stop_status": "active", "attempts": []}
    interp = interpret_order_state(order)
    assert interp.terminal is False
    assert interp.should_wait is True
    assert decide_retry_or_wait(order) == "wait"


def test_retry_reuses_trade_id_semantics() -> None:
    """The agent's decision logic must never treat different attempt ids as new trades."""
    order = {
        "status": "failed",
        "filled_qty": 0.0,
        "quantity": 10.0,
        "protective_stop_status": "active",
        "trade_id": "trd-1",
        "attempts": [{"attempt_id": "a1"}],
    }
    decision = decide_retry_or_wait(order)
    assert decision == "retry"
    # The same order payload stays associated with trade_id 'trd-1' even on retry.
    assert order["trade_id"] == "trd-1"


def test_bracket_desync_triggers_escalation() -> None:
    order = {
        "order_id": "ord-2",
        "symbol": "SPY",
        "status": "accepted",
        "filled_qty": 0.0,
        "quantity": 10.0,
        "protective_stop_status": "missing",
        "rejection_reason": "bracket_desync:leg-1",
        "attempts": [{"attempt_id": "a1"}],
    }
    interp = interpret_order_state(order)
    assert interp.escalate is True
    assert decide_retry_or_wait(order) == "escalate"
