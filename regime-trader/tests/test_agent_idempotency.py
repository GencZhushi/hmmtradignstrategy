"""Phase C5 - agent idempotency, retry behaviour, and locked-state handling."""
from __future__ import annotations

from pathlib import Path

from integrations.openclaw.interpreters import (
    build_idempotency_key,
    handle_locked_or_pending_state,
    resume_pending_intent,
)
from integrations.openclaw.policy import AgentPolicy, PermissionTier
from integrations.openclaw.tool_adapter import OpenClawAdapter
from tests._api_fixtures import boot_client


def _build_adapter(tmp_path: Path, approval_mode: str = "auto_paper") -> tuple[OpenClawAdapter, object]:
    client, app = boot_client(tmp_path, approval_mode=approval_mode)
    service = app.state.service
    adapter = OpenClawAdapter(
        service=service,
        policy=AgentPolicy(tier=PermissionTier.PAPER_EXECUTE, allow_paper_auto_execute=True),
    )
    return adapter, service


def test_build_idempotency_key_is_deterministic() -> None:
    a = build_idempotency_key("openclaw", "SPY", "LONG", 0.05)
    b = build_idempotency_key("openclaw", "SPY", "LONG", 0.05)
    assert a == b
    # Different actor -> different key
    c = build_idempotency_key("user", "SPY", "LONG", 0.05)
    assert c != a


def test_agent_retry_reuses_existing_idempotency_key(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path)
    params = {
        "symbol": "SPY",
        "direction": "LONG",
        "allocation_pct": 0.05,
        "idempotency_key": "agent-retry-key",
    }
    first = adapter.invoke("submit_trade_intent", params)
    second = adapter.invoke("submit_trade_intent", params)
    assert first.data["intent_id"] == second.data["intent_id"]
    assert second.data["duplicate"] is True
    orders = service.list_orders(limit=20)
    assert sum(1 for o in orders if o["intent_id"] == first.data["intent_id"]) <= 1


def test_resume_pending_intent_returns_reuse_for_executed() -> None:
    decision = resume_pending_intent({"intent_id": "i-1", "status": "executed"})
    assert decision.action == "reuse"
    assert decision.intent_id == "i-1"


def test_resume_pending_intent_returns_wait_for_inflight() -> None:
    decision = resume_pending_intent({"intent_id": "i-1", "status": "previewed"})
    assert decision.action == "wait"


def test_resume_pending_intent_allows_resubmit_after_rejection() -> None:
    decision = resume_pending_intent({"intent_id": "i-1", "status": "rejected"})
    assert decision.action == "new_submit"


def test_resume_pending_intent_new_submit_when_no_prior() -> None:
    decision = resume_pending_intent(None)
    assert decision.action == "new_submit"


def test_handle_locked_or_pending_waits_on_lock() -> None:
    decision = handle_locked_or_pending_state(record_status="executed", lock_held=True)
    assert decision.action == "wait"


def test_handle_locked_or_pending_waits_on_previewed_status() -> None:
    decision = handle_locked_or_pending_state(record_status="pending", lock_held=False)
    assert decision.action == "wait"


def test_handle_locked_or_pending_reuses_on_terminal() -> None:
    decision = handle_locked_or_pending_state(record_status="executed", lock_held=False)
    assert decision.action == "reuse"


def test_agent_reports_pending_locked_state_via_submit(tmp_path: Path) -> None:
    """If the policy requires approval, the agent's submit must carry requires_human_approval=True."""
    adapter, service = _build_adapter(tmp_path, approval_mode="manual")
    result = adapter.invoke(
        "submit_trade_intent",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    assert result.ok is True
    assert result.requires_human_approval is True
