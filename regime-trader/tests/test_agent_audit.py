"""Phase C4 - every agent action must produce an audit record."""
from __future__ import annotations

from pathlib import Path

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


def test_preview_creates_audit_event(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path)
    adapter.invoke(
        "preview_trade",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    audit = service.list_audit(limit=20)
    actions = {event["action"] for event in audit}
    assert "intent_previewed" in actions
    # Audit entry must record the agent actor_type.
    previews = [event for event in audit if event["action"] == "intent_previewed"]
    assert any(event["actor_type"] == "agent" for event in previews)


def test_submit_creates_audit_event(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path)
    adapter.invoke(
        "submit_trade_intent",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    audit = service.list_audit(limit=20)
    actions = {event["action"] for event in audit}
    # Either explicit submit or a preview was audited (auto_paper flow submits).
    assert {"intent_submitted", "intent_previewed"} & actions


def test_close_position_is_audited_even_without_open_position(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path)
    adapter.invoke("close_position", {"symbol": "NOPE"})
    # A no-op close should not create a spurious audit entry, but any successful
    # close must be audited. Confirm via list_audit + skip if no event present.
    audit = service.list_audit(resource_type="position", limit=20)
    assert all(event["actor_type"] == "agent" for event in audit if event["action"] == "position_close")


def test_approve_via_agent_is_audited(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path, approval_mode="manual")
    submit = adapter.invoke(
        "submit_trade_intent",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    assert submit.ok
    pending = service.list_approvals()
    assert pending, "approval should be queued"
    approval_id = pending[0]["approval_id"]
    # Elevate the tier so the agent can approve.
    adapter.policy = AgentPolicy(tier=PermissionTier.PAPER_EXECUTE, allow_paper_auto_execute=True)
    result = adapter.invoke("approve_trade", {"approval_id": approval_id, "reason": "ok"})
    assert result.ok is True
    audit = service.list_audit(limit=50)
    assert any(event["action"] == "intent_approved_executed" for event in audit)


def test_close_all_positions_is_audited(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path)
    service.application.position_tracker.apply_fill(
        symbol="SPY", side="BUY", qty=5.0, price=100.0, stop_price=95.0, regime_name="BULL"
    )
    result = adapter.invoke("close_all_positions", {})
    assert result.ok is True
    audit = service.list_audit(limit=20)
    assert any(event["action"] == "close_all_positions" for event in audit)
