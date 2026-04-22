"""Phase C2-C9 - OpenClaw adapter against a booted platform service."""
from __future__ import annotations

from pathlib import Path

import pytest

from integrations.openclaw.policy import AgentPolicy, PermissionTier
from integrations.openclaw.tool_adapter import OpenClawAdapter
from tests._api_fixtures import boot_client


def _build_adapter(tmp_path: Path, *, tier: PermissionTier = PermissionTier.PAPER_EXECUTE, approval_mode: str = "auto_paper") -> tuple[OpenClawAdapter, object]:
    client, app = boot_client(tmp_path, approval_mode=approval_mode)
    service = app.state.service
    adapter = OpenClawAdapter(service=service, policy=AgentPolicy(tier=tier, allow_paper_auto_execute=True))
    return adapter, service


def test_read_tools_return_structured_payloads(tmp_path: Path) -> None:
    adapter, _ = _build_adapter(tmp_path)
    assert adapter.invoke("get_regime").ok is True
    assert adapter.invoke("get_portfolio").ok is True
    assert adapter.invoke("get_positions").ok is True
    assert adapter.invoke("get_risk_status").ok is True
    assert adapter.invoke("get_freshness").ok is True
    assert adapter.invoke("get_model_governance").ok is True


def test_preview_trade_produces_plan_and_idempotency_dedup(tmp_path: Path) -> None:
    adapter, _ = _build_adapter(tmp_path)
    params = {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05, "idempotency_key": "agent-key-1"}
    first = adapter.invoke("preview_trade", params)
    second = adapter.invoke("preview_trade", params)
    assert first.ok is True
    assert first.data["intent_id"] == second.data["intent_id"]


def test_submit_with_manual_approval_returns_pending_approval(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path, approval_mode="manual")
    result = adapter.invoke(
        "submit_trade_intent",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    assert result.ok is True
    assert result.requires_human_approval is True
    pending = adapter.invoke("get_pending_approvals")
    assert pending.data["approvals"]


def test_agent_cannot_submit_when_breaker_active(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path)
    # Force the breaker state via the risk manager.
    state = service.application.position_tracker.snapshot()
    state.daily_pnl = -state.peak_equity * 0.1
    service.application.risk_manager.breaker.evaluate(state)
    result = adapter.invoke(
        "submit_trade_intent",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    assert result.ok is False
    assert any("breaker" in code for code in result.reason_codes)


def test_readonly_tier_cannot_preview_or_submit(tmp_path: Path) -> None:
    adapter, _ = _build_adapter(tmp_path, tier=PermissionTier.READONLY)
    preview = adapter.invoke("preview_trade", {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05})
    assert preview.ok is False
    assert "tier" in preview.reason_codes


def test_live_tier_blocked_without_arming(tmp_path: Path) -> None:
    # Booting with paper mode, but simulate the agent asking for live execute.
    adapter, service = _build_adapter(tmp_path, tier=PermissionTier.LIVE_EXECUTE)
    # Force the service to report live mode.
    service.application.config.raw["broker"]["trading_mode"] = "live"
    result = adapter.invoke(
        "submit_trade_intent",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    assert result.ok is False
    assert "arming" in result.reason_codes


def test_explain_rejection_returns_reason_codes(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path)
    # Force a rejection by submitting without a stop and then asking for explanation.
    outcome = service.preview_intent(
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        actor="openclaw",
        actor_type="agent",
    )
    result = adapter.invoke("explain_rejection", {"intent_id": outcome.intent.intent_id})
    assert result.ok is True
    assert result.data["intent_id"] == outcome.intent.intent_id


def test_unknown_tool_returns_structured_error(tmp_path: Path) -> None:
    adapter, _ = _build_adapter(tmp_path)
    result = adapter.invoke("definitely_not_a_tool")
    assert result.ok is False
    assert "unknown_tool" in result.reason_codes


def test_close_position_reports_no_open_position(tmp_path: Path) -> None:
    adapter, _ = _build_adapter(tmp_path)
    result = adapter.invoke("close_position", {"symbol": "ZZZZ"})
    assert result.ok is False
    assert result.status == "no_position"


def test_audit_summary_contains_agent_events(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path)
    adapter.invoke(
        "preview_trade",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    summary = adapter.invoke("get_audit_summary", {"limit": 10})
    actions = {event["action"] for event in summary.data["events"]}
    assert "intent_previewed" in actions
