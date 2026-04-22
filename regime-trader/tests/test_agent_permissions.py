"""Phase C4 - agent permission tiers, live arming, and breaker enforcement."""
from __future__ import annotations

from pathlib import Path

from integrations.openclaw.policy import AgentPolicy, PermissionTier
from integrations.openclaw.tool_adapter import OpenClawAdapter
from tests._api_fixtures import boot_client


def _build_adapter(
    tmp_path: Path,
    *,
    tier: PermissionTier = PermissionTier.PAPER_EXECUTE,
    approval_mode: str = "auto_paper",
) -> tuple[OpenClawAdapter, object]:
    client, app = boot_client(tmp_path, approval_mode=approval_mode)
    service = app.state.service
    adapter = OpenClawAdapter(
        service=service,
        policy=AgentPolicy(tier=tier, allow_paper_auto_execute=True),
    )
    return adapter, service


def test_agent_cannot_execute_live_trades_when_session_not_armed(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path, tier=PermissionTier.LIVE_EXECUTE)
    service.application.config.raw["broker"]["trading_mode"] = "live"
    result = adapter.invoke(
        "submit_trade_intent",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    assert result.ok is False
    assert "arming" in result.reason_codes


def test_agent_cannot_bypass_approval_mode(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path, approval_mode="manual")
    result = adapter.invoke(
        "submit_trade_intent",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    assert result.ok is True
    # The engine auto-creates an approval instead of executing.
    assert result.requires_human_approval is True
    pending = service.list_approvals()
    assert pending, "approval queue must receive the intent"


def test_breaker_state_blocks_new_agent_execution(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path)
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
    preview = adapter.invoke(
        "preview_trade",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    submit = adapter.invoke(
        "submit_trade_intent",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    assert preview.ok is False
    assert submit.ok is False


def test_preview_tier_cannot_execute_writes(tmp_path: Path) -> None:
    adapter, _ = _build_adapter(tmp_path, tier=PermissionTier.PREVIEW)
    close_all = adapter.invoke("close_all_positions", {})
    assert close_all.ok is False
    assert "tier" in close_all.reason_codes


def test_stale_data_blocks_writes_for_all_tiers(tmp_path: Path) -> None:
    """Even LIVE_EXECUTE tier must respect stale-data blocks (Spec C4 safety rule)."""
    adapter, service = _build_adapter(tmp_path, tier=PermissionTier.LIVE_EXECUTE)

    # Monkeypatch freshness so stale_data_blocked=True.
    original = service.get_freshness

    def _stale_freshness():
        payload = original()
        payload["stale_data_blocked"] = True
        return payload

    service.get_freshness = _stale_freshness  # type: ignore[assignment]
    result = adapter.invoke(
        "submit_trade_intent",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    assert result.ok is False
    assert "stale_data" in result.reason_codes
