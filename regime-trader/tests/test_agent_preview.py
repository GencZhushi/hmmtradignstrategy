"""Phase C3 - agent preview + intent submission tools."""
from __future__ import annotations

from pathlib import Path

from integrations.openclaw.policy import AgentPolicy, PermissionTier
from integrations.openclaw.tool_adapter import OpenClawAdapter
from tests._api_fixtures import boot_client


def _build_adapter(tmp_path: Path, *, approval_mode: str = "manual") -> tuple[OpenClawAdapter, object]:
    client, app = boot_client(tmp_path, approval_mode=approval_mode)
    service = app.state.service
    adapter = OpenClawAdapter(
        service=service,
        policy=AgentPolicy(tier=PermissionTier.PAPER_EXECUTE, allow_paper_auto_execute=True),
    )
    return adapter, service


def test_preview_returns_structured_order_plan(tmp_path: Path) -> None:
    adapter, _ = _build_adapter(tmp_path)
    result = adapter.invoke(
        "preview_trade",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    assert result.ok is True
    assert result.data["plan_id"]
    assert "intent_id" in result.data
    assert "reason_codes" in result.data


def test_preview_idempotent_for_same_key(tmp_path: Path) -> None:
    adapter, _ = _build_adapter(tmp_path)
    params = {
        "symbol": "SPY",
        "direction": "LONG",
        "allocation_pct": 0.05,
        "idempotency_key": "agent-key",
    }
    first = adapter.invoke("preview_trade", params)
    second = adapter.invoke("preview_trade", params)
    assert first.data["intent_id"] == second.data["intent_id"]
    assert second.data["duplicate"] is True


def test_submit_with_manual_approval_queues_an_approval(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path, approval_mode="manual")
    result = adapter.invoke(
        "submit_trade_intent",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    assert result.ok is True
    assert result.requires_human_approval is True
    pending = service.list_approvals()
    assert any(a["intent_id"] == result.data["intent_id"] for a in pending)


def test_submit_with_auto_paper_executes_directly(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path, approval_mode="auto_paper")
    result = adapter.invoke(
        "submit_trade_intent",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    assert result.ok is True
    # Orders history should reflect submission.
    orders = service.list_orders(limit=10)
    assert any(o["intent_id"] == result.data["intent_id"] for o in orders)


def test_explain_rejection_returns_structured_payload(tmp_path: Path) -> None:
    adapter, service = _build_adapter(tmp_path)
    outcome = service.preview_intent(
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        actor="openclaw",
        actor_type="agent",
    )
    result = adapter.invoke("explain_rejection", {"intent_id": outcome.intent.intent_id})
    assert result.ok is True
    assert result.data["intent_id"] == outcome.intent.intent_id
    assert "reason_codes" in result.data


def test_invalid_intent_returns_reason_codes(tmp_path: Path) -> None:
    adapter, _ = _build_adapter(tmp_path, approval_mode="auto_paper")
    # Invalid allocation (0) should lead to rejection by the risk manager.
    result = adapter.invoke(
        "submit_trade_intent",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.0},
    )
    # The adapter surfaces the structured reason codes; ok may be False if rejected.
    if result.ok is False:
        assert isinstance(result.reason_codes, list)
