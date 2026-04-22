"""Phase C2 — Read-only agent tools.

Every read tool must:

- return structured JSON via AgentActionResult
- match API state (or empty state is handled cleanly)
- not hallucinate data when API returns empty/unavailable
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests._api_fixtures import admin_token, boot_client

# Import the adapter and policy classes used to construct the agent
from integrations.openclaw.tool_adapter import OpenClawAdapter
from integrations.openclaw.policy import AgentPolicy, PermissionTier


@pytest.fixture()
def adapter_env(tmp_path: Path):
    """Boot the full stack and return an OpenClawAdapter + service."""
    client, app = boot_client(tmp_path)
    service = app.state.service
    policy = AgentPolicy(tier=PermissionTier.PREVIEW)
    adapter = OpenClawAdapter(service=service, policy=policy)
    return adapter, service


# ------------------------------------------------------------------ get_regime


def test_get_regime_returns_structured_json(adapter_env) -> None:
    adapter, _ = adapter_env
    result = adapter.invoke("get_regime")
    assert result.ok is True
    assert result.action == "get_regime"
    assert result.status == "ok"
    assert isinstance(result.data, dict)
    assert "regime_id" in result.data or "regime_name" in result.data


# ------------------------------------------------------------------ get_portfolio


def test_get_portfolio_returns_structured_json(adapter_env) -> None:
    adapter, _ = adapter_env
    result = adapter.invoke("get_portfolio")
    assert result.ok is True
    assert result.action == "get_portfolio"
    data = result.data
    assert "equity" in data
    assert "positions" in data
    assert isinstance(data["positions"], list)


def test_get_portfolio_empty_positions(adapter_env) -> None:
    adapter, _ = adapter_env
    result = adapter.invoke("get_portfolio")
    # Fresh env has no positions → empty list, not an error
    assert result.ok is True
    assert result.data["positions"] == []


# ------------------------------------------------------------------ get_positions


def test_get_positions_returns_structured_json(adapter_env) -> None:
    adapter, _ = adapter_env
    result = adapter.invoke("get_positions")
    assert result.ok is True
    assert result.action == "get_positions"
    assert "positions" in result.data


# ------------------------------------------------------------------ get_risk_status


def test_get_risk_status_returns_structured_json(adapter_env) -> None:
    adapter, _ = adapter_env
    result = adapter.invoke("get_risk_status")
    assert result.ok is True
    assert result.action == "get_risk_status"
    data = result.data
    assert "breaker_state" in data
    assert "active_constraints" in data


# ------------------------------------------------------------------ get_pending_approvals


def test_get_pending_approvals_returns_list(adapter_env) -> None:
    adapter, _ = adapter_env
    result = adapter.invoke("get_pending_approvals")
    assert result.ok is True
    assert isinstance(result.data["approvals"], list)


# ------------------------------------------------------------------ get_freshness


def test_get_freshness_returns_session_data(adapter_env) -> None:
    adapter, _ = adapter_env
    result = adapter.invoke("get_freshness")
    assert result.ok is True
    data = result.data
    assert "exchange_timezone" in data
    assert "data_freshness_status" in data


# ------------------------------------------------------------------ get_model_governance


def test_get_model_governance_returns_model_info(adapter_env) -> None:
    adapter, _ = adapter_env
    result = adapter.invoke("get_model_governance")
    assert result.ok is True
    data = result.data
    assert "active_model_version" in data
    assert "candidates" in data


# ------------------------------------------------------------------ get_audit_summary


def test_get_audit_summary_returns_events(adapter_env) -> None:
    adapter, _ = adapter_env
    result = adapter.invoke("get_audit_summary", {"limit": 10})
    assert result.ok is True
    assert "events" in result.data
    assert isinstance(result.data["events"], list)


# ------------------------------------------------------------------ unknown tool


def test_unknown_tool_returns_error(adapter_env) -> None:
    adapter, _ = adapter_env
    result = adapter.invoke("nonexistent_tool")
    assert result.ok is False
    assert result.status == "unknown_tool"


# ------------------------------------------------------------------ consistency


def test_all_read_tools_return_ok_status(adapter_env) -> None:
    """Batch check: every read tool returns ok=True."""
    adapter, _ = adapter_env
    read_tools = [
        "get_regime",
        "get_portfolio",
        "get_positions",
        "get_risk_status",
        "get_pending_approvals",
        "get_freshness",
        "get_model_governance",
    ]
    for tool_name in read_tools:
        result = adapter.invoke(tool_name)
        assert result.ok is True, f"{tool_name} returned ok=False: {result.message}"
        assert isinstance(result.data, dict), f"{tool_name} data is not dict"
