"""Phase C1 - agent tool surface, policy, and command parser."""
from __future__ import annotations

from integrations.openclaw.command_parser import parse_agent_request
from integrations.openclaw.policy import AgentPolicy, PermissionTier
from integrations.openclaw.tool_adapter import TOOL_SPECS, tool_registry


def test_required_tools_present() -> None:
    names = {spec.name for spec in TOOL_SPECS}
    required = {
        "get_regime",
        "get_portfolio",
        "get_positions",
        "get_risk_status",
        "get_pending_approvals",
        "preview_trade",
        "submit_trade_intent",
        "approve_trade",
        "reject_trade",
        "close_position",
        "close_all_positions",
        "explain_rejection",
    }
    assert required <= names


def test_tool_registry_returns_specs_by_name() -> None:
    registry = tool_registry()
    assert registry["get_regime"].handler_name == "tool_get_regime"
    assert "submit_trade_intent" in registry


def test_policy_allows_reads_for_all_tiers() -> None:
    for tier in PermissionTier:
        policy = AgentPolicy(tier=tier)
        decision = policy.evaluate(
            "get_regime",
            trading_mode="paper",
            live_session_armed=False,
            breaker_state="clear",
            stale_data_blocked=False,
            uncertainty_mode=False,
        )
        assert decision.allowed is True


def test_policy_blocks_writes_for_readonly() -> None:
    policy = AgentPolicy(tier=PermissionTier.READONLY)
    decision = policy.evaluate(
        "submit_trade_intent",
        trading_mode="paper",
        live_session_armed=False,
        breaker_state="clear",
        stale_data_blocked=False,
        uncertainty_mode=False,
    )
    assert decision.allowed is False
    assert "tier" in decision.blocked_by


def test_policy_blocks_live_submit_without_arming() -> None:
    policy = AgentPolicy(tier=PermissionTier.LIVE_EXECUTE)
    decision = policy.evaluate(
        "submit_trade_intent",
        trading_mode="live",
        live_session_armed=False,
        breaker_state="clear",
        stale_data_blocked=False,
        uncertainty_mode=False,
    )
    assert decision.allowed is False
    assert "arming" in decision.blocked_by


def test_policy_blocks_writes_when_breaker_active() -> None:
    policy = AgentPolicy(tier=PermissionTier.PAPER_EXECUTE)
    decision = policy.evaluate(
        "submit_trade_intent",
        trading_mode="paper",
        live_session_armed=False,
        breaker_state="daily_halt",
        stale_data_blocked=False,
        uncertainty_mode=False,
    )
    assert decision.allowed is False
    assert any("breaker" in b for b in decision.blocked_by)


def test_policy_blocks_when_data_is_stale() -> None:
    policy = AgentPolicy(tier=PermissionTier.PAPER_EXECUTE)
    decision = policy.evaluate(
        "preview_trade",
        trading_mode="paper",
        live_session_armed=False,
        breaker_state="clear",
        stale_data_blocked=True,
        uncertainty_mode=False,
    )
    assert decision.allowed is False
    assert "stale_data" in decision.blocked_by


def test_uncertainty_mode_requires_approval() -> None:
    policy = AgentPolicy(tier=PermissionTier.PAPER_EXECUTE)
    decision = policy.evaluate(
        "submit_trade_intent",
        trading_mode="paper",
        live_session_armed=False,
        breaker_state="clear",
        stale_data_blocked=False,
        uncertainty_mode=True,
    )
    assert decision.allowed is True
    assert decision.requires_approval is True


def test_command_parser_detects_read_intents() -> None:
    assert parse_agent_request("what is the current regime?").tool == "get_regime"
    assert parse_agent_request("what positions are open?").tool == "get_positions"
    assert parse_agent_request("show portfolio").tool == "get_portfolio"
    assert parse_agent_request("is the market data stale?").tool == "get_freshness"


def test_command_parser_parses_preview_trade() -> None:
    parsed = parse_agent_request("preview SPY 10%")
    assert parsed.tool == "preview_trade"
    assert parsed.params["symbol"] == "SPY"
    assert parsed.params["allocation_pct"] == 0.10
    assert parsed.params["direction"] == "LONG"


def test_command_parser_parses_close_position() -> None:
    parsed = parse_agent_request("close my position in QQQ")
    assert parsed.tool == "close_position"
    assert parsed.params["symbol"] == "QQQ"


def test_command_parser_unknown_returns_parse_error() -> None:
    parsed = parse_agent_request("tell me a joke")
    assert parsed.tool == "parse_error"
    assert parsed.params["reason"] == "unrecognized_command"
