"""Phase C7 - agent behaviour under stale data and session constraints."""
from __future__ import annotations

from pathlib import Path

from integrations.openclaw.interpreters import (
    decide_wait_vs_act,
    interpret_freshness_status,
    respect_regime_effective_session,
)
from integrations.openclaw.policy import AgentPolicy, PermissionTier
from integrations.openclaw.tool_adapter import OpenClawAdapter
from tests._api_fixtures import boot_client


def test_fresh_payload_is_interpreted_as_not_stale() -> None:
    payload = {
        "exchange_session_state": "open",
        "daily_data_stale": False,
        "intraday_data_stale": False,
        "stale_data_blocked": False,
        "regime_effective_session_date": "2024-10-01",
    }
    interp = interpret_freshness_status(payload)
    assert interp.stale is False
    assert interp.blocks_execution is False
    assert interp.reason.startswith("session_")


def test_stale_payload_blocks_execution() -> None:
    payload = {
        "exchange_session_state": "post_market",
        "daily_data_stale": True,
        "intraday_data_stale": False,
        "stale_data_blocked": True,
        "regime_effective_session_date": None,
    }
    interp = interpret_freshness_status(payload)
    assert interp.stale is True
    assert interp.blocks_execution is True


def test_decide_wait_vs_act_waits_when_stale() -> None:
    payload = {
        "exchange_session_state": "open",
        "stale_data_blocked": True,
        "daily_data_stale": True,
        "intraday_data_stale": False,
    }
    assert decide_wait_vs_act(payload, action="submit_trade_intent") == "wait"


def test_decide_wait_vs_act_acts_when_fresh() -> None:
    payload = {
        "exchange_session_state": "open",
        "stale_data_blocked": False,
        "daily_data_stale": False,
        "intraday_data_stale": False,
    }
    assert decide_wait_vs_act(payload, action="submit_trade_intent") == "act"


def test_reads_always_allowed_even_when_stale() -> None:
    payload = {
        "exchange_session_state": "closed",
        "stale_data_blocked": True,
        "daily_data_stale": True,
        "intraday_data_stale": True,
    }
    assert decide_wait_vs_act(payload, action="get_regime") == "act"


def test_respect_regime_effective_session_matches_date() -> None:
    payload = {
        "stale_data_blocked": False,
        "daily_data_stale": False,
        "intraday_data_stale": False,
        "exchange_session_state": "open",
        "regime_effective_session_date": "2024-10-01",
    }
    assert respect_regime_effective_session(payload, current_regime_date="2024-10-01") is True
    assert respect_regime_effective_session(payload, current_regime_date="2024-10-02") is False


def test_respect_regime_effective_session_false_when_stale() -> None:
    payload = {
        "stale_data_blocked": True,
        "daily_data_stale": True,
        "intraday_data_stale": False,
        "exchange_session_state": "closed",
        "regime_effective_session_date": "2024-10-01",
    }
    assert respect_regime_effective_session(payload, current_regime_date="2024-10-01") is False


def test_agent_respects_freshness_policy_via_adapter(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    service = app.state.service
    adapter = OpenClawAdapter(
        service=service,
        policy=AgentPolicy(tier=PermissionTier.PAPER_EXECUTE, allow_paper_auto_execute=True),
    )

    # Force stale freshness so the policy must block the submit.
    original = service.get_freshness

    def _stale():
        payload = original()
        payload["stale_data_blocked"] = True
        return payload

    service.get_freshness = _stale  # type: ignore[assignment]

    result = adapter.invoke(
        "submit_trade_intent",
        {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
    )
    assert result.ok is False
    assert "stale_data" in result.reason_codes
