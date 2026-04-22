"""Phase C8 - agent behaviour under portfolio concentration constraints."""
from __future__ import annotations

from integrations.openclaw.interpreters import (
    handle_scaled_trade_decision,
    interpret_concentration_rejection,
    respect_joint_breach_resolution,
)


def test_concentration_rejection_surfaces_matching_codes() -> None:
    plan = {
        "status": "rejected",
        "reason_codes": ["sector_limit_breach", "exposure_cap"],
        "projected_exposure": 0.85,
    }
    interp = interpret_concentration_rejection(plan)
    assert interp.blocked is True
    assert interp.can_retry_without_change is False
    assert "sector_limit_breach" in interp.reason_codes
    assert interp.recommendation == "reduce_exposure_before_resubmit"


def test_scaled_plan_is_not_blocked() -> None:
    plan = {
        "status": "approved",
        "reason_codes": ["correlation_limit_scaled"],
        "projected_exposure": 0.10,
    }
    interp = interpret_concentration_rejection(plan)
    assert interp.blocked is False
    assert interp.scaled is True
    assert interp.recommendation == "accept_scaled_or_revise"


def test_handle_scaled_trade_decision_submits_when_accepting() -> None:
    plan = {
        "status": "approved",
        "reason_codes": ["correlation_limit"],
        "projected_exposure": 0.10,
    }
    assert handle_scaled_trade_decision(plan, accept_scaled=True)["action"] == "submit_scaled"


def test_handle_scaled_trade_decision_skips_when_blocked() -> None:
    plan = {
        "status": "rejected",
        "reason_codes": ["sector_limit"],
        "projected_exposure": 0.40,
    }
    decision = handle_scaled_trade_decision(plan)
    assert decision["action"] == "skip"


def test_handle_scaled_trade_decision_revises_when_rejecting_scaling() -> None:
    plan = {
        "status": "approved",
        "reason_codes": ["correlation_limit"],
        "projected_exposure": 0.05,
    }
    decision = handle_scaled_trade_decision(plan, accept_scaled=False)
    assert decision["action"] == "revise"


def test_agent_does_not_resubmit_blocked_trades_without_change() -> None:
    """Spec C8 rule: after a concentration block the agent must change exposure first."""
    plan = {
        "status": "rejected",
        "reason_codes": ["sector_limit"],
        "projected_exposure": 0.50,
    }
    interp = interpret_concentration_rejection(plan)
    assert interp.can_retry_without_change is False


def test_joint_breach_triggers_defer() -> None:
    plan = {
        "status": "rejected",
        "reason_codes": ["joint_breach"],
        "symbol": "SPY",
    }
    result = respect_joint_breach_resolution(plan)
    assert result["action"] == "defer"


def test_conflicting_pending_plans_trigger_wait() -> None:
    plan = {"status": "approved", "symbol": "SPY", "reason_codes": []}
    other = [{"symbol": "SPY", "status": "pending"}]
    result = respect_joint_breach_resolution(plan, other_pending_plans=other)
    assert result["action"] == "wait"


def test_clean_plan_proceeds_without_joint_breach() -> None:
    plan = {"status": "approved", "symbol": "SPY", "reason_codes": []}
    result = respect_joint_breach_resolution(plan)
    assert result["action"] == "proceed"
