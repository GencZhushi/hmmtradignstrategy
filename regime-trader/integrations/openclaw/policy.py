"""OpenClaw agent permission model + live-trading guardrails (Spec C)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping


class PermissionTier(str, Enum):
    READONLY = "agent_readonly"
    PREVIEW = "agent_preview"
    PAPER_EXECUTE = "agent_paper_execute"
    LIVE_EXECUTE = "agent_live_execute"


READ_ACTIONS: frozenset[str] = frozenset(
    {
        "get_regime",
        "get_portfolio",
        "get_positions",
        "get_risk_status",
        "get_pending_approvals",
        "get_freshness",
        "get_model_governance",
        "get_audit_summary",
        "explain_rejection",
    }
)

PREVIEW_ACTIONS: frozenset[str] = frozenset({"preview_trade"})
INTENT_ACTIONS: frozenset[str] = frozenset({"submit_trade_intent"})
WRITE_ACTIONS: frozenset[str] = frozenset(
    {
        "approve_trade",
        "reject_trade",
        "close_position",
        "close_all_positions",
    }
)


@dataclass
class AgentPolicyDecision:
    """Result of evaluating a single tool call against the policy."""

    allowed: bool
    reason: str = ""
    requires_approval: bool = False
    blocked_by: list[str] = field(default_factory=list)


@dataclass
class AgentPolicy:
    """Enforces the rules in Spec C (permissions, live arming, breakers, freshness)."""

    tier: PermissionTier = PermissionTier.PREVIEW
    allow_paper_auto_execute: bool = False
    require_confirmation_for_live: bool = True

    def evaluate(
        self,
        action: str,
        *,
        trading_mode: str,
        live_session_armed: bool,
        breaker_state: str,
        stale_data_blocked: bool,
        uncertainty_mode: bool,
    ) -> AgentPolicyDecision:
        if action in READ_ACTIONS:
            return AgentPolicyDecision(allowed=True)

        # Stale data universally blocks writes and previews.
        if stale_data_blocked and action in PREVIEW_ACTIONS | INTENT_ACTIONS | WRITE_ACTIONS:
            return AgentPolicyDecision(
                allowed=False,
                reason="stale_data_blocks_execution",
                blocked_by=["stale_data"],
            )
        if breaker_state not in {"clear"} and action in INTENT_ACTIONS | WRITE_ACTIONS:
            return AgentPolicyDecision(
                allowed=False,
                reason=f"breaker_active:{breaker_state}",
                blocked_by=[f"breaker:{breaker_state}"],
            )

        if action in PREVIEW_ACTIONS:
            if self.tier == PermissionTier.READONLY:
                return AgentPolicyDecision(allowed=False, reason="readonly_tier", blocked_by=["tier"])
            return AgentPolicyDecision(allowed=True)

        if action in INTENT_ACTIONS:
            if self.tier == PermissionTier.READONLY:
                return AgentPolicyDecision(allowed=False, reason="readonly_tier", blocked_by=["tier"])
            if trading_mode == "live":
                if self.tier != PermissionTier.LIVE_EXECUTE:
                    return AgentPolicyDecision(
                        allowed=False,
                        reason="live_execute_permission_required",
                        blocked_by=["tier:live"],
                    )
                if not live_session_armed:
                    return AgentPolicyDecision(
                        allowed=False,
                        reason="live_session_not_armed",
                        blocked_by=["arming"],
                    )
            if uncertainty_mode:
                return AgentPolicyDecision(
                    allowed=True,
                    requires_approval=True,
                    reason="uncertainty_mode_requires_approval",
                )
            return AgentPolicyDecision(allowed=True, requires_approval=not self.allow_paper_auto_execute)

        if action in WRITE_ACTIONS:
            if self.tier in {PermissionTier.READONLY, PermissionTier.PREVIEW}:
                return AgentPolicyDecision(allowed=False, reason="insufficient_tier", blocked_by=["tier"])
            if trading_mode == "live" and self.tier != PermissionTier.LIVE_EXECUTE:
                return AgentPolicyDecision(allowed=False, reason="live_requires_live_tier", blocked_by=["tier:live"])
            if trading_mode == "live" and not live_session_armed:
                return AgentPolicyDecision(allowed=False, reason="live_session_not_armed", blocked_by=["arming"])
            return AgentPolicyDecision(allowed=True)

        return AgentPolicyDecision(allowed=False, reason="unknown_action", blocked_by=["action"])


def enforce_agent_policy(
    policy: AgentPolicy,
    action: str,
    *,
    service_snapshot: Mapping[str, object],
) -> AgentPolicyDecision:
    trading_mode = str(service_snapshot.get("trading_mode", "paper"))
    live_session_armed = bool(service_snapshot.get("live_session_armed", False))
    breaker_state = str(service_snapshot.get("breaker_state", "clear"))
    stale_data_blocked = bool(service_snapshot.get("stale_data_blocked", False))
    uncertainty_mode = bool(service_snapshot.get("uncertainty_mode", False))
    return policy.evaluate(
        action,
        trading_mode=trading_mode,
        live_session_armed=live_session_armed,
        breaker_state=breaker_state,
        stale_data_blocked=stale_data_blocked,
        uncertainty_mode=uncertainty_mode,
    )


def snapshot_for_policy(service, *, action: str | None = None) -> dict:
    """Snapshot the platform service state the policy cares about."""
    risk = service.get_risk_status()
    freshness = service.get_freshness()
    active_arming = service.repository.active_arming()
    return {
        "action": action,
        "trading_mode": service.trading_mode,
        "live_session_armed": active_arming is not None,
        "breaker_state": risk.get("breaker_state", "clear"),
        "stale_data_blocked": bool(freshness.get("stale_data_blocked")),
        "uncertainty_mode": bool(risk.get("uncertainty_mode", False)),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
