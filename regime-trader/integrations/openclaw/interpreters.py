"""Agent interpretation helpers (Spec C5-C9).

These are pure functions that encode how the OpenClaw agent is allowed to
reason about platform state. Keeping them as small, side-effect-free helpers
makes them trivial to unit-test and keeps the agent from re-implementing
platform policy.

Covered phases:

- **C5** - build idempotency keys, resume pending intents, handle locked state.
- **C6** - interpret order lifecycle (partial fills, retries, stop failures).
- **C7** - interpret data-freshness / session status.
- **C8** - interpret concentration rejections and scaled-trade decisions.
- **C9** - interpret active-model governance state.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


# ---------------------------------------------------------- Phase C5 helpers
def build_idempotency_key(
    actor: str,
    symbol: str,
    direction: str,
    allocation_pct: float | None,
    intent_type: str = "open_position",
    *,
    retry_tag: str | None = None,
) -> str:
    """Deterministic idempotency key so retries collapse to one intent."""
    parts = [actor, symbol.upper(), direction.upper(), str(allocation_pct or ""), intent_type]
    if retry_tag:
        parts.append(f"retry:{retry_tag}")
    payload = "|".join(parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


@dataclass
class ResumeDecision:
    """Agent behaviour when re-encountering a previously submitted intent."""

    action: str  # reuse | new_submit | wait
    intent_id: str | None = None
    reason: str = ""


def resume_pending_intent(existing_intent: Mapping[str, Any] | None) -> ResumeDecision:
    """Decide whether to reuse a stored intent or submit a fresh one.

    Spec C5 forbids creating a new intent when a duplicate idempotency key
    already points at an executed/previewed record. If the prior record is
    still in-flight (``pending`` or ``previewed``) the agent must wait for a
    final decision.
    """
    if existing_intent is None:
        return ResumeDecision(action="new_submit", reason="no_prior_intent")
    status = str(existing_intent.get("status", "")).lower()
    intent_id = str(existing_intent.get("intent_id")) if existing_intent.get("intent_id") else None
    if status in {"executed", "filled", "approved"}:
        return ResumeDecision(action="reuse", intent_id=intent_id, reason="already_executed")
    if status in {"rejected", "failed", "cancelled"}:
        return ResumeDecision(
            action="new_submit",
            intent_id=intent_id,
            reason=f"prior_{status}_ok_to_resubmit",
        )
    # pending / previewed / in-flight -> wait
    return ResumeDecision(action="wait", intent_id=intent_id, reason=f"prior_{status or 'pending'}")


def handle_locked_or_pending_state(
    *,
    record_status: str | None,
    lock_held: bool,
) -> ResumeDecision:
    """Encapsulate the retry-vs-wait choice when a symbol lock is active."""
    if lock_held:
        return ResumeDecision(action="wait", reason="symbol_lock_held")
    if record_status and record_status.lower() in {"pending", "previewed"}:
        return ResumeDecision(action="wait", reason=f"intent_{record_status.lower()}")
    return ResumeDecision(action="reuse", reason="record_terminal")


# ---------------------------------------------------------- Phase C6 helpers
TERMINAL_ORDER_STATES: frozenset[str] = frozenset(
    {"filled", "cancelled", "rejected", "dead"}
)
NON_TERMINAL_ORDER_STATES: frozenset[str] = frozenset(
    {"new", "submitted", "accepted", "partially_filled"}
)


@dataclass
class OrderStateInterpretation:
    """Summary of what the agent should do with an order status payload."""

    terminal: bool
    partially_filled: bool
    should_retry: bool
    should_wait: bool
    escalate: bool
    reason: str = ""


def interpret_order_state(order_payload: Mapping[str, Any]) -> OrderStateInterpretation:
    """Translate an order record into safe agent behaviour."""
    status = str(order_payload.get("status", "")).lower()
    protective = str(order_payload.get("protective_stop_status", "")).lower()
    filled_qty = float(order_payload.get("filled_qty", 0.0) or 0.0)
    quantity = float(order_payload.get("quantity", 0.0) or 0.0)
    attempts = order_payload.get("attempts") or []
    partially_filled = status == "partially_filled" or (
        quantity > 0 and 0 < filled_qty < quantity
    )
    terminal = status in TERMINAL_ORDER_STATES and not partially_filled
    escalate = protective in {"failed", "missing"}
    if partially_filled:
        return OrderStateInterpretation(
            terminal=False,
            partially_filled=True,
            should_retry=False,
            should_wait=True,
            escalate=escalate,
            reason="partial_fill_wait_for_remaining",
        )
    if status == "failed":
        return OrderStateInterpretation(
            terminal=False,
            partially_filled=False,
            should_retry=len(attempts) < 3,
            should_wait=False,
            escalate=escalate,
            reason="failed_retryable" if len(attempts) < 3 else "failed_escalate",
        )
    if status in {"rejected", "dead"}:
        return OrderStateInterpretation(
            terminal=True,
            partially_filled=False,
            should_retry=False,
            should_wait=False,
            escalate=True,
            reason=f"terminal_{status}",
        )
    if status == "cancelled":
        return OrderStateInterpretation(
            terminal=True,
            partially_filled=False,
            should_retry=False,
            should_wait=False,
            escalate=False,
            reason="terminal_cancelled",
        )
    if status == "filled":
        return OrderStateInterpretation(
            terminal=True,
            partially_filled=False,
            should_retry=False,
            should_wait=False,
            escalate=escalate,
            reason="terminal_filled",
        )
    # new / submitted / accepted -> wait for broker to progress
    return OrderStateInterpretation(
        terminal=False,
        partially_filled=False,
        should_retry=False,
        should_wait=True,
        escalate=escalate,
        reason=f"awaiting_broker_{status or 'unknown'}",
    )


def decide_retry_or_wait(
    order_payload: Mapping[str, Any],
    *,
    max_attempts: int = 3,
) -> str:
    """Return ``"retry"``, ``"wait"``, ``"escalate"``, or ``"abort"``."""
    interp = interpret_order_state(order_payload)
    attempts = order_payload.get("attempts") or []
    if interp.escalate:
        return "escalate"
    if interp.terminal:
        return "abort" if interp.reason.startswith("terminal_") and interp.reason != "terminal_filled" else "done"
    if interp.should_retry and len(attempts) < max_attempts:
        return "retry"
    if interp.should_wait:
        return "wait"
    return "wait"


def escalate_protection_failure(order_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Produce the structured escalation payload when a stop fails."""
    status = str(order_payload.get("protective_stop_status", "")).lower()
    reason = str(order_payload.get("rejection_reason", "") or "")
    return {
        "escalate": status in {"failed", "missing"},
        "severity": "critical" if status == "failed" else "warning",
        "reason": reason or status or "unknown",
        "order_id": order_payload.get("order_id"),
        "symbol": order_payload.get("symbol"),
    }


# ---------------------------------------------------------- Phase C7 helpers
@dataclass
class FreshnessInterpretation:
    """Agent-facing view of the freshness payload."""

    session_state: str
    stale: bool
    blocks_execution: bool
    regime_effective_date: str | None
    reason: str = ""


def interpret_freshness_status(payload: Mapping[str, Any]) -> FreshnessInterpretation:
    stale = bool(payload.get("stale_data_blocked")) or bool(payload.get("daily_data_stale")) or bool(payload.get("intraday_data_stale"))
    session_state = str(payload.get("exchange_session_state", "unknown"))
    effective = payload.get("regime_effective_session_date")
    reason = "stale" if stale else f"session_{session_state}"
    return FreshnessInterpretation(
        session_state=session_state,
        stale=stale,
        blocks_execution=stale or session_state not in {"open"},
        regime_effective_date=str(effective) if effective else None,
        reason=reason,
    )


def decide_wait_vs_act(payload: Mapping[str, Any], *, action: str) -> str:
    """Decide whether the agent should ``act`` or ``wait`` given freshness."""
    interp = interpret_freshness_status(payload)
    if action in {"read", "get_regime", "get_portfolio", "get_freshness"}:
        return "act"
    if interp.stale:
        return "wait"
    if interp.blocks_execution and action not in {"preview_trade"}:
        return "wait"
    return "act"


def respect_regime_effective_session(
    payload: Mapping[str, Any],
    *,
    current_regime_date: str | None,
) -> bool:
    """Return True when the agent may use ``current_regime_date`` as authoritative."""
    interp = interpret_freshness_status(payload)
    if interp.stale:
        return False
    if current_regime_date is None or interp.regime_effective_date is None:
        return interp.regime_effective_date is not None
    return str(current_regime_date) == interp.regime_effective_date


# ---------------------------------------------------------- Phase C8 helpers
CONCENTRATION_REASON_CODES: frozenset[str] = frozenset(
    {
        "sector_limit",
        "correlation_limit",
        "portfolio_limit",
        "joint_breach",
        "sector_cap",
        "exposure_cap",
    }
)


@dataclass
class ConcentrationInterpretation:
    """Agent's summary of why a plan was scaled or rejected."""

    reason_codes: list[str]
    blocked: bool
    scaled: bool
    can_retry_without_change: bool
    recommendation: str


def interpret_concentration_rejection(plan_payload: Mapping[str, Any]) -> ConcentrationInterpretation:
    reason_codes = [str(code) for code in plan_payload.get("reason_codes") or []]
    status = str(plan_payload.get("status", "")).lower()
    matched = [code for code in reason_codes if any(tag in code.lower() for tag in CONCENTRATION_REASON_CODES)]
    blocked = status == "rejected" and bool(matched)
    scaled = status == "approved" and bool(plan_payload.get("projected_exposure"))
    if blocked:
        recommendation = "reduce_exposure_before_resubmit"
    elif scaled:
        recommendation = "accept_scaled_or_revise"
    else:
        recommendation = "proceed"
    return ConcentrationInterpretation(
        reason_codes=matched or reason_codes,
        blocked=blocked,
        scaled=scaled,
        can_retry_without_change=not blocked,
        recommendation=recommendation,
    )


def handle_scaled_trade_decision(
    plan_payload: Mapping[str, Any],
    *,
    accept_scaled: bool = True,
) -> dict[str, Any]:
    """Agent-facing policy for scaled plans."""
    interp = interpret_concentration_rejection(plan_payload)
    if interp.blocked:
        return {"action": "skip", "reason": "concentration_blocked", "codes": interp.reason_codes}
    if interp.scaled and not accept_scaled:
        return {"action": "revise", "reason": "scaled_but_not_accepted"}
    if interp.scaled:
        return {"action": "submit_scaled", "reason": "accept_risk_scaling"}
    return {"action": "submit", "reason": "no_scaling_applied"}


def respect_joint_breach_resolution(
    plan_payload: Mapping[str, Any],
    *,
    other_pending_plans: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """When multiple trades would jointly breach limits, yield to the resolution."""
    reason_codes = [str(code).lower() for code in plan_payload.get("reason_codes") or []]
    joint = any("joint" in code for code in reason_codes)
    if joint:
        return {"action": "defer", "reason": "joint_breach_resolution", "codes": reason_codes}
    conflicting = [
        p
        for p in other_pending_plans
        if p.get("symbol") == plan_payload.get("symbol")
        and str(p.get("status", "")).lower() in {"previewed", "pending"}
    ]
    if conflicting:
        return {"action": "wait", "reason": "conflicting_pending_plans", "count": len(conflicting)}
    return {"action": "proceed"}


# ---------------------------------------------------------- Phase C9 helpers
@dataclass
class ModelGovernanceInterpretation:
    """Agent-facing view of model governance."""

    active_version: str | None
    fallback_version: str | None
    candidate_versions: list[str]
    active_is_fallback: bool
    has_unpromoted_candidate: bool
    reason: str = ""


def interpret_active_model_status(payload: Mapping[str, Any]) -> ModelGovernanceInterpretation:
    active = payload.get("active_model_version")
    fallback = payload.get("fallback_model_version")
    candidates = payload.get("candidates") or []
    candidate_versions = [str(c.get("model_version")) for c in candidates if c.get("model_version")]
    unpromoted = [
        c for c in candidates
        if str(c.get("status", "")).lower() == "candidate"
        and c.get("model_version") != active
    ]
    reason = "no_active_model" if active is None else (
        "on_fallback" if active == fallback and fallback is not None else "active_model_ok"
    )
    return ModelGovernanceInterpretation(
        active_version=str(active) if active else None,
        fallback_version=str(fallback) if fallback else None,
        candidate_versions=candidate_versions,
        active_is_fallback=bool(active) and active == fallback,
        has_unpromoted_candidate=bool(unpromoted),
        reason=reason,
    )


def handle_model_rollback_event(
    previous_payload: Mapping[str, Any],
    current_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two governance snapshots and produce a rollback summary."""
    prev_active = previous_payload.get("active_model_version")
    curr_active = current_payload.get("active_model_version")
    if prev_active == curr_active:
        return {"rolled_back": False, "reason": "active_unchanged"}
    return {
        "rolled_back": True,
        "reason": "active_model_changed",
        "previous": prev_active,
        "current": curr_active,
        "reuse_prior_reasoning": False,
    }


def respect_unpromoted_candidate_model(payload: Mapping[str, Any]) -> bool:
    """Return True when the agent is allowed to reason using the active model.

    Agents must never treat a candidate as authoritative. If no active model is
    promoted the agent should refrain from execution-oriented reasoning.
    """
    interp = interpret_active_model_status(payload)
    return interp.active_version is not None
