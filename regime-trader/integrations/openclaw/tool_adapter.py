"""OpenClaw tool adapter (Spec C).

Every tool returns a structured ``AgentActionResult``. Writes are routed through
the same ``PlatformService`` that the dashboard uses, so the engine enforces risk
and audit rules identically for UI, API, and agent requests.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from api.services import PlatformService
from core.types import AgentActionResult, Direction, IntentStatus, stable_idempotency_key
from integrations.openclaw.policy import (
    AgentPolicy,
    PermissionTier,
    enforce_agent_policy,
    snapshot_for_policy,
)

LOG = logging.getLogger(__name__)


@dataclass
class AgentTool:
    """Specification for a single agent-facing tool."""

    name: str
    description: str
    params_schema: dict[str, Any]
    handler_name: str


TOOL_SPECS: list[AgentTool] = [
    AgentTool(
        name="get_regime",
        description="Return the current regime, probability, and stability diagnostics.",
        params_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler_name="tool_get_regime",
    ),
    AgentTool(
        name="get_portfolio",
        description="Return portfolio equity, cash, drawdown, and active positions.",
        params_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler_name="tool_get_portfolio",
    ),
    AgentTool(
        name="get_positions",
        description="Return currently open positions and their stop levels.",
        params_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler_name="tool_get_positions",
    ),
    AgentTool(
        name="get_risk_status",
        description="Return breaker state, sector caps, and other active risk constraints.",
        params_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler_name="tool_get_risk_status",
    ),
    AgentTool(
        name="get_pending_approvals",
        description="List intents awaiting human approval.",
        params_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler_name="tool_get_pending_approvals",
    ),
    AgentTool(
        name="get_freshness",
        description="Return exchange session + market-data freshness summary.",
        params_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler_name="tool_get_freshness",
    ),
    AgentTool(
        name="get_model_governance",
        description="Return the active model, fallback, and candidate metadata.",
        params_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler_name="tool_get_model_governance",
    ),
    AgentTool(
        name="preview_trade",
        description=(
            "Preview a trade intent. Runs the engine risk checks without submitting."
        ),
        params_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "direction": {"type": "string", "enum": ["LONG", "FLAT"]},
                "allocation_pct": {"type": "number", "minimum": 0, "maximum": 1.5},
                "requested_leverage": {"type": "number", "minimum": 0, "maximum": 4.0},
                "thesis": {"type": "string"},
                "intent_type": {"type": "string"},
            },
            "required": ["symbol", "direction", "allocation_pct"],
        },
        handler_name="tool_preview_trade",
    ),
    AgentTool(
        name="submit_trade_intent",
        description=(
            "Submit a trade intent. If approvals are required, the intent is queued; "
            "otherwise the coordinator releases it to the executor."
        ),
        params_schema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "direction": {"type": "string", "enum": ["LONG", "FLAT"]},
                "allocation_pct": {"type": "number"},
                "requested_leverage": {"type": "number"},
                "thesis": {"type": "string"},
                "intent_type": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["symbol", "direction", "allocation_pct"],
        },
        handler_name="tool_submit_trade_intent",
    ),
    AgentTool(
        name="approve_trade",
        description="Approve a pending intent.",
        params_schema={
            "type": "object",
            "properties": {
                "approval_id": {"type": "string"},
                "reason": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["approval_id"],
        },
        handler_name="tool_approve_trade",
    ),
    AgentTool(
        name="reject_trade",
        description="Reject a pending intent with an explicit reason.",
        params_schema={
            "type": "object",
            "properties": {
                "approval_id": {"type": "string"},
                "reason": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["approval_id", "reason"],
        },
        handler_name="tool_reject_trade",
    ),
    AgentTool(
        name="close_position",
        description="Close a single open position.",
        params_schema={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
        handler_name="tool_close_position",
    ),
    AgentTool(
        name="close_all_positions",
        description="Close every open position via the single-writer execution path.",
        params_schema={"type": "object", "properties": {}, "additionalProperties": False},
        handler_name="tool_close_all_positions",
    ),
    AgentTool(
        name="explain_rejection",
        description=(
            "Return a structured reason why a prior intent/preview was rejected or scaled."
        ),
        params_schema={
            "type": "object",
            "properties": {"intent_id": {"type": "string"}},
            "required": ["intent_id"],
        },
        handler_name="tool_explain_rejection",
    ),
    AgentTool(
        name="get_audit_summary",
        description="Return the most recent audit events for situational awareness.",
        params_schema={
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 200}},
        },
        handler_name="tool_get_audit_summary",
    ),
]


def tool_registry() -> dict[str, AgentTool]:
    return {spec.name: spec for spec in TOOL_SPECS}


def tool_not_found(name: str) -> AgentActionResult:
    return AgentActionResult(
        ok=False,
        action=name,
        status="unknown_tool",
        message=f"Unknown tool '{name}'",
        reason_codes=["unknown_tool"],
    )


@dataclass
class OpenClawAdapter:
    """Single entrypoint OpenClaw uses to invoke tools against the platform."""

    service: PlatformService
    policy: AgentPolicy = field(default_factory=AgentPolicy)
    actor: str = "openclaw"
    _registry: dict[str, AgentTool] = field(default_factory=tool_registry, init=False)

    # -------------------------------------------------------------- dispatch
    def available_tools(self) -> list[AgentTool]:
        return list(self._registry.values())

    def invoke(self, tool: str, params: Mapping[str, Any] | None = None) -> AgentActionResult:
        params = dict(params or {})
        spec = self._registry.get(tool)
        if spec is None:
            return tool_not_found(tool)
        decision = enforce_agent_policy(
            self.policy,
            tool,
            service_snapshot=snapshot_for_policy(self.service, action=tool),
        )
        if not decision.allowed:
            return AgentActionResult(
                ok=False,
                action=tool,
                status="blocked",
                message=decision.reason,
                requires_human_approval=decision.requires_approval,
                reason_codes=decision.blocked_by or [decision.reason],
            )
        handler: Callable[[Mapping[str, Any]], AgentActionResult] = getattr(self, spec.handler_name)
        try:
            result = handler(params)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            LOG.exception("Agent tool %s failed", tool)
            return AgentActionResult(
                ok=False,
                action=tool,
                status="error",
                message=str(exc),
                reason_codes=["tool_exception"],
            )
        if decision.requires_approval and not result.requires_human_approval:
            result.requires_human_approval = True
        return result

    # -------------------------------------------------------------- read tools
    def tool_get_regime(self, params: Mapping[str, Any]) -> AgentActionResult:
        data = self.service.get_regime()
        return AgentActionResult(ok=True, action="get_regime", status="ok", data=data)

    def tool_get_portfolio(self, params: Mapping[str, Any]) -> AgentActionResult:
        data = self.service.get_portfolio()
        return AgentActionResult(ok=True, action="get_portfolio", status="ok", data=data)

    def tool_get_positions(self, params: Mapping[str, Any]) -> AgentActionResult:
        data = self.service.get_portfolio()
        return AgentActionResult(ok=True, action="get_positions", status="ok", data={"positions": data["positions"]})

    def tool_get_risk_status(self, params: Mapping[str, Any]) -> AgentActionResult:
        data = self.service.get_risk_status()
        return AgentActionResult(ok=True, action="get_risk_status", status="ok", data=data)

    def tool_get_pending_approvals(self, params: Mapping[str, Any]) -> AgentActionResult:
        data = self.service.list_approvals()
        return AgentActionResult(ok=True, action="get_pending_approvals", status="ok", data={"approvals": data})

    def tool_get_freshness(self, params: Mapping[str, Any]) -> AgentActionResult:
        data = self.service.get_freshness()
        return AgentActionResult(ok=True, action="get_freshness", status="ok", data=data)

    def tool_get_model_governance(self, params: Mapping[str, Any]) -> AgentActionResult:
        data = self.service.get_model_governance()
        return AgentActionResult(ok=True, action="get_model_governance", status="ok", data=data)

    def tool_get_audit_summary(self, params: Mapping[str, Any]) -> AgentActionResult:
        limit = int(params.get("limit", 25))
        events = self.service.list_audit(limit=limit)
        return AgentActionResult(
            ok=True,
            action="get_audit_summary",
            status="ok",
            data={"events": events},
        )

    # -------------------------------------------------------------- preview / intent
    def tool_preview_trade(self, params: Mapping[str, Any]) -> AgentActionResult:
        payload = self._intent_payload(params)
        outcome = self.service.preview_intent(payload, actor=self.actor, actor_type="agent")
        return _outcome_to_result(action="preview_trade", outcome=outcome)

    def tool_submit_trade_intent(self, params: Mapping[str, Any]) -> AgentActionResult:
        payload = self._intent_payload(params)
        outcome = self.service.submit_intent(payload, actor=self.actor, actor_type="agent")
        return _outcome_to_result(action="submit_trade_intent", outcome=outcome)

    def tool_explain_rejection(self, params: Mapping[str, Any]) -> AgentActionResult:
        intent_id = str(params.get("intent_id", ""))
        record = self.service.repository.get_intent(intent_id)
        if record is None:
            return AgentActionResult(
                ok=False,
                action="explain_rejection",
                resource_id=intent_id,
                status="not_found",
                message=f"No intent with id {intent_id}",
                reason_codes=["unknown_intent"],
            )
        plan_payload = record.payload.get("plan") if record.payload else None
        return AgentActionResult(
            ok=True,
            action="explain_rejection",
            resource_id=intent_id,
            status=record.status,
            message=(plan_payload or {}).get("rejection_reason", "no rejection reason recorded"),
            data={
                "intent_id": intent_id,
                "status": record.status,
                "reason_codes": (plan_payload or {}).get("reason_codes", []),
                "plan": plan_payload,
            },
        )

    # -------------------------------------------------------------- write tools
    def tool_approve_trade(self, params: Mapping[str, Any]) -> AgentActionResult:
        approval_id = str(params["approval_id"])
        try:
            data = self.service.approve(
                approval_id,
                actor=self.actor,
                actor_type="agent",
                reason=str(params.get("reason", "")),
                notes=str(params.get("notes", "")),
            )
        except KeyError:
            return AgentActionResult(ok=False, action="approve_trade", status="not_found", reason_codes=["unknown_approval"])
        except ValueError as exc:
            return AgentActionResult(ok=False, action="approve_trade", status="conflict", message=str(exc), reason_codes=["approval_state"])
        return AgentActionResult(
            ok=True,
            action="approve_trade",
            resource_id=approval_id,
            status="approved",
            data=data,
        )

    def tool_reject_trade(self, params: Mapping[str, Any]) -> AgentActionResult:
        approval_id = str(params["approval_id"])
        try:
            data = self.service.reject(
                approval_id,
                actor=self.actor,
                actor_type="agent",
                reason=str(params.get("reason", "rejected by agent")),
                notes=str(params.get("notes", "")),
            )
        except KeyError:
            return AgentActionResult(ok=False, action="reject_trade", status="not_found", reason_codes=["unknown_approval"])
        except ValueError as exc:
            return AgentActionResult(ok=False, action="reject_trade", status="conflict", message=str(exc), reason_codes=["approval_state"])
        return AgentActionResult(
            ok=True,
            action="reject_trade",
            resource_id=approval_id,
            status="rejected",
            data=data,
        )

    def tool_close_position(self, params: Mapping[str, Any]) -> AgentActionResult:
        symbol = str(params["symbol"])
        result = self.service.close_position(symbol=symbol, actor=self.actor, actor_type="agent")
        return AgentActionResult(
            ok=bool(result.get("closed")),
            action="close_position",
            resource_id=symbol,
            status="closed" if result.get("closed") else "no_position",
            data=result,
        )

    def tool_close_all_positions(self, params: Mapping[str, Any]) -> AgentActionResult:
        closed = self.service.close_all_positions(actor=self.actor, actor_type="agent")
        return AgentActionResult(
            ok=True,
            action="close_all_positions",
            status="ok",
            data={"closed": closed},
        )

    # -------------------------------------------------------------- helpers
    def _intent_payload(self, params: Mapping[str, Any]) -> dict[str, Any]:
        symbol = str(params.get("symbol", "")).upper()
        direction = str(params.get("direction", "LONG")).upper()
        idempotency_key = params.get("idempotency_key") or stable_idempotency_key(
            "openclaw",
            symbol,
            direction,
            params.get("allocation_pct"),
            params.get("intent_type", "open_position"),
        )
        return {
            "symbol": symbol,
            "direction": direction,
            "allocation_pct": float(params.get("allocation_pct", 0.0)),
            "requested_leverage": float(params.get("requested_leverage", 1.0)),
            "intent_type": str(params.get("intent_type", "open_position")),
            "thesis": str(params.get("thesis", "")),
            "timeframe": str(params.get("timeframe", "5m")),
            "requires_confirmation": bool(params.get("requires_confirmation", True)),
            "idempotency_key": idempotency_key,
            "source": "openclaw",
        }


def _outcome_to_result(*, action: str, outcome) -> AgentActionResult:
    plan = outcome.plan
    approved = outcome.decision.approved
    data = {
        "intent_id": outcome.intent.intent_id,
        "plan_id": plan.plan_id,
        "approved": approved,
        "status": plan.status,
        "reason_codes": outcome.decision.reason_codes,
        "reason_message": outcome.decision.reason_message,
        "duplicate": outcome.duplicate,
        "projected_exposure": plan.projected_exposure,
        "projected_sector_exposure": plan.projected_sector_exposure,
        "risk_adjusted_size": plan.risk_adjusted_size,
        "risk_adjusted_leverage": plan.risk_adjusted_leverage,
        "limit_price": plan.limit_price,
        "stop_loss": plan.stop_loss,
    }
    return AgentActionResult(
        ok=approved,
        action=action,
        resource_id=outcome.intent.intent_id,
        status=plan.status,
        message=outcome.decision.reason_message or plan.status,
        requires_human_approval=outcome.intent.status == IntentStatus.PREVIEWED,
        data=data,
        reason_codes=list(outcome.decision.reason_codes),
    )
