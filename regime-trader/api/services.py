"""Platform service layer: the only place the API uses the engine directly.

API routes must call into this module instead of engine internals so Spec B
stays decoupled from Spec A. Responsibilities:

- wrap the ``ExecutionCoordinator`` so previews/executions go through the
  engine's single-writer path;
- persist intents/orders/approvals/audit events via the ``Repository``;
- fan out lifecycle events to WebSocket/SSE listeners;
- expose the read-model that routes and the dashboard need.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping

from core.correlation_risk import compute_rolling_return_correlation
from core.execution_coordinator import CoordinatorOutcome, ExecutionCoordinator
from core.model_registry import ModelRegistry, PromotionRejected
from core.types import (
    AgentActionResult,
    AuditEvent,
    Direction,
    IntentStatus,
    OrderPlan,
    PortfolioState,
    TradeIntent,
    new_intent_id,
    stable_idempotency_key,
)
from data.exchange_calendar import ExchangeCalendar, freshness_payload
from monitoring.application import TradingApplication
from storage.repository import Repository

LOG = logging.getLogger(__name__)


@dataclass
class ApprovalPolicy:
    """Mutable view of the approval policy used by the platform."""

    mode: str = "manual"  # manual | auto_paper
    require_approval_in_paper: bool = True

    def auto_execute(self, *, trading_mode: str) -> bool:
        if self.mode == "auto_paper" and trading_mode == "paper":
            return True
        return False


@dataclass
class PlatformEvent:
    """Envelope pushed to the UI / agent streaming sink."""

    event: str
    payload: dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


EventListener = Callable[[PlatformEvent], None]


@dataclass
class PlatformService:
    """Facade the API routes and OpenClaw adapter share."""

    application: TradingApplication
    repository: Repository
    approval_policy: ApprovalPolicy
    event_queue: deque[PlatformEvent] = field(default_factory=lambda: deque(maxlen=500))
    listeners: list[EventListener] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.application.register_listener(self._on_engine_event)

    # ---------------------------------------------------------- identity
    @property
    def config(self):
        return self.application.config

    @property
    def trading_mode(self) -> str:
        return str(self.config.get("broker.trading_mode", "paper"))

    # ---------------------------------------------------------- read model
    def get_health(self) -> dict[str, Any]:
        broker = getattr(self.application.executor, "broker", None)
        return {
            "status": "ok",
            "trading_mode": self.trading_mode,
            "execution_enabled": bool(self.config.get("broker.execution_enabled", True)),
            "dry_run": self.application.dry_run,
            "active_model": self.application.model_registry.active_version,
            "broker_connected": getattr(broker, "is_connected", True),
            "last_reconciliation": None,
        }

    def get_portfolio(self) -> dict[str, Any]:
        state = self.application.position_tracker.snapshot()
        classifier = self.application.risk_manager.sector_classifier
        sector_map = state.sector_exposure(classifier.get_sector_bucket)
        positions = [
            {
                "symbol": pos.symbol,
                "quantity": pos.quantity,
                "avg_entry_price": pos.avg_entry_price,
                "current_price": pos.current_price,
                "stop_price": pos.stop_price,
                "regime_at_entry": pos.regime_at_entry,
                "unrealized_pnl": pos.unrealized_pnl,
                "market_value": pos.market_value,
                "sector_bucket": classifier.get_sector_bucket(pos.symbol),
                "etf_bucket": classifier.get_etf_risk_bucket(pos.symbol),
            }
            for pos in state.positions.values()
        ]
        return {
            "equity": state.equity,
            "cash": state.cash,
            "buying_power": state.buying_power,
            "daily_pnl": state.daily_pnl,
            "weekly_pnl": state.weekly_pnl,
            "peak_equity": state.peak_equity,
            "drawdown": state.drawdown,
            "breaker_state": state.breaker_state.value if hasattr(state.breaker_state, "value") else str(state.breaker_state),
            "total_exposure_pct": state.total_exposure_pct,
            "positions": positions,
            "sector_exposure": sector_map,
        }

    def get_regime(self) -> dict[str, Any]:
        diag = self.application.signal_generator.diagnostics
        state = diag.state
        return {
            "regime_id": state.regime_id if state else None,
            "regime_name": state.regime_name if state else None,
            "probability": state.probability if state else 0.0,
            "state_probabilities": state.state_probabilities if state else [],
            "is_confirmed": state.is_confirmed if state else False,
            "consecutive_bars": state.consecutive_bars if state else 0,
            "flicker_rate": state.flicker_rate if state else 0.0,
            "effective_session_date": (
                diag.last_update_session_date.isoformat()
                if diag.last_update_session_date
                else None
            ),
            "active_model_version": self.application.model_registry.active_version,
        }

    def get_freshness(self) -> dict[str, Any]:
        symbols = self.config.get("broker.symbols", [])
        snapshot = self.application.market_data.freshness_snapshot(symbols)
        return freshness_payload(
            last_daily_bar=_first_timestamp(snapshot, "last_completed_daily_bar_time"),
            last_intraday_bar=_first_timestamp(snapshot, "last_completed_intraday_bar_time"),
            now=datetime.now(timezone.utc),
            calendar=self.application.calendar,
        )

    def get_risk_status(self) -> dict[str, Any]:
        state = self.application.position_tracker.snapshot()
        limits = self.application.risk_manager.limits
        return {
            "breaker_state": state.breaker_state.value if hasattr(state.breaker_state, "value") else str(state.breaker_state),
            "active_constraints": {
                "max_exposure": limits.max_exposure,
                "max_single_position": limits.max_single_position,
                "max_concurrent": limits.max_concurrent,
                "max_leverage": limits.max_leverage,
                "max_sector_exposure": limits.max_sector_exposure,
            },
            "blocked_symbols": sorted(state.blocked_symbols),
            "uncertainty_mode": self.application.risk_manager.uncertainty_mode,
            "daily_trade_count": state.daily_trade_count,
        }

    def get_model_governance(self) -> dict[str, Any]:
        registry: ModelRegistry = self.application.model_registry
        active = registry.active_entry()
        fallback = registry.fallback_entry()
        return {
            "active_model_version": active.model_version if active else None,
            "fallback_model_version": fallback.model_version if fallback else None,
            "active_training_metadata": asdict(active) if active else None,
            "candidates": [asdict(entry) for entry in registry.list_versions()],
            "last_promotion_decision": None,
        }

    def get_concentration(self) -> dict[str, Any]:
        state = self.application.position_tracker.snapshot()
        classifier = self.application.risk_manager.sector_classifier
        sector_map = state.sector_exposure(classifier.get_sector_bucket)
        correlations: dict[str, float] = {}
        returns_history = self.application.risk_manager.returns_history
        if not returns_history.empty:
            corr = compute_rolling_return_correlation(returns_history)
            if not corr.empty:
                columns = list(corr.columns)
                for i, a in enumerate(columns):
                    for b in columns[i + 1 :]:
                        correlations[f"{a}~{b}"] = float(corr.loc[a, b])
        return {
            "sector_exposure": sector_map,
            "projected_post_trade_exposure": sector_map,
            "correlation_metrics": correlations,
            "blocked_reasons": [],
        }

    def list_orders(self, *, limit: int = 100) -> list[dict[str, Any]]:
        records = self.repository.list_orders(limit=limit)
        return [_order_to_dict(r) for r in records]

    def list_intents(self, *, limit: int = 50) -> list[dict[str, Any]]:
        records = self.repository.list_intents(limit=limit)
        return [_intent_to_dict(r) for r in records]

    def list_audit(self, *, limit: int = 100, resource_type: str | None = None, actor: str | None = None) -> list[dict[str, Any]]:
        records = self.repository.list_audit(limit=limit, resource_type=resource_type, actor=actor)
        return [_audit_to_dict(r) for r in records]

    def list_approvals(self, *, include_history: bool = False) -> list[dict[str, Any]]:
        records = self.repository.list_approvals() if include_history else self.repository.pending_approvals()
        return [_approval_to_dict(r) for r in records]

    def latest_signals(self, *, limit: int = 25) -> list[dict[str, Any]]:
        return list(self.application.recent_signals)[-limit:]

    # ---------------------------------------------------------- write actions
    def preview_intent(
        self,
        payload: Mapping[str, Any],
        *,
        actor: str,
        actor_type: str,
    ) -> CoordinatorOutcome:
        intent = self._build_intent(payload, actor=actor, actor_type=actor_type)
        outcome = self.application.coordinator.preview_intent(intent)
        self._persist_intent(outcome)
        self._record_audit(
            action="intent_previewed",
            resource_type="intent",
            resource_id=outcome.intent.intent_id,
            actor=actor,
            actor_type=actor_type,
            reason=outcome.decision.reason_message,
            after={
                "plan_id": outcome.plan.plan_id,
                "approved": outcome.decision.approved,
                "reason_codes": outcome.decision.reason_codes,
            },
        )
        return outcome

    def submit_intent(
        self,
        payload: Mapping[str, Any],
        *,
        actor: str,
        actor_type: str,
    ) -> CoordinatorOutcome:
        policy_auto = self.approval_policy.auto_execute(trading_mode=self.trading_mode)
        if self.approval_policy.require_approval_in_paper and not policy_auto:
            outcome = self.preview_intent(payload, actor=actor, actor_type=actor_type)
            if outcome.decision.approved:
                self.repository.create_approval(
                    intent_id=outcome.intent.intent_id,
                    plan_id=outcome.plan.plan_id,
                    requested_by=actor,
                    requested_by_type=actor_type,
                    payload={
                        "symbol": outcome.intent.symbol,
                        "allocation_pct": outcome.intent.allocation_pct,
                        "risk_adjusted_size": outcome.plan.risk_adjusted_size,
                        "reason_codes": outcome.decision.reason_codes,
                    },
                )
                outcome.intent.status = IntentStatus.PREVIEWED
                self._emit("approval_pending", {
                    "intent_id": outcome.intent.intent_id,
                    "plan_id": outcome.plan.plan_id,
                    "requested_by": actor,
                    "requested_by_type": actor_type,
                })
            return outcome

        intent = self._build_intent(payload, actor=actor, actor_type=actor_type)
        outcome = self.application.coordinator.submit_intent(intent)
        self._persist_intent(outcome)
        self._record_audit(
            action="intent_submitted",
            resource_type="intent",
            resource_id=outcome.intent.intent_id,
            actor=actor,
            actor_type=actor_type,
            reason=outcome.decision.reason_message,
            after={
                "plan_id": outcome.plan.plan_id,
                "approved": outcome.decision.approved,
                "status": outcome.plan.status,
                "reason_codes": outcome.decision.reason_codes,
            },
        )
        if outcome.order is not None:
            self.repository.upsert_order(self.application.state_machine.summary(outcome.order.order_id))
        return outcome

    def approve(
        self,
        approval_id: str,
        *,
        actor: str,
        actor_type: str,
        reason: str = "",
        notes: str = "",
    ) -> dict[str, Any]:
        approval = self.repository.resolve_approval(
            approval_id,
            status="approved",
            decided_by=actor,
            decided_by_type=actor_type,
            reason=reason,
            notes=notes,
        )
        intent_record = self.repository.get_intent(approval.intent_id)
        if intent_record is None:
            raise KeyError(f"Intent {approval.intent_id} missing for approval")
        # Execute via the coordinator directly, bypassing the approval policy and
        # using a fresh idempotency key so the engine doesn't treat the approval
        # handoff as a duplicate of the original preview.
        post_key = f"{intent_record.idempotency_key}:approved:{approval.approval_id}"
        intent = TradeIntent(
            symbol=intent_record.symbol,
            direction=Direction(intent_record.direction),
            allocation_pct=float(intent_record.allocation_pct),
            requested_leverage=float(intent_record.requested_leverage),
            intent_type=intent_record.intent_type,
            idempotency_key=post_key,
            source=intent_record.actor_type,
            actor=actor,
            thesis=intent_record.thesis,
            requires_confirmation=False,
        )
        outcome = self.application.coordinator.submit_intent(intent)
        self._persist_intent(outcome)
        self._record_audit(
            action="intent_approved_executed",
            resource_type="intent",
            resource_id=approval.intent_id,
            actor=actor,
            actor_type=actor_type,
            reason=reason or approval.reason,
            after={
                "plan_id": outcome.plan.plan_id,
                "status": outcome.plan.status,
                "approval_id": approval.approval_id,
            },
        )
        if outcome.order is not None:
            self.repository.upsert_order(self.application.state_machine.summary(outcome.order.order_id))
        self._emit("approval_resolved", {
            "approval_id": approval.approval_id,
            "intent_id": approval.intent_id,
            "status": "approved",
            "decided_by": actor,
        })
        return {
            "approval": _approval_to_dict(approval),
            "outcome": {
                "intent_id": outcome.intent.intent_id,
                "plan_id": outcome.plan.plan_id,
                "approved": outcome.decision.approved,
                "status": outcome.plan.status,
            },
        }

    def reject(
        self,
        approval_id: str,
        *,
        actor: str,
        actor_type: str,
        reason: str = "",
        notes: str = "",
    ) -> dict[str, Any]:
        approval = self.repository.resolve_approval(
            approval_id,
            status="rejected",
            decided_by=actor,
            decided_by_type=actor_type,
            reason=reason,
            notes=notes,
        )
        self._record_audit(
            action="approval_rejected",
            resource_type="approval",
            resource_id=approval.approval_id,
            actor=actor,
            actor_type=actor_type,
            reason=reason,
            after={"status": "rejected", "intent_id": approval.intent_id},
        )
        self._emit("approval_resolved", {
            "approval_id": approval.approval_id,
            "intent_id": approval.intent_id,
            "status": "rejected",
            "decided_by": actor,
        })
        return {"approval": _approval_to_dict(approval)}

    def close_position(self, *, symbol: str, actor: str, actor_type: str) -> dict[str, Any]:
        state = self.application.position_tracker.snapshot()
        if symbol not in state.positions:
            return {"closed": False, "reason": "no_open_position"}
        quantity = state.positions[symbol].quantity
        self.application.executor.close_position(symbol=symbol, qty=quantity)
        self._record_audit(
            action="position_close",
            resource_type="position",
            resource_id=symbol,
            actor=actor,
            actor_type=actor_type,
            reason="manual close",
            after={"quantity": quantity},
        )
        return {"closed": True, "symbol": symbol, "quantity": quantity}

    def close_all_positions(self, *, actor: str, actor_type: str) -> list[dict[str, Any]]:
        state = self.application.position_tracker.snapshot()
        closed: list[dict[str, Any]] = []
        for symbol, pos in state.positions.items():
            self.application.executor.close_position(symbol=symbol, qty=pos.quantity)
            closed.append({"symbol": symbol, "quantity": pos.quantity})
        self._record_audit(
            action="close_all_positions",
            resource_type="portfolio",
            resource_id="root",
            actor=actor,
            actor_type=actor_type,
            reason="operator close-all",
            after={"closed": closed},
        )
        return closed

    def arm_live(self, *, actor: str, ttl_minutes: int, reason: str) -> dict[str, Any]:
        record = self.repository.arm_live(armed_by=actor, ttl=timedelta(minutes=ttl_minutes), reason=reason)
        self._record_audit(
            action="arm_live",
            resource_type="session",
            resource_id=str(record.id),
            actor=actor,
            actor_type="user",
            reason=reason,
            after={"expires_at": record.expires_at.isoformat()},
        )
        self._emit("live_armed", {"actor": actor, "expires_at": record.expires_at.isoformat()})
        return {
            "armed_by": record.armed_by,
            "expires_at": record.expires_at.isoformat(),
            "reason": record.reason,
        }

    def reload_config(self, *, actor: str, reason: str) -> dict[str, Any]:
        self._record_audit(
            action="reload_config",
            resource_type="config",
            resource_id="runtime",
            actor=actor,
            actor_type="user",
            reason=reason,
        )
        self._emit("config_reloaded", {"actor": actor, "reason": reason})
        return {"status": "ok", "actor": actor}

    def promote_model(self, *, version: str, actor: str) -> dict[str, Any]:
        try:
            entry = self.application.model_registry.promote_model(version, actor=actor)
        except PromotionRejected as exc:
            return {"approved": False, "reason": exc.decision.reason}
        self._emit("model_promoted", {"model_version": entry.model_version, "actor": actor})
        return {"approved": True, "model_version": entry.model_version}

    def rollback_model(self, *, actor: str) -> dict[str, Any]:
        entry = self.application.model_registry.rollback_model(actor=actor)
        self._emit("model_rollback", {"model_version": entry.model_version if entry else None, "actor": actor})
        return {"model_version": entry.model_version if entry else None}

    # ---------------------------------------------------------- streaming
    def register_listener(self, listener: EventListener) -> None:
        self.listeners.append(listener)

    def drain_events(self) -> list[PlatformEvent]:
        with self._lock:
            events = list(self.event_queue)
            self.event_queue.clear()
            return events

    def recent_events(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {"event": e.event, "payload": e.payload, "timestamp": e.timestamp.isoformat()}
                for e in list(self.event_queue)[-limit:]
            ]

    # ---------------------------------------------------------- internals
    def _build_intent(
        self,
        payload: Mapping[str, Any],
        *,
        actor: str,
        actor_type: str,
    ) -> TradeIntent:
        direction = Direction(payload.get("direction", "LONG"))
        idempotency_key = payload.get("idempotency_key") or stable_idempotency_key(
            actor_type,
            payload.get("symbol"),
            direction.value,
            payload.get("allocation_pct"),
            payload.get("intent_type", "open_position"),
        )
        return TradeIntent(
            symbol=str(payload["symbol"]),
            direction=direction,
            allocation_pct=float(payload.get("allocation_pct", 0.0)),
            requested_leverage=float(payload.get("requested_leverage", 1.0)),
            intent_type=str(payload.get("intent_type", "open_position")),
            intent_id=new_intent_id(),
            idempotency_key=idempotency_key,
            source=str(payload.get("source", actor_type)),
            actor=actor,
            thesis=str(payload.get("thesis", "")),
            timeframe=str(payload.get("timeframe", "5m")),
            requires_confirmation=bool(payload.get("requires_confirmation", True)),
        )

    def _persist_intent(self, outcome: CoordinatorOutcome) -> None:
        status = outcome.plan.status if outcome.decision.approved else "rejected"
        self.repository.upsert_intent(outcome.intent, plan=outcome.plan, status=status)

    def _record_audit(
        self,
        *,
        action: str,
        resource_type: str,
        resource_id: str,
        actor: str,
        actor_type: str,
        reason: str = "",
        before: Mapping[str, Any] | None = None,
        after: Mapping[str, Any] | None = None,
    ) -> None:
        event = AuditEvent(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            actor=actor,
            actor_type=actor_type,
            reason=reason,
            before=dict(before) if before else None,
            after=dict(after) if after else None,
        )
        self.repository.record_audit(event)
        self._emit("audit_event", {
            "event_id": event.event_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "actor": actor,
            "actor_type": actor_type,
        })
        self.application.audit_log.append(event)

    def _emit(self, event: str, payload: Mapping[str, Any]) -> None:
        envelope = PlatformEvent(event=event, payload=dict(payload))
        with self._lock:
            self.event_queue.append(envelope)
        for listener in list(self.listeners):
            try:
                listener(envelope)
            except Exception as exc:  # pragma: no cover - listener isolation
                LOG.error("Platform listener failed on %s: %s", event, exc)

    def _on_engine_event(self, event: str, payload: Mapping[str, Any]) -> None:
        self._emit(event, payload)


def _intent_to_dict(record) -> dict[str, Any]:
    return {
        "intent_id": record.intent_id,
        "idempotency_key": record.idempotency_key,
        "symbol": record.symbol,
        "direction": record.direction,
        "allocation_pct": record.allocation_pct,
        "requested_leverage": record.requested_leverage,
        "intent_type": record.intent_type,
        "thesis": record.thesis,
        "status": record.status,
        "actor": record.actor,
        "actor_type": record.actor_type,
        "plan_id": record.plan_id,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _order_to_dict(record) -> dict[str, Any]:
    return {
        "order_id": record.order_id,
        "trade_id": record.trade_id,
        "intent_id": record.intent_id,
        "symbol": record.symbol,
        "side": record.side,
        "quantity": record.quantity,
        "filled_qty": record.filled_qty,
        "avg_fill_price": record.avg_fill_price,
        "status": record.status,
        "limit_price": record.limit_price,
        "stop_price": record.stop_price,
        "take_profit": record.take_profit,
        "protective_stop_status": record.protective_stop_status,
        "rejection_reason": record.rejection_reason,
        "attempts": list(record.attempts or []),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _audit_to_dict(record) -> dict[str, Any]:
    return {
        "event_id": record.event_id,
        "action": record.action,
        "resource_type": record.resource_type,
        "resource_id": record.resource_id,
        "actor": record.actor,
        "actor_type": record.actor_type,
        "reason": record.reason,
        "before": record.before,
        "after": record.after,
        "timestamp": record.timestamp,
    }


def _approval_to_dict(record) -> dict[str, Any]:
    return {
        "approval_id": record.approval_id,
        "intent_id": record.intent_id,
        "plan_id": record.plan_id,
        "status": record.status,
        "requested_by": record.requested_by,
        "requested_by_type": record.requested_by_type,
        "decided_by": record.decided_by,
        "decided_by_type": record.decided_by_type,
        "reason": record.reason,
        "notes": record.notes,
        "created_at": record.created_at,
        "decided_at": record.decided_at,
    }


def _first_timestamp(snapshot: Mapping[str, Mapping[str, Any]], key: str) -> datetime | None:
    for entry in snapshot.values():
        value = entry.get(key)
        if value:
            try:
                return datetime.fromisoformat(str(value))
            except ValueError:
                return None
    return None
