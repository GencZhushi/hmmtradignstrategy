"""Single-writer execution coordinator (Phase A9).

Ensures the engine has exactly one writer path that can mutate broker state.
Responsibilities:

- Accept a ``TradeIntent``, deduplicate via the idempotency store, serialize
  per-symbol via the ``LockManager``.
- Run ``RiskManager.validate_signal`` before anything touches the broker.
- Create an ``OrderRecord`` in the state machine and hand it off to the
  ``OrderExecutor`` interface (any implementation - live Alpaca or sim).
- Broadcast lifecycle events so the API / dashboard / audit log stay in sync.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Protocol

from core.idempotency import IdempotencyStore, build_idempotency_key
from core.lock_manager import LockManager, LockUnavailable
from core.order_state_machine import OrderRecord, OrderStateMachine, OrderStatus
from core.risk_manager import RiskManager
from core.types import (
    AuditEvent,
    Direction,
    IntentStatus,
    OrderPlan,
    PortfolioState,
    RiskDecision,
    Signal,
    TradeIntent,
    new_plan_id,
    stable_idempotency_key,
)

LOG = logging.getLogger(__name__)


class OrderExecutorProtocol(Protocol):
    """Narrow surface the coordinator uses to talk to any executor implementation."""

    def submit_order(self, order: OrderRecord) -> str: ...
    def modify_stop(self, order_id: str, new_stop: float) -> None: ...
    def cancel_order(self, order_id: str) -> None: ...


EventListener = Callable[[str, Mapping[str, Any]], None]


@dataclass
class CoordinatorOutcome:
    """Summary returned to the API / agent / UI after submit_intent."""

    intent: TradeIntent
    plan: OrderPlan
    decision: RiskDecision
    order: OrderRecord | None = None
    duplicate: bool = False
    audit_event: AuditEvent | None = None


@dataclass
class ExecutionCoordinator:
    """The only component allowed to mutate broker state."""

    risk_manager: RiskManager
    state_machine: OrderStateMachine
    idempotency: IdempotencyStore
    lock_manager: LockManager
    executor: OrderExecutorProtocol | None = None
    portfolio_provider: Callable[[], PortfolioState] | None = None
    market_price_provider: Callable[[str], float] | None = None
    listeners: list[EventListener] = field(default_factory=list)
    _mutation_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    # ------------------------------------------------------------------ public API
    def preview_intent(self, intent: TradeIntent) -> CoordinatorOutcome:
        """Run the full risk pipeline without mutating broker state."""
        return self._process(intent, execute=False)

    def submit_intent(self, intent: TradeIntent) -> CoordinatorOutcome:
        """Preview + execute when the caller passes the approval gate."""
        return self._process(intent, execute=True)

    def register_listener(self, listener: EventListener) -> None:
        self.listeners.append(listener)

    def reconcile_after_fill(
        self,
        order_id: str,
        *,
        filled_qty: float,
        fill_price: float,
    ) -> OrderRecord:
        order = self.state_machine.handle_partial_fill(
            order_id, filled_qty=filled_qty, fill_price=fill_price
        )
        self._emit(
            "order_filled" if order.status == OrderStatus.FILLED else "order_partially_filled",
            {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "filled_qty": order.filled_qty,
                "avg_fill_price": order.avg_fill_price,
                "status": order.status.value,
            },
        )
        return order

    def reconcile_after_reconnect(
        self,
        *,
        broker_orders: Iterable[Mapping[str, Any]],
    ) -> list[OrderRecord]:
        """Sync local state with the broker's current view after a reconnect."""
        updated: list[OrderRecord] = []
        broker_index = {bo.get("broker_order_id"): bo for bo in broker_orders if bo.get("broker_order_id")}
        for order in list(self.state_machine.orders.values()):
            attempt = order.attempts[-1] if order.attempts else None
            if attempt is None or attempt.broker_order_id is None:
                continue
            broker_view = broker_index.get(attempt.broker_order_id)
            if broker_view is None:
                self.state_machine.mark_order_dead(order.order_id, reason="broker_reports_unknown")
                updated.append(order)
                continue
            status_str = str(broker_view.get("status", "")).lower()
            if status_str == "filled" and order.status not in (OrderStatus.FILLED,):
                order = self.state_machine.handle_partial_fill(
                    order.order_id,
                    filled_qty=float(broker_view.get("filled_qty", order.quantity)) - order.filled_qty,
                    fill_price=float(broker_view.get("avg_fill_price", order.avg_fill_price or 0.0)),
                )
            elif status_str in {"canceled", "cancelled"}:
                self.state_machine.advance_order_state(order.order_id, to=OrderStatus.CANCELLED, reason="broker_cancelled")
            updated.append(order)
        self._emit("reconciliation_completed", {"orders": [o.order_id for o in updated]})
        return updated

    # ------------------------------------------------------------------ internals
    def _process(self, intent: TradeIntent, *, execute: bool) -> CoordinatorOutcome:
        key = intent.idempotency_key or build_idempotency_key(
            intent.source, intent.symbol, intent.direction.value, intent.allocation_pct, intent.intent_type
        )
        intent.idempotency_key = key
        # Serialize on the per-symbol lock BEFORE checking idempotency so that
        # concurrent retries of the same intent observe the fully populated
        # result instead of racing against an in-flight pipeline.
        lock_key = f"symbol:{intent.symbol.upper()}"
        with self._mutation_lock:
            with self.lock_manager.guard(lock_key, owner=intent.intent_id, timeout=5.0):
                record = self.idempotency.register_intent(
                    key,
                    intent_id=intent.intent_id,
                    actor=intent.actor or intent.source,
                    resource_type="trade_intent",
                    payload={
                        "symbol": intent.symbol,
                        "direction": intent.direction.value,
                        "allocation_pct": intent.allocation_pct,
                        "intent_type": intent.intent_type,
                    },
                )
                if record.intent_id != intent.intent_id:
                    return self._duplicate_outcome(intent=intent, record=record)
                return self._run_pipeline(intent, execute=execute, key=key)

    def _duplicate_outcome(self, *, intent: TradeIntent, record) -> CoordinatorOutcome:
        existing = record.result or {}
        LOG.info("Duplicate intent %s -> returning prior outcome", intent.intent_id)
        # Rewrite the caller's intent_id to match the prior record so
        # downstream serializers expose the original intent identifier.
        intent.intent_id = record.intent_id
        intent.status = IntentStatus.EXECUTED if existing.get("status") == "executed" else IntentStatus.PREVIEWED
        return CoordinatorOutcome(
            intent=intent,
            plan=OrderPlan(
                intent_id=record.intent_id,
                plan_id=existing.get("plan_id", new_plan_id()),
                approved_signal=existing.get("approved", False),
                symbol=intent.symbol,
                direction=intent.direction,
                risk_adjusted_size=existing.get("risk_adjusted_size", 0.0),
                risk_adjusted_leverage=existing.get("risk_adjusted_leverage", 1.0),
                status=existing.get("status", "duplicate"),
                rejection_reason=existing.get("reason_message"),
                reason_codes=list(existing.get("reason_codes", [])),
            ),
            decision=RiskDecision(
                approved=existing.get("approved", False),
                modified=existing.get("modified", False),
                signal=None,
                reason_codes=list(existing.get("reason_codes", [])),
                reason_message=existing.get("reason_message", "duplicate intent"),
            ),
            duplicate=True,
        )

    def _run_pipeline(self, intent: TradeIntent, *, execute: bool, key: str) -> CoordinatorOutcome:
        portfolio = self.portfolio_provider() if self.portfolio_provider else _empty_portfolio()
        signal = self._signal_from_intent(intent)
        decision = self.risk_manager.validate_signal(signal, portfolio)
        plan = _plan_from_decision(intent=intent, decision=decision)
        audit_payload = {
            "approved": decision.approved,
            "modified": decision.modified,
            "reason_codes": decision.reason_codes,
            "reason_message": decision.reason_message,
            "risk_adjusted_size": plan.risk_adjusted_size,
            "risk_adjusted_leverage": plan.risk_adjusted_leverage,
            "plan_id": plan.plan_id,
            "status": plan.status,
        }
        self.idempotency.mark_status(
            key,
            status="previewed" if not execute or not decision.approved else "executed",
            result=audit_payload,
        )
        self._emit("intent_previewed" if not execute else "intent_submitted", {
            "intent_id": intent.intent_id,
            "plan_id": plan.plan_id,
            **audit_payload,
        })
        if not execute or not decision.approved:
            intent.status = IntentStatus.REJECTED if not decision.approved else IntentStatus.PREVIEWED
            return CoordinatorOutcome(intent=intent, plan=plan, decision=decision)

        order = self._create_order(intent, plan, decision.signal)  # type: ignore[arg-type]
        if self.executor is not None:
            try:
                broker_id = self.executor.submit_order(order)
                self.state_machine.register_attempt(order.order_id, attempt_id=f"att-{len(order.attempts)+1}", broker_order_id=broker_id)
                self.state_machine.advance_order_state(order.order_id, to=OrderStatus.ACCEPTED)
                plan.status = "submitted"
            except Exception as exc:  # pragma: no cover - external executor
                LOG.error("Executor rejected order %s: %s", order.order_id, exc)
                self.state_machine.advance_order_state(order.order_id, to=OrderStatus.FAILED, reason=str(exc))
                plan.status = "failed"
                intent.status = IntentStatus.FAILED
                return CoordinatorOutcome(intent=intent, plan=plan, decision=decision, order=order)

        intent.status = IntentStatus.EXECUTED
        return CoordinatorOutcome(intent=intent, plan=plan, decision=decision, order=order)

    def _signal_from_intent(self, intent: TradeIntent) -> Signal:
        price = self.market_price_provider(intent.symbol) if self.market_price_provider else 0.0
        if price <= 0:
            # Fallback for dry-run/tests when no market data has been seeded yet.
            # Real deployments always have a price via the position tracker.
            price = 100.0
        stop = None if intent.direction == Direction.FLAT else price * 0.97
        return Signal(
            symbol=intent.symbol,
            direction=intent.direction,
            target_allocation_pct=intent.allocation_pct,
            leverage=intent.requested_leverage,
            entry_price=price,
            stop_loss=stop,
            strategy_name=f"intent:{intent.source}",
            reasoning=[intent.thesis] if intent.thesis else [],
            metadata={"intent_id": intent.intent_id},
        )

    def _create_order(self, intent: TradeIntent, plan: OrderPlan, signal: Signal) -> OrderRecord:
        portfolio = self.portfolio_provider() if self.portfolio_provider else _empty_portfolio()
        price = float(signal.entry_price) if signal.entry_price and signal.entry_price > 0 else 100.0
        notional = float(plan.risk_adjusted_size) * float(plan.risk_adjusted_leverage) * max(portfolio.equity, 0.0)
        quantity = max(notional / price, 1.0) if notional > 0 else 1.0
        side = "BUY" if intent.direction == Direction.LONG else "SELL"
        order = self.state_machine.create_order(
            intent_id=intent.intent_id,
            symbol=intent.symbol,
            side=side,
            quantity=quantity,
            limit_price=price if signal.entry_price else None,
            stop_price=signal.stop_loss,
            take_profit=signal.take_profit,
            idempotency_key=intent.idempotency_key or "",
        )
        return order

    def _emit(self, event: str, payload: Mapping[str, Any]) -> None:
        for listener in self.listeners:
            try:
                listener(event, payload)
            except Exception as exc:  # pragma: no cover - listener isolation
                LOG.error("Listener error on %s: %s", event, exc)


def _plan_from_decision(*, intent: TradeIntent, decision: RiskDecision) -> OrderPlan:
    signal = decision.signal
    plan = OrderPlan(
        intent_id=intent.intent_id,
        approved_signal=decision.approved,
        symbol=intent.symbol,
        direction=intent.direction,
        risk_adjusted_size=float(signal.target_allocation_pct) if signal else 0.0,
        risk_adjusted_leverage=float(signal.leverage) if signal else intent.requested_leverage,
        entry_type="limit",
        limit_price=float(signal.entry_price) if signal and signal.entry_price else None,
        stop_loss=float(signal.stop_loss) if signal and signal.stop_loss else None,
        take_profit=float(signal.take_profit) if signal and signal.take_profit else None,
        status="approved" if decision.approved else "rejected",
        rejection_reason=None if decision.approved else decision.reason_message,
        reason_codes=list(decision.reason_codes),
        projected_exposure=decision.projected_exposure,
        projected_sector_exposure=dict(decision.projected_sector_exposure) if decision.projected_sector_exposure else None,
    )
    return plan


def _empty_portfolio() -> PortfolioState:
    return PortfolioState(equity=100_000.0, cash=100_000.0, buying_power=100_000.0)
