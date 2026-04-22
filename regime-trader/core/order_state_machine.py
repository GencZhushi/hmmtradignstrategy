"""Explicit order lifecycle state machine (Phase A10).

Every transition is validated against a whitelist. Partial fills, bracket
desync repairs, stop-order failures, and retry attempts are first-class events
so the audit log always carries enough context to reconstruct history.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Mapping

from core.types import OrderStatus, new_order_id, new_trade_id

LOG = logging.getLogger(__name__)


VALID_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.NEW: frozenset({OrderStatus.SUBMITTED, OrderStatus.REJECTED, OrderStatus.DEAD, OrderStatus.FAILED}),
    OrderStatus.SUBMITTED: frozenset({OrderStatus.ACCEPTED, OrderStatus.REJECTED, OrderStatus.CANCELLED, OrderStatus.FAILED, OrderStatus.DEAD}),
    OrderStatus.ACCEPTED: frozenset({OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.FAILED, OrderStatus.DEAD}),
    OrderStatus.PARTIALLY_FILLED: frozenset({OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.DEAD}),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
    OrderStatus.DEAD: frozenset(),
    OrderStatus.FAILED: frozenset({OrderStatus.DEAD}),
}


class InvalidTransition(RuntimeError):
    """Raised when a caller requests an impossible state change."""


@dataclass
class OrderAttempt:
    """One broker submission attempt for an order (used during retries)."""

    attempt_id: str
    submitted_at: datetime
    broker_order_id: str | None = None
    status: OrderStatus = OrderStatus.NEW
    error: str | None = None
    filled_qty: float = 0.0
    avg_fill_price: float | None = None


@dataclass
class OrderRecord:
    """Persistent view of an order and its lifecycle history."""

    order_id: str
    trade_id: str
    intent_id: str
    symbol: str
    side: str  # "BUY" or "SELL"
    quantity: float
    limit_price: float | None
    stop_price: float | None
    take_profit: float | None
    idempotency_key: str
    status: OrderStatus = OrderStatus.NEW
    filled_qty: float = 0.0
    avg_fill_price: float | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rejection_reason: str | None = None
    attempts: list[OrderAttempt] = field(default_factory=list)
    bracket_child_ids: list[str] = field(default_factory=list)
    protective_stop_status: str = "pending"   # pending | active | missing | failed

    def remaining_qty(self) -> float:
        return max(self.quantity - self.filled_qty, 0.0)


@dataclass
class OrderStateMachine:
    """Enforces legal transitions and records lifecycle events."""

    orders: dict[str, OrderRecord] = field(default_factory=dict)

    def create_order(
        self,
        *,
        intent_id: str,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float | None,
        stop_price: float | None,
        take_profit: float | None,
        idempotency_key: str,
        trade_id: str | None = None,
    ) -> OrderRecord:
        order = OrderRecord(
            order_id=new_order_id(),
            trade_id=trade_id or new_trade_id(),
            intent_id=intent_id,
            symbol=symbol,
            side=side.upper(),
            quantity=quantity,
            limit_price=limit_price,
            stop_price=stop_price,
            take_profit=take_profit,
            idempotency_key=idempotency_key,
        )
        self.orders[order.order_id] = order
        return order

    def advance_order_state(
        self,
        order_id: str,
        *,
        to: OrderStatus,
        reason: str | None = None,
    ) -> OrderRecord:
        order = self._get(order_id)
        allowed = VALID_TRANSITIONS.get(order.status, frozenset())
        if to not in allowed:
            raise InvalidTransition(
                f"Illegal transition {order.status.value} -> {to.value} for {order_id}"
            )
        LOG.debug("Order %s transition %s -> %s (%s)", order_id, order.status.value, to.value, reason)
        order.status = to
        order.updated_at = datetime.now(timezone.utc)
        if reason:
            order.rejection_reason = reason
        return order

    def register_attempt(
        self,
        order_id: str,
        *,
        attempt_id: str,
        broker_order_id: str | None,
    ) -> OrderAttempt:
        order = self._get(order_id)
        attempt = OrderAttempt(
            attempt_id=attempt_id,
            submitted_at=datetime.now(timezone.utc),
            broker_order_id=broker_order_id,
            status=OrderStatus.SUBMITTED,
        )
        order.attempts.append(attempt)
        if order.status == OrderStatus.NEW:
            self.advance_order_state(order_id, to=OrderStatus.SUBMITTED)
        return attempt

    def handle_partial_fill(
        self,
        order_id: str,
        *,
        filled_qty: float,
        fill_price: float,
    ) -> OrderRecord:
        order = self._get(order_id)
        if filled_qty <= 0:
            raise ValueError("filled_qty must be positive")
        total_filled = order.filled_qty + filled_qty
        if total_filled > order.quantity + 1e-9:
            raise ValueError("partial fill exceeds order quantity")
        if order.avg_fill_price is None:
            order.avg_fill_price = fill_price
        else:
            order.avg_fill_price = (
                (order.avg_fill_price * order.filled_qty + fill_price * filled_qty)
                / total_filled
            )
        order.filled_qty = total_filled
        order.updated_at = datetime.now(timezone.utc)
        next_state = OrderStatus.FILLED if order.filled_qty >= order.quantity - 1e-9 else OrderStatus.PARTIALLY_FILLED
        if order.status != next_state:
            if next_state == OrderStatus.PARTIALLY_FILLED and order.status == OrderStatus.ACCEPTED:
                self.advance_order_state(order_id, to=OrderStatus.PARTIALLY_FILLED)
            elif next_state == OrderStatus.FILLED:
                self.advance_order_state(order_id, to=OrderStatus.FILLED)
        return order

    def update_trailing_stop_after_partial_exit(
        self,
        order_id: str,
        *,
        new_stop: float,
    ) -> OrderRecord:
        order = self._get(order_id)
        if order.stop_price is None:
            order.stop_price = new_stop
        else:
            order.stop_price = max(order.stop_price, new_stop)
        order.updated_at = datetime.now(timezone.utc)
        return order

    def handle_stop_failure(self, order_id: str, *, reason: str) -> OrderRecord:
        order = self._get(order_id)
        order.protective_stop_status = "failed"
        order.rejection_reason = reason
        order.updated_at = datetime.now(timezone.utc)
        LOG.warning("Stop failure recorded for %s: %s", order_id, reason)
        return order

    def handle_bracket_desync(
        self,
        order_id: str,
        *,
        missing_child_ids: Iterable[str],
    ) -> OrderRecord:
        order = self._get(order_id)
        order.protective_stop_status = "missing"
        order.rejection_reason = f"bracket_desync:{','.join(missing_child_ids)}"
        order.updated_at = datetime.now(timezone.utc)
        LOG.warning("Bracket desync for %s: missing=%s", order_id, list(missing_child_ids))
        return order

    def mark_order_dead(self, order_id: str, *, reason: str) -> OrderRecord:
        order = self._get(order_id)
        order.rejection_reason = reason
        order.updated_at = datetime.now(timezone.utc)
        if order.status not in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.DEAD):
            try:
                self.advance_order_state(order_id, to=OrderStatus.DEAD, reason=reason)
            except InvalidTransition:
                order.status = OrderStatus.DEAD
        return order

    def attempts_for_trade(self, trade_id: str) -> list[OrderAttempt]:
        return [attempt for order in self.orders.values() if order.trade_id == trade_id for attempt in order.attempts]

    def summary(self, order_id: str) -> Mapping[str, object]:
        order = self._get(order_id)
        return {
            "order_id": order.order_id,
            "trade_id": order.trade_id,
            "intent_id": order.intent_id,
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "limit_price": order.limit_price,
            "stop_price": order.stop_price,
            "take_profit": order.take_profit,
            "status": order.status.value,
            "filled_qty": order.filled_qty,
            "remaining_qty": order.remaining_qty(),
            "avg_fill_price": order.avg_fill_price,
            "attempts": [
                {
                    "attempt_id": a.attempt_id,
                    "broker_order_id": a.broker_order_id,
                    "status": a.status.value if hasattr(a.status, "value") else str(a.status),
                    "submitted_at": a.submitted_at.isoformat() if a.submitted_at else None,
                    "error": a.error,
                    "filled_qty": a.filled_qty,
                    "avg_fill_price": a.avg_fill_price,
                }
                for a in order.attempts
            ],
            "protective_stop_status": order.protective_stop_status,
            "rejection_reason": order.rejection_reason,
        }

    def _get(self, order_id: str) -> OrderRecord:
        if order_id not in self.orders:
            raise KeyError(order_id)
        return self.orders[order_id]
