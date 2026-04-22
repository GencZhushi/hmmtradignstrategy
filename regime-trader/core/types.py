"""Shared dataclasses used across the core engine, API, and agent layers.

Kept dependency-free so every subsystem (engine, API, OpenClaw) can import these
without circular imports.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Direction(str, Enum):
    LONG = "LONG"
    FLAT = "FLAT"


class IntentStatus(str, Enum):
    PENDING = "pending"
    PREVIEWED = "previewed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class OrderStatus(str, Enum):
    NEW = "new"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    DEAD = "dead"
    FAILED = "failed"


class BreakerState(str, Enum):
    CLEAR = "clear"
    DAILY_REDUCE = "daily_reduce"
    DAILY_HALT = "daily_halt"
    WEEKLY_REDUCE = "weekly_reduce"
    WEEKLY_HALT = "weekly_halt"
    PEAK_HALT = "peak_halt"


@dataclass
class Position:
    """Single-symbol portfolio position snapshot."""

    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    stop_price: float | None = None
    regime_at_entry: str | None = None
    entered_at: datetime = field(default_factory=_utcnow)

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_entry_price) * self.quantity


@dataclass
class PortfolioState:
    """Mutable view of portfolio + risk metrics shared with the risk manager."""

    equity: float
    cash: float
    buying_power: float
    positions: dict[str, Position] = field(default_factory=dict)
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_equity: float = 0.0
    drawdown: float = 0.0
    breaker_state: BreakerState = BreakerState.CLEAR
    flicker_rate: float = 0.0
    daily_trade_count: int = 0
    blocked_symbols: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.peak_equity <= 0:
            self.peak_equity = self.equity

    @property
    def total_exposure_pct(self) -> float:
        if self.equity <= 0:
            return 0.0
        return sum(abs(p.market_value) for p in self.positions.values()) / self.equity

    def sector_exposure(self, sector_of: "callable[[str], str]") -> dict[str, float]:
        buckets: dict[str, float] = {}
        if self.equity <= 0:
            return buckets
        for sym, pos in self.positions.items():
            bucket = sector_of(sym)
            buckets[bucket] = buckets.get(bucket, 0.0) + abs(pos.market_value) / self.equity
        return buckets


@dataclass
class Signal:
    """Strategy output. May be modified by the risk manager before execution."""

    symbol: str
    direction: Direction
    target_allocation_pct: float
    leverage: float
    entry_price: float
    stop_loss: float | None
    take_profit: float | None = None
    regime_id: int | None = None
    regime_name: str | None = None
    regime_probability: float = 0.0
    confidence: float = 0.0
    strategy_name: str = "unknown"
    reasoning: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_utcnow)

    def with_modifications(self, **changes: Any) -> "Signal":
        import dataclasses as _dc

        return _dc.replace(self, **changes)


@dataclass
class RiskDecision:
    """Result of the risk manager's validation of a signal."""

    approved: bool
    modified: bool
    signal: Signal | None
    reason_codes: list[str] = field(default_factory=list)
    reason_message: str = ""
    projected_exposure: float | None = None
    projected_sector_exposure: dict[str, float] | None = None
    projected_leverage: float | None = None
    breaker_state: BreakerState = BreakerState.CLEAR
    scaled_allocation_pct: float | None = None


@dataclass
class TradeIntent:
    """Structured intent used by the API platform and OpenClaw (Spec C.C1)."""

    symbol: str
    direction: Direction
    allocation_pct: float
    requested_leverage: float = 1.0
    intent_type: str = "open_position"
    intent_id: str = field(default_factory=lambda: _new_id("intent"))
    idempotency_key: str | None = None
    source: str = "user"
    actor: str | None = None
    thesis: str = ""
    timeframe: str = "5m"
    requires_confirmation: bool = True
    created_at: datetime = field(default_factory=_utcnow)
    status: IntentStatus = IntentStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderPlan:
    """Engine-generated preview of an order that will be submitted on approval."""

    intent_id: str
    plan_id: str = field(default_factory=lambda: _new_id("plan"))
    approved_signal: bool = False
    symbol: str = ""
    direction: Direction = Direction.FLAT
    risk_adjusted_size: float = 0.0
    risk_adjusted_leverage: float = 1.0
    entry_type: str = "limit"
    limit_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    status: str = "previewed"
    rejection_reason: str | None = None
    reason_codes: list[str] = field(default_factory=list)
    projected_exposure: float | None = None
    projected_sector_exposure: dict[str, float] | None = None
    sector_bucket: str | None = None
    etf_bucket: str | None = None
    created_at: datetime = field(default_factory=_utcnow)


@dataclass
class AgentActionResult:
    """OpenClaw-facing response wrapper (Spec C.C3)."""

    ok: bool
    action: str
    resource_id: str | None = None
    status: str | None = None
    message: str = ""
    requires_human_approval: bool = False
    data: dict[str, Any] = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)


@dataclass
class AuditEvent:
    """Spec B.B7 audit record."""

    action: str
    resource_type: str
    resource_id: str
    actor: str
    actor_type: str
    reason: str = ""
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    timestamp: datetime = field(default_factory=_utcnow)
    event_id: str = field(default_factory=lambda: _new_id("evt"))


def new_intent_id() -> str:
    return _new_id("intent")


def new_plan_id() -> str:
    return _new_id("plan")


def new_order_id() -> str:
    return _new_id("ord")


def new_trade_id() -> str:
    return _new_id("trd")


def stable_idempotency_key(*parts: Any) -> str:
    """Deterministic key helper used by API/agent layers when the caller omits one."""
    import hashlib

    joined = "|".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()
