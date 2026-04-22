"""Pydantic request/response models for the API platform (Phase B1)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class APIResponse(BaseModel):
    """Base response helper."""

    model_config = ConfigDict(from_attributes=True)


class HealthResponse(APIResponse):
    status: Literal["ok", "degraded", "unavailable"] = "ok"
    trading_mode: str
    execution_enabled: bool
    dry_run: bool
    active_model: str | None = None
    broker_connected: bool = False
    last_reconciliation: datetime | None = None


class FreshnessPayload(APIResponse):
    exchange_timezone: str
    exchange_session_state: str
    last_completed_daily_bar_time: datetime | None = None
    last_completed_intraday_bar_time: datetime | None = None
    data_freshness_status: str
    daily_data_stale: bool = False
    intraday_data_stale: bool = False
    regime_effective_session_date: str | None = None
    stale_data_blocked: bool = False


class RegimeSnapshot(APIResponse):
    regime_id: int | None = None
    regime_name: str | None = None
    probability: float = 0.0
    state_probabilities: list[float] = Field(default_factory=list)
    is_confirmed: bool = False
    consecutive_bars: int = 0
    flicker_rate: float = 0.0
    effective_session_date: str | None = None
    active_model_version: str | None = None


class PositionSchema(APIResponse):
    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    stop_price: float | None = None
    regime_at_entry: str | None = None
    unrealized_pnl: float = 0.0
    market_value: float = 0.0
    sector_bucket: str | None = None
    etf_bucket: str | None = None


class PortfolioSchema(APIResponse):
    equity: float
    cash: float
    buying_power: float
    daily_pnl: float = 0.0
    weekly_pnl: float = 0.0
    peak_equity: float = 0.0
    drawdown: float = 0.0
    breaker_state: str = "clear"
    total_exposure_pct: float = 0.0
    positions: list[PositionSchema] = Field(default_factory=list)
    sector_exposure: dict[str, float] = Field(default_factory=dict)


class SignalSchema(APIResponse):
    symbol: str
    direction: str
    target_allocation_pct: float
    leverage: float
    entry_price: float
    stop_loss: float | None = None
    take_profit: float | None = None
    strategy_name: str = "unknown"
    regime_id: int | None = None
    regime_name: str | None = None
    regime_probability: float = 0.0
    confidence: float = 0.0
    reasoning: list[str] = Field(default_factory=list)
    timestamp: datetime | None = None


class TradeIntentRequest(BaseModel):
    symbol: str
    direction: Literal["LONG", "FLAT"]
    allocation_pct: float = Field(ge=0.0, le=1.5)
    requested_leverage: float = Field(default=1.0, ge=0.0, le=4.0)
    intent_type: str = "open_position"
    thesis: str = ""
    timeframe: str = "5m"
    requires_confirmation: bool = True
    idempotency_key: str | None = None
    source: str = "user"


class OrderPlanSchema(APIResponse):
    intent_id: str
    plan_id: str
    approved_signal: bool
    symbol: str
    direction: str
    risk_adjusted_size: float
    risk_adjusted_leverage: float
    entry_type: str
    limit_price: float | None
    stop_loss: float | None
    take_profit: float | None
    status: str
    rejection_reason: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    projected_exposure: float | None = None
    projected_sector_exposure: dict[str, float] | None = None
    sector_bucket: str | None = None
    etf_bucket: str | None = None


class IntentSchema(APIResponse):
    intent_id: str
    idempotency_key: str
    symbol: str
    direction: str
    allocation_pct: float
    requested_leverage: float
    intent_type: str
    thesis: str
    status: str
    actor: str
    actor_type: str
    plan_id: str | None = None
    created_at: datetime
    updated_at: datetime


class ApprovalSchema(APIResponse):
    approval_id: str
    intent_id: str
    plan_id: str | None = None
    status: str
    requested_by: str
    requested_by_type: str
    decided_by: str | None = None
    decided_by_type: str | None = None
    reason: str = ""
    notes: str = ""
    created_at: datetime
    decided_at: datetime | None = None


class ApprovalDecisionRequest(BaseModel):
    approval_id: str
    reason: str = ""
    notes: str = ""


class OrderSchema(APIResponse):
    order_id: str
    trade_id: str
    intent_id: str
    symbol: str
    side: str
    quantity: float
    filled_qty: float
    avg_fill_price: float | None
    status: str
    limit_price: float | None
    stop_price: float | None
    take_profit: float | None
    protective_stop_status: str
    rejection_reason: str | None = None
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class AuditEventSchema(APIResponse):
    event_id: str
    action: str
    resource_type: str
    resource_id: str
    actor: str
    actor_type: str
    reason: str
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None
    timestamp: datetime


class ModelGovernanceSchema(APIResponse):
    active_model_version: str | None = None
    fallback_model_version: str | None = None
    active_training_metadata: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    last_promotion_decision: dict[str, Any] | None = None


class ConfigReloadRequest(BaseModel):
    reason: str = ""


class ArmLiveRequest(BaseModel):
    ttl_minutes: int = Field(default=15, ge=1, le=240)
    reason: str = ""


class ConcentrationSchema(APIResponse):
    sector_exposure: dict[str, float]
    projected_post_trade_exposure: dict[str, float] = Field(default_factory=dict)
    correlation_metrics: dict[str, float] = Field(default_factory=dict)
    blocked_reasons: list[str] = Field(default_factory=list)


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    role: str
    username: str
