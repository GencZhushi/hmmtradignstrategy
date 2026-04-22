"""Portfolio-level risk manager with absolute veto power (Phase A6 + A12).

The risk manager is independent of the HMM. Even if the model produces a
pathological signal, these rules must refuse or modify it before it reaches the
broker. Every rejection/modification is logged with a structured reason code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, Mapping

import pandas as pd

from core.correlation_risk import (
    CorrelationBreach,
    check_correlation_limit,
    compute_rolling_return_correlation,
)
from core.sector_mapping import SectorClassifier
from core.types import (
    BreakerState,
    Direction,
    PortfolioState,
    Position,
    RiskDecision,
    Signal,
)

LOG = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Typed view over the ``risk:`` block of ``settings.yaml``."""

    max_risk_per_trade: float = 0.01
    max_exposure: float = 0.80
    max_leverage: float = 1.25
    max_single_position: float = 0.15
    max_concurrent: int = 5
    max_daily_trades: int = 20
    daily_dd_reduce: float = 0.02
    daily_dd_halt: float = 0.03
    weekly_dd_reduce: float = 0.05
    weekly_dd_halt: float = 0.07
    max_dd_from_peak: float = 0.10
    max_sector_exposure: float = 0.30
    correlation_reduce_threshold: float = 0.70
    correlation_reject_threshold: float = 0.85
    correlation_lookback_days: int = 60

    @classmethod
    def from_config(cls, risk: Mapping[str, float | int]) -> "RiskLimits":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in risk.items() if k in fields}
        return cls(**kwargs)


@dataclass
class CircuitBreaker:
    """Drawdown-aware breaker that operates on actual P&L independently of the HMM."""

    limits: RiskLimits
    halted_until_manual_reset: bool = False
    last_reset_day: datetime | None = None
    last_reset_week: datetime | None = None
    history: list[dict] = field(default_factory=list)

    def evaluate(self, portfolio: PortfolioState) -> BreakerState:
        if self.halted_until_manual_reset:
            state = BreakerState.PEAK_HALT
        elif portfolio.drawdown >= self.limits.max_dd_from_peak:
            self.halted_until_manual_reset = True
            state = BreakerState.PEAK_HALT
        elif portfolio.equity > 0 and abs(portfolio.weekly_pnl) / portfolio.peak_equity >= self.limits.weekly_dd_halt:
            state = BreakerState.WEEKLY_HALT
        elif portfolio.equity > 0 and abs(portfolio.weekly_pnl) / portfolio.peak_equity >= self.limits.weekly_dd_reduce:
            state = BreakerState.WEEKLY_REDUCE
        elif portfolio.equity > 0 and abs(portfolio.daily_pnl) / portfolio.peak_equity >= self.limits.daily_dd_halt:
            state = BreakerState.DAILY_HALT
        elif portfolio.equity > 0 and abs(portfolio.daily_pnl) / portfolio.peak_equity >= self.limits.daily_dd_reduce:
            state = BreakerState.DAILY_REDUCE
        else:
            state = BreakerState.CLEAR
        portfolio.breaker_state = state
        if state != BreakerState.CLEAR:
            self.history.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "state": state.value,
                    "equity": portfolio.equity,
                    "daily_pnl": portfolio.daily_pnl,
                    "weekly_pnl": portfolio.weekly_pnl,
                    "drawdown": portfolio.drawdown,
                }
            )
        return state

    def enforce_drawdown_rules(self, portfolio: PortfolioState, signal: Signal) -> tuple[Signal | None, list[str]]:
        reasons: list[str] = []
        state = self.evaluate(portfolio)
        if state in (BreakerState.DAILY_HALT, BreakerState.WEEKLY_HALT, BreakerState.PEAK_HALT):
            reasons.append(f"breaker_halt:{state.value}")
            return None, reasons
        if state in (BreakerState.DAILY_REDUCE, BreakerState.WEEKLY_REDUCE):
            reasons.append(f"breaker_reduce:{state.value}")
            return (
                signal.with_modifications(
                    target_allocation_pct=signal.target_allocation_pct * 0.5,
                    leverage=min(signal.leverage, 1.0),
                ),
                reasons,
            )
        return signal, reasons

    def reset_daily(self) -> None:
        self.last_reset_day = datetime.now(timezone.utc)

    def reset_weekly(self) -> None:
        self.last_reset_week = datetime.now(timezone.utc)

    def manual_clear(self) -> None:
        self.halted_until_manual_reset = False


@dataclass
class RiskManager:
    """Absolute veto layer over every signal before it touches the broker."""

    limits: RiskLimits
    sector_classifier: SectorClassifier = field(default_factory=SectorClassifier)
    breaker: CircuitBreaker = field(init=False)
    returns_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    uncertainty_mode: bool = False
    _recent_submissions: dict[str, datetime] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.breaker = CircuitBreaker(limits=self.limits)

    # ---------------------------------------------------------------- validation
    def validate_signal(self, signal: Signal, portfolio: PortfolioState) -> RiskDecision:
        reasons: list[str] = []
        modified = False

        if signal.target_allocation_pct <= 0 and signal.direction == Direction.FLAT:
            return RiskDecision(
                approved=True,
                modified=False,
                signal=signal,
                reason_codes=["close_position"],
                reason_message="Signal requests exit - pass-through to executor",
                projected_exposure=portfolio.total_exposure_pct,
                breaker_state=portfolio.breaker_state,
            )

        # Pre-flight blockers (missing stops, duplicates, breaker halt).
        if signal.stop_loss is None or signal.stop_loss <= 0:
            return self._reject(signal, portfolio, "missing_stop", "Every long order must carry a stop-loss.")
        if signal.symbol in portfolio.blocked_symbols:
            return self._reject(signal, portfolio, "symbol_blocked", f"{signal.symbol} is on the blocked list")
        if self._is_duplicate(signal):
            return self._reject(signal, portfolio, "duplicate_request", "Repeat submission within cooldown window")
        if portfolio.daily_trade_count >= self.limits.max_daily_trades:
            return self._reject(signal, portfolio, "daily_trade_cap", "Max daily trades reached")
        if len(portfolio.positions) >= self.limits.max_concurrent and signal.symbol not in portfolio.positions:
            return self._reject(signal, portfolio, "max_concurrent_positions", "Max concurrent positions reached")

        # Breaker-driven halt/reduce logic.
        adjusted, breaker_reasons = self.breaker.enforce_drawdown_rules(portfolio, signal)
        if adjusted is None:
            return self._reject(signal, portfolio, ",".join(breaker_reasons), "Circuit breaker halted new entries")
        if breaker_reasons:
            modified = True
            reasons.extend(breaker_reasons)
            signal = adjusted

        if self.uncertainty_mode:
            signal = signal.with_modifications(
                target_allocation_pct=signal.target_allocation_pct * 0.5,
                leverage=min(signal.leverage, 1.0),
            )
            reasons.append("uncertainty_mode:size_halved")
            modified = True

        # Position/exposure caps.
        allocation_cap = min(self.limits.max_single_position, signal.target_allocation_pct)
        if signal.target_allocation_pct > self.limits.max_single_position:
            modified = True
            reasons.append("position_cap:max_single_position")
        signal = signal.with_modifications(target_allocation_pct=allocation_cap)

        projected = self._project_exposure(portfolio, signal)
        if projected["total_exposure"] > self.limits.max_exposure:
            overshoot = projected["total_exposure"] - self.limits.max_exposure
            scaled = max(signal.target_allocation_pct - overshoot, 0.0)
            if scaled <= 0:
                return self._reject(signal, portfolio, "exposure_cap", "No exposure headroom left")
            signal = signal.with_modifications(target_allocation_pct=scaled)
            modified = True
            reasons.append("exposure_cap:scaled")
            projected = self._project_exposure(portfolio, signal)
        if projected["leverage"] > self.limits.max_leverage:
            if signal.leverage > 1.0:
                signal = signal.with_modifications(leverage=1.0)
                modified = True
                reasons.append("leverage_cap:forced_1x")
                projected = self._project_exposure(portfolio, signal)
            else:
                return self._reject(signal, portfolio, "leverage_cap", "Projected leverage exceeds limit")

        # Sector concentration (projected post-trade).
        sector_decision, sector_projected, sector_reasons = self._apply_sector_limit(signal, portfolio)
        if sector_decision is None:
            return self._reject(
                signal,
                portfolio,
                ",".join(sector_reasons) or "sector_cap",
                "Sector concentration would exceed limit",
                projected_exposure=projected["total_exposure"],
                projected_sector_exposure=sector_projected,
            )
        if sector_decision != signal:
            modified = True
            signal = sector_decision
            reasons.extend(sector_reasons)

        # Correlation checks against open positions.
        breaches = self._correlation_breaches(signal, portfolio)
        if any(b.scope == "reject" for b in breaches):
            return self._reject(
                signal,
                portfolio,
                ";".join(b.as_reason_code() for b in breaches),
                "Correlation exceeds hard reject threshold",
            )
        if breaches:
            signal = signal.with_modifications(target_allocation_pct=signal.target_allocation_pct * 0.5)
            modified = True
            reasons.extend(b.as_reason_code() for b in breaches)

        self._recent_submissions[signal.symbol] = datetime.now(timezone.utc)

        return RiskDecision(
            approved=True,
            modified=modified,
            signal=signal,
            reason_codes=reasons,
            reason_message=", ".join(reasons) or "approved",
            projected_exposure=projected["total_exposure"],
            projected_sector_exposure=sector_projected,
            projected_leverage=projected["leverage"],
            breaker_state=portfolio.breaker_state,
            scaled_allocation_pct=signal.target_allocation_pct if modified else None,
        )

    # ---------------------------------------------------------------- helpers
    def _is_duplicate(self, signal: Signal, window_seconds: int = 60) -> bool:
        last = self._recent_submissions.get(signal.symbol)
        if last is None:
            return False
        return (datetime.now(timezone.utc) - last) < timedelta(seconds=window_seconds)

    def _project_exposure(self, portfolio: PortfolioState, signal: Signal) -> dict[str, float]:
        if portfolio.equity <= 0:
            return {"total_exposure": 0.0, "leverage": 0.0}
        current = {sym: abs(pos.market_value) / portfolio.equity for sym, pos in portfolio.positions.items()}
        current[signal.symbol] = signal.target_allocation_pct * signal.leverage
        total = sum(current.values())
        return {"total_exposure": total, "leverage": max(signal.leverage, total)}

    def _apply_sector_limit(
        self,
        signal: Signal,
        portfolio: PortfolioState,
    ) -> tuple[Signal | None, dict[str, float], list[str]]:
        sector_of = self.sector_classifier.get_sector_bucket
        current = portfolio.sector_exposure(sector_of)
        bucket = sector_of(signal.symbol)
        projected = dict(current)
        projected[bucket] = projected.get(bucket, 0.0) + signal.target_allocation_pct * signal.leverage
        if projected[bucket] <= self.limits.max_sector_exposure:
            return signal, projected, []

        overshoot = projected[bucket] - self.limits.max_sector_exposure
        if overshoot >= signal.target_allocation_pct * signal.leverage:
            return None, projected, ["sector_cap:hard_reject"]
        scaled = signal.target_allocation_pct - overshoot / max(signal.leverage, 1e-9)
        scaled = max(scaled, 0.0)
        projected[bucket] = current.get(bucket, 0.0) + scaled * signal.leverage
        # Floating-point subtraction can leave a residual `scaled` that is
        # numerically positive but effectively zero (e.g. ~1e-17). Treat
        # anything below a basis-point as no headroom so the sector cap is a
        # hard reject instead of an approved-but-invisible allocation.
        if scaled <= 1e-6:
            return None, projected, ["sector_cap:no_headroom"]
        scaled_signal = signal.with_modifications(target_allocation_pct=scaled)
        return scaled_signal, projected, ["sector_cap:scaled"]

    def _correlation_breaches(self, signal: Signal, portfolio: PortfolioState) -> list[CorrelationBreach]:
        if self.returns_history.empty or signal.symbol not in self.returns_history.columns:
            return []
        corr = compute_rolling_return_correlation(
            self.returns_history,
            lookback=self.limits.correlation_lookback_days,
        )
        return check_correlation_limit(
            corr,
            candidate=signal.symbol,
            open_symbols=portfolio.positions.keys(),
            reduce_threshold=self.limits.correlation_reduce_threshold,
            reject_threshold=self.limits.correlation_reject_threshold,
        )

    def _reject(
        self,
        signal: Signal,
        portfolio: PortfolioState,
        code: str,
        message: str,
        *,
        projected_exposure: float | None = None,
        projected_sector_exposure: Mapping[str, float] | None = None,
    ) -> RiskDecision:
        LOG.info("Rejected %s: %s", signal.symbol, message)
        return RiskDecision(
            approved=False,
            modified=False,
            signal=None,
            reason_codes=[code],
            reason_message=message,
            projected_exposure=projected_exposure,
            projected_sector_exposure=dict(projected_sector_exposure) if projected_sector_exposure else None,
            breaker_state=portfolio.breaker_state,
        )

    # ---------------------------------------------------------------- helpers for callers
    def compute_stop_levels(
        self,
        *,
        entry_price: float,
        atr: float,
        regime_rank: str,
    ) -> float:
        if regime_rank == "low_vol":
            return entry_price - 3.0 * atr
        if regime_rank == "high_vol":
            return entry_price - 1.0 * atr
        return entry_price - 1.5 * atr

    def check_position_limits(self, portfolio: PortfolioState) -> dict[str, bool]:
        return {
            "max_exposure_ok": portfolio.total_exposure_pct <= self.limits.max_exposure,
            "max_concurrent_ok": len(portfolio.positions) <= self.limits.max_concurrent,
        }

    def update_returns_history(self, returns: pd.DataFrame) -> None:
        self.returns_history = returns
