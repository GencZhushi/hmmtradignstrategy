"""Phase A6 + A12 - risk manager, circuit breaker, concentration + correlation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.correlation_risk import (
    check_correlation_limit,
    compute_rolling_return_correlation,
    resolve_joint_breach,
)
from core.risk_manager import CircuitBreaker, RiskLimits, RiskManager
from core.sector_mapping import SectorClassifier
from core.types import BreakerState, Direction, PortfolioState, Position, Signal


def _base_state(**kwargs) -> PortfolioState:
    return PortfolioState(
        equity=kwargs.get("equity", 100_000.0),
        cash=kwargs.get("cash", 100_000.0),
        buying_power=kwargs.get("buying_power", 100_000.0),
        positions={},
        daily_pnl=kwargs.get("daily_pnl", 0.0),
        weekly_pnl=kwargs.get("weekly_pnl", 0.0),
        peak_equity=kwargs.get("peak_equity", 100_000.0),
        drawdown=kwargs.get("drawdown", 0.0),
    )


def _signal(symbol: str = "SPY", *, allocation: float = 0.1, leverage: float = 1.0, stop: float | None = 95.0) -> Signal:
    return Signal(
        symbol=symbol,
        direction=Direction.LONG,
        target_allocation_pct=allocation,
        leverage=leverage,
        entry_price=100.0,
        stop_loss=stop,
        strategy_name="test",
    )


def test_missing_stop_rejected() -> None:
    rm = RiskManager(limits=RiskLimits())
    decision = rm.validate_signal(_signal(stop=None), _base_state())
    assert decision.approved is False
    assert "missing_stop" in decision.reason_codes


def test_daily_trade_cap_rejected() -> None:
    rm = RiskManager(limits=RiskLimits(max_daily_trades=1))
    portfolio = _base_state()
    portfolio.daily_trade_count = 1
    decision = rm.validate_signal(_signal(), portfolio)
    assert decision.approved is False
    assert "daily_trade_cap" in decision.reason_codes


def test_exposure_cap_scales_allocation() -> None:
    rm = RiskManager(limits=RiskLimits(max_exposure=0.3))
    portfolio = _base_state()
    portfolio.positions["QQQ"] = Position(symbol="QQQ", quantity=100, avg_entry_price=200.0, current_price=200.0, stop_price=190.0)
    decision = rm.validate_signal(_signal(allocation=0.2, leverage=1.0), portfolio)
    assert decision.approved is True
    assert decision.modified is True
    assert any("exposure_cap" in code for code in decision.reason_codes)
    assert decision.signal.target_allocation_pct <= 0.3


def test_single_position_cap_applied() -> None:
    rm = RiskManager(limits=RiskLimits(max_single_position=0.1))
    decision = rm.validate_signal(_signal(allocation=0.5), _base_state())
    assert decision.approved is True
    assert decision.signal.target_allocation_pct <= 0.1
    assert decision.modified is True


def test_uncertainty_mode_halves_size() -> None:
    rm = RiskManager(limits=RiskLimits())
    rm.uncertainty_mode = True
    decision = rm.validate_signal(_signal(allocation=0.15, leverage=1.25), _base_state())
    assert decision.approved is True
    assert decision.signal.leverage == pytest.approx(1.0)
    assert decision.signal.target_allocation_pct < 0.15


def test_daily_drawdown_reduces_then_halts() -> None:
    breaker = CircuitBreaker(limits=RiskLimits(daily_dd_reduce=0.02, daily_dd_halt=0.03))
    portfolio = _base_state(daily_pnl=-2500.0)
    state = breaker.evaluate(portfolio)
    assert state == BreakerState.DAILY_REDUCE
    portfolio.daily_pnl = -3500.0
    assert breaker.evaluate(portfolio) == BreakerState.DAILY_HALT


def test_peak_drawdown_latches_until_manual_clear() -> None:
    breaker = CircuitBreaker(limits=RiskLimits(max_dd_from_peak=0.10))
    portfolio = _base_state(drawdown=0.12)
    assert breaker.evaluate(portfolio) == BreakerState.PEAK_HALT
    portfolio.drawdown = 0.0
    assert breaker.evaluate(portfolio) == BreakerState.PEAK_HALT
    breaker.manual_clear()
    assert breaker.evaluate(portfolio) == BreakerState.CLEAR


def test_duplicate_submission_blocked_within_cooldown() -> None:
    rm = RiskManager(limits=RiskLimits())
    portfolio = _base_state()
    assert rm.validate_signal(_signal(), portfolio).approved
    second = rm.validate_signal(_signal(), portfolio)
    assert second.approved is False
    assert "duplicate_request" in second.reason_codes


def test_sector_limit_scales_then_rejects() -> None:
    classifier = SectorClassifier(sectors={"AAPL": "Technology", "MSFT": "Technology"})
    rm = RiskManager(
        limits=RiskLimits(max_sector_exposure=0.10, max_single_position=0.9, max_exposure=0.9),
        sector_classifier=classifier,
    )
    portfolio = _base_state()
    portfolio.positions["AAPL"] = Position(symbol="AAPL", quantity=100, avg_entry_price=100.0, current_price=100.0, stop_price=90.0)
    decision = rm.validate_signal(_signal(symbol="MSFT", allocation=0.08), portfolio)
    assert decision.approved is False
    assert any("sector_cap" in code for code in decision.reason_codes)


def test_correlation_reject_threshold_blocks_trade() -> None:
    rm = RiskManager(limits=RiskLimits(correlation_reject_threshold=0.80))
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, 120)
    returns = pd.DataFrame(
        {
            "AAPL": base,
            "MSFT": base * 0.98 + rng.normal(0, 0.001, 120),
        },
        index=pd.bdate_range("2023-01-02", periods=120),
    )
    rm.update_returns_history(returns)
    portfolio = _base_state()
    portfolio.positions["AAPL"] = Position(symbol="AAPL", quantity=10, avg_entry_price=180.0, current_price=180.0, stop_price=170.0)
    decision = rm.validate_signal(_signal(symbol="MSFT", allocation=0.1), portfolio)
    assert decision.approved is False
    assert any("correlation" in code for code in decision.reason_codes)


def test_rolling_correlation_basic() -> None:
    rng = np.random.default_rng(1)
    returns = pd.DataFrame(
        {
            "A": rng.normal(0, 0.01, 80),
            "B": rng.normal(0, 0.01, 80),
        },
        index=pd.bdate_range("2023-01-02", periods=80),
    )
    returns["C"] = returns["A"] * 0.95 + rng.normal(0, 0.001, 80)
    corr = compute_rolling_return_correlation(returns, lookback=60)
    assert corr.loc["A", "C"] > 0.8
    breaches = check_correlation_limit(corr, candidate="C", open_symbols=["A"], reduce_threshold=0.5, reject_threshold=0.9)
    assert breaches and breaches[0].counterparty == "A"


def test_resolve_joint_breach_scales_contributors() -> None:
    classifier = SectorClassifier(sectors={"A": "Tech", "B": "Tech", "C": "Energy"})
    candidate = {"A": 0.20, "B": 0.20, "C": 0.10}
    sector_exposure = {"Tech": 0.0, "Energy": 0.0}
    breaches = {"Tech": 0.40}
    scaled = resolve_joint_breach(
        candidate_allocations=candidate,
        breaches=breaches,
        sector_limit=0.30,
        sector_of=classifier.get_sector_bucket,
    )
    assert scaled["A"] + scaled["B"] == pytest.approx(0.30)
    assert scaled["C"] == pytest.approx(0.10)
