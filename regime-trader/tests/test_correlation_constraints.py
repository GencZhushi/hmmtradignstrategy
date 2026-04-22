"""Phase A12 - rolling return correlation and correlation-limit guard.

These tests are deterministic so the backtester and the live risk manager
evaluate the same correlation decisions from identical inputs.

Guarantees:

- 60-day rolling correlation is computed from a returns DataFrame
- diagonal is clamped to 1.0 (exact)
- ``check_correlation_limit`` respects reduce/reject thresholds
- breaches carry the symbol, counterparty, correlation, and scope
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.correlation_risk import (
    CorrelationBreach,
    check_correlation_limit,
    compute_rolling_return_correlation,
)


def _synthetic_returns(correlation: float, *, periods: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.01, periods)
    b = correlation * a + np.sqrt(max(1 - correlation**2, 0.0)) * rng.normal(0, 0.01, periods)
    return pd.DataFrame(
        {"A": a, "B": b},
        index=pd.bdate_range("2023-01-02", periods=periods),
    )


def test_rolling_correlation_diagonal_is_one() -> None:
    returns = _synthetic_returns(0.5)
    corr = compute_rolling_return_correlation(returns, lookback=60)
    assert corr.loc["A", "A"] == pytest.approx(1.0)
    assert corr.loc["B", "B"] == pytest.approx(1.0)


def test_rolling_correlation_matches_induced_correlation() -> None:
    returns = _synthetic_returns(0.85)
    corr = compute_rolling_return_correlation(returns, lookback=60)
    # Allow sampling noise at 60 periods.
    assert corr.loc["A", "B"] == pytest.approx(0.85, abs=0.1)


def test_rolling_correlation_empty_input_returns_empty() -> None:
    corr = compute_rolling_return_correlation(pd.DataFrame())
    assert corr.empty


def test_rolling_correlation_short_input_returns_nan_frame() -> None:
    returns = _synthetic_returns(0.5, periods=5)
    corr = compute_rolling_return_correlation(returns, lookback=60)
    # Short window -> NaN placeholder frame with the correct shape.
    assert list(corr.columns) == ["A", "B"]
    assert corr.isna().all().all()


def test_check_correlation_limit_rejects_above_reject_threshold() -> None:
    returns = _synthetic_returns(0.95)
    corr = compute_rolling_return_correlation(returns, lookback=60)
    breaches = check_correlation_limit(
        corr,
        candidate="A",
        open_symbols=["B"],
        reduce_threshold=0.70,
        reject_threshold=0.85,
    )
    assert breaches and breaches[0].scope == "reject"
    assert breaches[0].counterparty == "B"
    assert breaches[0].symbol == "A"
    assert "correlation_reject" in breaches[0].as_reason_code()


def test_check_correlation_limit_reduces_between_thresholds() -> None:
    returns = _synthetic_returns(0.80)
    corr = compute_rolling_return_correlation(returns, lookback=60)
    breaches = check_correlation_limit(
        corr,
        candidate="A",
        open_symbols=["B"],
        reduce_threshold=0.70,
        reject_threshold=0.85,
    )
    assert breaches and breaches[0].scope == "reduce"


def test_check_correlation_limit_ignores_same_symbol() -> None:
    returns = _synthetic_returns(0.5)
    corr = compute_rolling_return_correlation(returns, lookback=60)
    # Candidate == open symbol -> not a breach (self-correlation is 1.0 but filtered).
    breaches = check_correlation_limit(
        corr,
        candidate="A",
        open_symbols=["A"],
        reduce_threshold=0.70,
        reject_threshold=0.85,
    )
    assert breaches == []


def test_check_correlation_limit_empty_corr_returns_empty() -> None:
    assert check_correlation_limit(pd.DataFrame(), candidate="A", open_symbols=["B"]) == []


def test_correlation_breach_reason_code_structure() -> None:
    breach = CorrelationBreach(symbol="AAPL", counterparty="MSFT", correlation=0.9, scope="reject")
    assert breach.as_reason_code() == "correlation_reject:AAPL~MSFT"


def test_check_correlation_limit_skips_unknown_open_symbol() -> None:
    returns = _synthetic_returns(0.9)
    corr = compute_rolling_return_correlation(returns, lookback=60)
    breaches = check_correlation_limit(
        corr,
        candidate="A",
        open_symbols=["UNKNOWN"],
        reduce_threshold=0.70,
        reject_threshold=0.85,
    )
    # Unknown counterparties are silently ignored rather than raising.
    assert breaches == []
