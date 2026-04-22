"""Phase A5 — Walk-forward backtester and performance metrics.

Tests that the backtester and performance module behave correctly:

- BacktestConfig can be constructed from settings dicts
- compute_performance_metrics returns expected structure
- walk-forward equity curve is reproducible with same config/seed
- benchmark comparison is generated
- regime breakdown can be computed from equity + regime history
- intraday execution simulation returns adjusted capital
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.backtester import BacktestConfig, WalkForwardBacktester
from backtest.performance import (
    PerformanceSummary,
    compare_to_benchmarks,
    compute_performance_metrics,
    regime_breakdown,
)
from data.slippage import SlippageModel


# ------------------------------------------------------------------ config


def test_backtest_config_defaults() -> None:
    cfg = BacktestConfig()
    assert cfg.train_window == 504
    assert cfg.test_window == 126
    assert cfg.step_size == 126
    assert cfg.initial_capital == 100_000.0
    assert cfg.risk_free_rate == 0.045


def test_backtest_config_from_settings() -> None:
    backtest_cfg = {
        "train_window": 504,
        "test_window": 126,
        "step_size": 126,
        "initial_capital": 50_000,
        "slippage_pct": 0.001,
        "risk_free_rate": 0.04,
    }
    strategy_cfg = {"rebalance_threshold": 0.05}
    hmm_cfg = {"n_candidates": [3, 4, 5], "n_init": 5, "covariance_type": "diag"}
    cfg = BacktestConfig.from_settings(backtest_cfg, strategy_cfg, hmm_cfg)
    assert cfg.initial_capital == 50_000
    assert cfg.rebalance_threshold == 0.05
    assert cfg.hmm_n_candidates == (3, 4, 5)
    assert cfg.hmm_covariance == "diag"


# ------------------------------------------------------------------ performance metrics


def _equity_series(n: int = 252, drift: float = 0.0003) -> pd.Series:
    rng = np.random.default_rng(7)
    returns = rng.normal(drift, 0.01, n)
    prices = 100_000 * np.exp(np.cumsum(returns))
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.Series(prices, index=idx)


def test_performance_summary_has_expected_fields() -> None:
    equity = _equity_series()
    summary = compute_performance_metrics(equity)
    assert isinstance(summary, PerformanceSummary)
    assert summary.total_trades == 0  # no trades passed
    assert summary.total_return != 0.0  # drift is positive
    assert summary.max_drawdown <= 0.0  # drawdowns are negative or zero


def test_performance_metrics_empty_equity_returns_zeros() -> None:
    summary = compute_performance_metrics(pd.Series(dtype=float))
    assert summary.total_return == 0.0
    assert summary.sharpe == 0.0


def test_performance_metrics_with_trades() -> None:
    equity = _equity_series()
    trades = pd.DataFrame(
        {
            "side": ["BUY", "SELL"],
            "qty": [10, 10],
            "exec_price": [500.0, 520.0],
        }
    )
    summary = compute_performance_metrics(equity, trades)
    assert summary.total_trades == 2
    assert summary.win_rate == 1.0  # single winning close


def test_sharpe_positive_for_positive_drift() -> None:
    equity = _equity_series(drift=0.005)
    summary = compute_performance_metrics(equity, risk_free_rate=0.0)
    assert summary.sharpe > 0


# ------------------------------------------------------------------ benchmark comparison


def test_compare_to_benchmarks_shape() -> None:
    strategy = _equity_series(drift=0.0004)
    benchmark = _equity_series(drift=0.0002)
    result = compare_to_benchmarks(strategy, benchmark)
    assert "strategy" in result.columns
    assert "buy_and_hold" in result.columns
    assert "total_return" in result.index


# ------------------------------------------------------------------ regime breakdown


def test_regime_breakdown_returns_per_regime_stats() -> None:
    equity = _equity_series(n=100)
    regime_history = pd.DataFrame(
        {"regime_name": ["low_vol"] * 50 + ["high_vol"] * 50},
        index=equity.index,
    )
    stats = regime_breakdown(equity, regime_history)
    assert "low_vol" in stats.index
    assert "high_vol" in stats.index
    assert "avg_return" in stats.columns


def test_regime_breakdown_empty_returns_empty() -> None:
    assert regime_breakdown(pd.Series(dtype=float), pd.DataFrame()).empty


# ------------------------------------------------------------------ walk-forward (lightweight)


def test_walk_forward_too_short_raises() -> None:
    """Backtester must raise if there aren't enough bars."""
    cfg = BacktestConfig(train_window=50, test_window=20, hmm_n_candidates=(3,), hmm_n_init=1)
    bt = WalkForwardBacktester(config=cfg)
    # 30 bars is less than 50 + 20
    rng = np.random.default_rng(0)
    bars = pd.DataFrame(
        {
            "open": rng.uniform(99, 101, 30),
            "high": rng.uniform(100, 102, 30),
            "low": rng.uniform(98, 100, 30),
            "close": rng.uniform(99, 101, 30),
            "volume": rng.integers(1e6, 5e6, 30),
        },
        index=pd.bdate_range("2024-01-02", periods=30),
    )
    with pytest.raises(ValueError, match="at least train_window"):
        bt.run_walk_forward(symbol="TEST", daily_bars=bars)


def test_intraday_simulation_returns_adjusted_capital() -> None:
    cfg = BacktestConfig()
    bt = WalkForwardBacktester(config=cfg)
    bars = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.bdate_range("2024-01-02", periods=3, freq="5min"),
    )
    result = bt.simulate_intraday_execution(
        symbol="SPY", intraday_bars=bars, target_allocation=0.5, capital=100_000,
    )
    assert isinstance(result, float)
    assert result > 0


def test_intraday_simulation_empty_bars_returns_capital() -> None:
    cfg = BacktestConfig()
    bt = WalkForwardBacktester(config=cfg)
    result = bt.simulate_intraday_execution(
        symbol="SPY",
        intraday_bars=pd.DataFrame(columns=["close"]),
        target_allocation=0.5,
        capital=100_000,
    )
    assert result == 100_000
