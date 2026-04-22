"""Phase A4 - strategy + signal generator tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import RegimeInfo
from core.regime_strategies import (
    HighVolDefensiveStrategy,
    LowVolBullStrategy,
    MidVolCautiousStrategy,
    StrategyConfig,
    StrategyOrchestrator,
    apply_uncertainty_mode,
    generate_target_allocation,
)


def _make_bars(n: int = 120, drift: float = 0.0005, vol: float = 0.01, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, vol / 3, n)),
            "high": close * (1 + rng.uniform(0, vol, n)),
            "low": close * (1 - rng.uniform(0, vol, n)),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=pd.bdate_range("2022-01-03", periods=n),
    )
    df["high"] = np.maximum.reduce([df["open"], df["close"], df["high"]])
    df["low"] = np.minimum.reduce([df["open"], df["close"], df["low"]])
    return df


def _regime_infos() -> list[RegimeInfo]:
    return [
        RegimeInfo(regime_id=0, regime_name="BULL", expected_return=0.001, expected_volatility=0.008, vol_rank=0.0, label_return_rank=2, max_leverage_allowed=1.25),
        RegimeInfo(regime_id=1, regime_name="NEUTRAL", expected_return=0.0, expected_volatility=0.015, vol_rank=0.5, label_return_rank=1, max_leverage_allowed=1.0),
        RegimeInfo(regime_id=2, regime_name="BEAR", expected_return=-0.001, expected_volatility=0.03, vol_rank=1.0, label_return_rank=0, max_leverage_allowed=1.0),
    ]


def test_orchestrator_maps_low_vol_to_bull_strategy() -> None:
    config = StrategyConfig()
    orchestrator = StrategyOrchestrator(config=config, regime_infos=_regime_infos())
    assert isinstance(orchestrator.strategy_for(0), LowVolBullStrategy)
    assert isinstance(orchestrator.strategy_for(2), HighVolDefensiveStrategy)
    middle = orchestrator.strategy_for(1)
    assert isinstance(middle, MidVolCautiousStrategy)


def test_low_vol_bull_strategy_full_allocation() -> None:
    config = StrategyConfig()
    bars = _make_bars()
    strat = LowVolBullStrategy(config)
    signal = strat.generate_signal(
        symbol="SPY",
        bars=bars,
        regime_info=_regime_infos()[0],
        regime_probability=0.9,
    )
    assert signal is not None
    assert signal.target_allocation_pct == pytest.approx(config.low_vol_allocation)
    assert signal.leverage == pytest.approx(config.low_vol_leverage)
    assert signal.stop_loss is not None and signal.stop_loss < signal.entry_price


def test_high_vol_defensive_strategy_reduced_allocation_not_reversal() -> None:
    config = StrategyConfig()
    bars = _make_bars(vol=0.03, drift=-0.0005, seed=1)
    strat = HighVolDefensiveStrategy(config)
    signal = strat.generate_signal(
        symbol="SPY",
        bars=bars,
        regime_info=_regime_infos()[2],
        regime_probability=0.8,
    )
    assert signal is not None
    assert signal.direction.value == "LONG", "Engine is long-only"
    assert signal.target_allocation_pct == pytest.approx(config.high_vol_allocation)
    assert signal.leverage == pytest.approx(1.0)


def test_mid_vol_trend_vs_no_trend_allocation() -> None:
    config = StrategyConfig()
    strat = MidVolCautiousStrategy(config)

    uptrend = _make_bars(drift=0.001, vol=0.012, seed=2)
    downtrend = uptrend.copy()
    downtrend["close"] = downtrend["close"].iloc[::-1].to_numpy()
    downtrend["high"] = downtrend["close"] * 1.01
    downtrend["low"] = downtrend["close"] * 0.99
    downtrend["open"] = downtrend["close"]

    regime_info = _regime_infos()[1]
    up_signal = strat.generate_signal(symbol="SPY", bars=uptrend, regime_info=regime_info, regime_probability=0.7)
    down_signal = strat.generate_signal(symbol="SPY", bars=downtrend, regime_info=regime_info, regime_probability=0.7)
    assert up_signal is not None and down_signal is not None
    assert up_signal.target_allocation_pct >= down_signal.target_allocation_pct


def test_uncertainty_mode_halves_size_and_forces_1x_leverage() -> None:
    config = StrategyConfig(uncertainty_size_mult=0.5)
    strat = LowVolBullStrategy(config)
    signal = strat.generate_signal(
        symbol="SPY",
        bars=_make_bars(),
        regime_info=_regime_infos()[0],
        regime_probability=0.9,
    )
    assert signal is not None
    reduced = apply_uncertainty_mode(signal, config)
    assert reduced.target_allocation_pct == pytest.approx(signal.target_allocation_pct * 0.5)
    assert reduced.leverage == 1.0
    assert "UNCERTAINTY - size halved" in reduced.reasoning


def test_rebalance_threshold_blocks_small_changes() -> None:
    unchanged = generate_target_allocation(current_alloc_pct=0.95, target_alloc_pct=0.99, rebalance_threshold=0.10)
    assert unchanged == pytest.approx(0.95)
    changed = generate_target_allocation(current_alloc_pct=0.60, target_alloc_pct=0.95, rebalance_threshold=0.10)
    assert changed == pytest.approx(0.95)


def test_strategy_selection_deterministic_for_same_regimes() -> None:
    config = StrategyConfig()
    a = StrategyOrchestrator(config=config, regime_infos=_regime_infos())
    b = StrategyOrchestrator(config=config, regime_infos=_regime_infos())
    assert type(a.strategy_for(0)) is type(b.strategy_for(0))
    assert type(a.strategy_for(1)) is type(b.strategy_for(1))
    assert type(a.strategy_for(2)) is type(b.strategy_for(2))
