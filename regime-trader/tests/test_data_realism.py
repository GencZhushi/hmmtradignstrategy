"""Phase A11 - adjustments, slippage, overnight gap simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.adjustments import (
    AdjustmentSettings,
    apply_adjustment_policy,
    detect_missing_bars,
    detect_outlier_bars,
    normalize_price_series,
)
from data.slippage import SlippageModel, simulate_gap_fill_behavior


def _frame(n: int = 60, start: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0002, 0.01, n)
    close = start * np.exp(np.cumsum(rets))
    idx = pd.bdate_range("2024-01-02", periods=n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 2_000_000, n),
        },
        index=idx,
    )


def test_normalize_price_series_rejects_bad_prices() -> None:
    bars = _frame()
    bars.loc[bars.index[5], "close"] = -1.0
    with pytest.raises(ValueError):
        normalize_price_series(bars)


def test_apply_split_adjustment_scales_prior_bars() -> None:
    bars = _frame(20)
    split_date = bars.index[10]
    adjustments = pd.DataFrame(
        {"split_ratio": [0.5], "dividend": [0.0]},
        index=pd.DatetimeIndex([split_date]),
    )
    adjusted = apply_adjustment_policy(bars, adjustments=adjustments, settings=AdjustmentSettings(policy="split_only"))
    assert adjusted.loc[bars.index[0], "close"] == pytest.approx(bars.loc[bars.index[0], "close"] * 0.5)
    assert adjusted.loc[split_date, "close"] == pytest.approx(bars.loc[split_date, "close"])


def test_detect_missing_bars_returns_gaps() -> None:
    idx = pd.date_range("2024-01-02 09:30", periods=10, freq="5min").tolist()
    idx.pop(5)  # drop a bar
    df_index = pd.DatetimeIndex(idx)
    missing = detect_missing_bars(df_index, freq="5min")
    assert len(missing) == 1


def test_detect_outlier_bars() -> None:
    bars = _frame(120)
    bars.loc[bars.index[60], "close"] = bars["close"].iloc[59] * 2.0  # +100% jump
    outliers = detect_outlier_bars(bars, zscore=6.0)
    assert bars.index[60] in outliers


def test_slippage_scales_with_volatility_and_spread() -> None:
    model = SlippageModel(base_pct=0.0005, vol_multiplier=0.5, spread_multiplier=0.5)
    low = model.estimate_slippage(reference_price=100.0, realized_vol=0.01, spread_pct=0.001)
    high = model.estimate_slippage(reference_price=100.0, realized_vol=0.05, spread_pct=0.01)
    assert high > low


def test_slippage_apply_moves_price_against_side() -> None:
    model = SlippageModel(base_pct=0.01)
    buy = model.apply(reference_price=100.0, side="BUY")
    sell = model.apply(reference_price=100.0, side="SELL")
    assert buy > 100.0 and sell < 100.0


def test_overnight_gap_through_stop_exits_at_open() -> None:
    price = simulate_gap_fill_behavior(prior_close=100.0, gap_pct=-0.05, stop_price=97.0, side="LONG")
    # Gap is -5% -> open at 95; stop was 97 -> triggered; exit at open.
    assert price == pytest.approx(95.0)


def test_overnight_gap_keeps_normal_stop_when_in_bounds() -> None:
    price = simulate_gap_fill_behavior(prior_close=100.0, gap_pct=0.01, stop_price=97.0, side="LONG")
    assert price == pytest.approx(101.0)
