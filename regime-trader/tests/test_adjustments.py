"""Phase A11 - split/dividend adjustment policy must be explicit and deterministic.

Guarantees tested here:

- ``policy='raw'`` is an identity transform.
- ``policy='split_only'`` only applies ``split_ratio`` factors.
- ``policy='split_and_dividend'`` additionally back-adjusts for dividends.
- ``normalize_price_series`` rejects malformed OHLCV (missing columns,
  non-positive close, high < low).
- Volumes are inversely scaled by the same factor so notional is preserved.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.adjustments import (
    AdjustmentSettings,
    apply_adjustment_policy,
    detect_outlier_bars,
    normalize_price_series,
)


def _simple_bars() -> pd.DataFrame:
    idx = pd.bdate_range("2023-01-02", periods=10)
    return pd.DataFrame(
        {
            "open": np.linspace(100, 110, 10),
            "high": np.linspace(101, 111, 10),
            "low": np.linspace(99, 109, 10),
            "close": np.linspace(100, 110, 10),
            "volume": np.full(10, 1_000_000, dtype=float),
        },
        index=idx,
    )


def test_raw_policy_is_identity_transform() -> None:
    bars = _simple_bars()
    adjustments = pd.DataFrame(
        {"split_ratio": [2.0]},
        index=[bars.index[5]],
    )
    result = apply_adjustment_policy(
        bars,
        adjustments=adjustments,
        settings=AdjustmentSettings(policy="raw"),
    )
    pd.testing.assert_frame_equal(result, bars)


def test_split_only_scales_prices_before_effective_date() -> None:
    bars = _simple_bars()
    split_date = bars.index[5]
    adjustments = pd.DataFrame({"split_ratio": [2.0]}, index=[split_date])
    result = apply_adjustment_policy(
        bars,
        adjustments=adjustments,
        settings=AdjustmentSettings(policy="split_only"),
    )
    # Pre-split rows are re-expressed in the post-split price convention used
    # by the pipeline: ``adjusted = raw * split_ratio``. Volumes scale the
    # other way so notional stays invariant (see dedicated test below).
    before = bars.index < split_date
    assert np.allclose(result.loc[before, "close"], bars.loc[before, "close"] * 2.0)
    # Rows on/after the split stay the same.
    after = bars.index >= split_date
    assert np.allclose(result.loc[after, "close"], bars.loc[after, "close"])


def test_split_and_dividend_policy_applies_dividend_factor() -> None:
    bars = _simple_bars()
    div_date = bars.index[4]
    # A $1 dividend on a ~$104 prior close should knock ~1% off pre-ex prices.
    adjustments = pd.DataFrame({"split_ratio": [1.0], "dividend": [1.0]}, index=[div_date])
    result = apply_adjustment_policy(
        bars,
        adjustments=adjustments,
        settings=AdjustmentSettings(policy="split_and_dividend"),
    )
    # The first bar has no ``prior_close`` so it stays unchanged. Any row
    # between index 1 and the dividend date (exclusive) must be dampened by
    # the (1 - div/prior_close) factor.
    pre_ex = bars.index[1]
    assert result.loc[pre_ex, "close"] < bars.loc[pre_ex, "close"]
    # On/after ex-date stays unchanged.
    assert result.loc[bars.index[-1], "close"] == pytest.approx(bars.loc[bars.index[-1], "close"])


def test_split_scales_volume_inversely() -> None:
    bars = _simple_bars()
    split_date = bars.index[5]
    adjustments = pd.DataFrame({"split_ratio": [2.0]}, index=[split_date])
    result = apply_adjustment_policy(
        bars,
        adjustments=adjustments,
        settings=AdjustmentSettings(policy="split_only"),
    )
    before = bars.index < split_date
    # Price is multiplied by split_ratio so volume must be divided by the
    # same factor to preserve traded notional.
    assert np.allclose(result.loc[before, "volume"], bars.loc[before, "volume"] / 2.0)


def test_normalize_price_series_rejects_missing_columns() -> None:
    bad = _simple_bars().drop(columns=["volume"])
    with pytest.raises(ValueError, match="missing columns"):
        normalize_price_series(bad)


def test_normalize_price_series_rejects_non_positive_close() -> None:
    bars = _simple_bars().copy()
    bars.iloc[0, bars.columns.get_loc("close")] = 0.0
    with pytest.raises(ValueError, match="non-positive close"):
        normalize_price_series(bars)


def test_normalize_price_series_rejects_inverted_highlow() -> None:
    bars = _simple_bars().copy()
    bars.iloc[3, bars.columns.get_loc("low")] = bars.iloc[3]["high"] + 10
    with pytest.raises(ValueError, match="high < low"):
        normalize_price_series(bars)


def test_detect_outlier_bars_flags_runaway_returns() -> None:
    # 120 gentle bars plus one spike that is ~50 std devs wide.
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 120)))
    bars = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(120, 1_000_000, dtype=float),
        },
        index=pd.bdate_range("2022-01-03", periods=120),
    )
    # Inject a large outlier on the last bar. The rolling 60-bar std includes
    # the outlier itself, so the z threshold must allow for that self-inflation;
    # z=5 is comfortably above normal 5-std market moves yet below the ~8 std
    # produced by a 2x close spike on the last bar.
    bars.iloc[-1, bars.columns.get_loc("close")] *= 2.0
    outliers = detect_outlier_bars(bars, zscore=5.0)
    assert bars.index[-1] in outliers
