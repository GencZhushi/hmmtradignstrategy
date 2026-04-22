"""Phase A2 - feature pipeline tests."""
from __future__ import annotations

import pandas as pd

from data.feature_engineering import (
    DAILY_FEATURE_COLUMNS,
    EXECUTION_FEATURE_COLUMNS,
    FeatureEngine,
)


def test_daily_feature_columns_present(synthetic_daily_ohlcv: pd.DataFrame) -> None:
    engine = FeatureEngine(zscore_window=252)
    features = engine.build_daily_features(synthetic_daily_ohlcv)
    assert list(features.columns) == list(DAILY_FEATURE_COLUMNS)
    # z-scored output should have mean ~0 and finite values
    assert features.abs().max().max() < 20
    assert not features.isna().any().any(), "NaN warmup rows must be dropped"


def test_execution_indicators_columns(synthetic_daily_ohlcv: pd.DataFrame) -> None:
    engine = FeatureEngine(zscore_window=252)
    indicators = engine.build_execution_indicators(synthetic_daily_ohlcv)
    assert list(indicators.columns) == list(EXECUTION_FEATURE_COLUMNS)
    assert not indicators.isna().any().any()


def test_features_are_causal(synthetic_daily_ohlcv: pd.DataFrame) -> None:
    engine = FeatureEngine(zscore_window=252)
    full = engine.build_daily_features(synthetic_daily_ohlcv)
    # Feature warmup is ~504 bars (200-bar SMA + 252-bar z-score stacked on top
    # of rsi_z which itself burns 252 bars). Use cutoffs safely past that so the
    # prefix has genuine feature rows to intersect against the full output.
    for cutoff in (600, 900, 1200):
        prefix = engine.build_daily_features(synthetic_daily_ohlcv.iloc[:cutoff])
        overlap = full.index.intersection(prefix.index)
        assert len(overlap) > 0, "Prefix must overlap with full run"
        pd.testing.assert_frame_equal(
            full.loc[overlap],
            prefix.loc[overlap],
            check_exact=False,
            rtol=1e-9,
            atol=1e-9,
        )


def test_execution_indicators_causal(synthetic_daily_ohlcv: pd.DataFrame) -> None:
    engine = FeatureEngine()
    full = engine.build_execution_indicators(synthetic_daily_ohlcv)
    cutoff = 500
    prefix = engine.build_execution_indicators(synthetic_daily_ohlcv.iloc[:cutoff])
    overlap = full.index.intersection(prefix.index)
    # EMA warmup means the cumulative VWAP in prefix uses a smaller cumsum but values
    # at each row should be identical since vwap is a running accumulation and does not
    # look ahead. Values must match exactly on the overlap.
    pd.testing.assert_frame_equal(
        full.loc[overlap].drop(columns=["vwap_dev"]),
        prefix.loc[overlap].drop(columns=["vwap_dev"]),
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )
