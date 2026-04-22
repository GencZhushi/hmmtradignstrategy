"""Price-series adjustment helpers (Phase A11).

Split/dividend-adjusted bars are the default backtest assumption. The engine must
document the policy explicitly so historical results remain reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


AdjustmentPolicy = Literal["split_and_dividend", "split_only", "raw"]


@dataclass
class AdjustmentSettings:
    """Configuration for how raw bars are turned into adjusted bars."""

    policy: AdjustmentPolicy = "split_and_dividend"
    survivorship_bias_aware: bool = True


def apply_adjustment_policy(
    bars: pd.DataFrame,
    *,
    adjustments: pd.DataFrame | None = None,
    settings: AdjustmentSettings | None = None,
) -> pd.DataFrame:
    """Apply split/dividend adjustments to OHLCV bars deterministically.

    ``adjustments`` is an optional frame with columns ``split_ratio`` and
    ``dividend`` indexed by effective date. When the broker already returns
    adjusted data, pass ``adjustments=None`` and ``policy='raw'``.
    """
    settings = settings or AdjustmentSettings()
    if settings.policy == "raw" or adjustments is None or adjustments.empty:
        return bars.copy()

    ordered = bars.sort_index().copy()
    factors = pd.Series(1.0, index=ordered.index)
    if "split_ratio" in adjustments.columns:
        for ts, row in adjustments.sort_index(ascending=False).iterrows():
            ratio = float(row.get("split_ratio", 1.0))
            if ratio and ratio != 1.0:
                mask = ordered.index < ts
                factors.loc[mask] *= ratio
    if settings.policy == "split_and_dividend" and "dividend" in adjustments.columns:
        for ts, row in adjustments.sort_index(ascending=False).iterrows():
            div = float(row.get("dividend", 0.0))
            if div and div != 0.0:
                prior_close = ordered["close"].shift(1).reindex(ordered.index)
                adj = 1.0 - div / prior_close
                mask = ordered.index < ts
                factors.loc[mask] *= adj.loc[mask].fillna(1.0)

    price_cols = [c for c in ("open", "high", "low", "close") if c in ordered.columns]
    for col in price_cols:
        ordered[col] = ordered[col] * factors
    if "volume" in ordered.columns:
        ordered["volume"] = (ordered["volume"] / factors).round()
    return ordered


def normalize_price_series(
    bars: pd.DataFrame,
    *,
    adjustments: pd.DataFrame | None = None,
    settings: AdjustmentSettings | None = None,
) -> pd.DataFrame:
    """Validate and apply adjustments, returning a clean OHLCV frame."""
    if bars.empty:
        return bars.copy()
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"normalize_price_series missing columns: {sorted(missing)}")
    if (bars["close"] <= 0).any():
        raise ValueError("normalize_price_series: non-positive close price detected")
    if (bars["high"] < bars["low"]).any():
        raise ValueError("normalize_price_series: high < low in input bars")
    adjusted = apply_adjustment_policy(bars, adjustments=adjustments, settings=settings)
    return adjusted.astype(float, copy=False)


def detect_missing_bars(index: pd.DatetimeIndex, *, freq: str) -> pd.DatetimeIndex:
    """Return the timestamps that should exist in a full session for ``freq`` but do not."""
    if len(index) == 0:
        return pd.DatetimeIndex([])
    expected = pd.date_range(index[0], index[-1], freq=freq)
    return expected.difference(index)


def detect_outlier_bars(bars: pd.DataFrame, *, zscore: float = 10.0) -> pd.DatetimeIndex:
    """Flag bars whose absolute log-return is a runaway outlier."""
    if len(bars) < 30:
        return pd.DatetimeIndex([])
    log_ret = np.log(bars["close"]).diff()
    std = log_ret.rolling(60, min_periods=30).std().replace(0, np.nan)
    score = (log_ret / std).abs()
    mask = score > zscore
    return bars.index[mask.fillna(False)]
