"""Feature pipeline for HMM (daily) and execution (intraday) signals.

All functions are *strictly causal*: every row in the output uses only information
available at or before the corresponding input bar. NaN warmup rows are dropped
in a single, auditable step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DAILY_FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_20",
    "realized_vol_20",
    "vol_ratio_5_20",
    "volume_z",
    "volume_trend",
    "adx_14",
    "sma_slope_50",
    "rsi_z",
    "dist_sma_200",
    "roc_10",
    "roc_20",
    "atr_norm_14",
)


EXECUTION_FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ema_9",
    "ema_21",
    "atr_14",
    "vwap_dev",
    "rsi_14",
)


@dataclass
class FeatureEngine:
    """Stateless feature builder driven entirely by the rolling windows in config."""

    zscore_window: int = 252
    daily_columns: Sequence[str] = field(default_factory=lambda: DAILY_FEATURE_COLUMNS)

    def build_daily_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        _validate_ohlcv(bars)
        out = pd.DataFrame(index=bars.index.copy())
        close = bars["close"]
        high = bars["high"]
        low = bars["low"]
        volume = bars["volume"]

        log_ret = np.log(close).diff()
        out["ret_1"] = log_ret
        out["ret_5"] = log_ret.rolling(5).sum()
        out["ret_20"] = log_ret.rolling(20).sum()

        realized_vol_20 = log_ret.rolling(20).std()
        realized_vol_5 = log_ret.rolling(5).std()
        out["realized_vol_20"] = realized_vol_20
        out["vol_ratio_5_20"] = realized_vol_5 / realized_vol_20.replace(0, np.nan)

        vol_mean_50 = volume.rolling(50).mean()
        vol_std_50 = volume.rolling(50).std().replace(0, np.nan)
        out["volume_z"] = (volume - vol_mean_50) / vol_std_50
        out["volume_trend"] = _rolling_slope(volume.rolling(10).mean(), window=10)

        out["adx_14"] = _adx(high, low, close, period=14)
        out["sma_slope_50"] = _rolling_slope(close.rolling(50).mean(), window=10)

        rsi = _rsi(close, period=14)
        rsi_mean = rsi.rolling(self.zscore_window).mean()
        rsi_std = rsi.rolling(self.zscore_window).std().replace(0, np.nan)
        out["rsi_z"] = (rsi - rsi_mean) / rsi_std

        sma_200 = close.rolling(200).mean()
        out["dist_sma_200"] = (close - sma_200) / sma_200

        out["roc_10"] = close.pct_change(10)
        out["roc_20"] = close.pct_change(20)

        atr_14 = _atr(high, low, close, period=14)
        out["atr_norm_14"] = atr_14 / close

        # Z-score normalize every feature using a causal rolling window.
        normalized = pd.DataFrame(index=out.index)
        for col in self.daily_columns:
            series = out[col]
            mean = series.rolling(self.zscore_window, min_periods=self.zscore_window).mean()
            std = series.rolling(self.zscore_window, min_periods=self.zscore_window).std().replace(0, np.nan)
            normalized[col] = (series - mean) / std

        normalized = normalized.dropna(how="any")
        return normalized

    def build_execution_indicators(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Intraday indicators are *not* HMM inputs; they feed the executor's logic."""
        _validate_ohlcv(bars)
        close = bars["close"]
        high = bars["high"]
        low = bars["low"]
        out = pd.DataFrame(index=bars.index.copy())
        out["ret_1"] = close.pct_change()
        out["ema_9"] = close.ewm(span=9, adjust=False).mean()
        out["ema_21"] = close.ewm(span=21, adjust=False).mean()
        out["atr_14"] = _atr(high, low, close, period=14)
        typical = (high + low + close) / 3.0
        cum_vol = bars["volume"].cumsum().replace(0, np.nan)
        cum_vp = (typical * bars["volume"]).cumsum()
        vwap = cum_vp / cum_vol
        out["vwap_dev"] = (close - vwap) / vwap
        out["rsi_14"] = _rsi(close, period=14)
        return out.dropna(how="any")


def _validate_ohlcv(bars: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(bars.columns)
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {sorted(missing)}")
    if len(bars) < 2:
        raise ValueError("OHLCV frame must contain at least two rows")


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    if window < 2:
        raise ValueError("window must be >= 2")
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(values: np.ndarray) -> float:
        y_mean = values.mean()
        cov = ((x - x_mean) * (values - y_mean)).sum()
        return cov / x_var if x_var else np.nan

    return series.rolling(window).apply(_slope, raw=True)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = _true_range(high, low, close)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _true_range(high, low, close)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def detect_future_leakage(values: Iterable[np.ndarray]) -> bool:
    """Utility helper for tests: returns True if any array differs between prefixes."""
    arrays = list(values)
    if len(arrays) < 2:
        return False
    reference = arrays[0]
    for other in arrays[1:]:
        length = min(reference.shape[0], other.shape[0])
        if not np.allclose(reference[:length], other[:length], equal_nan=True):
            return True
    return False
