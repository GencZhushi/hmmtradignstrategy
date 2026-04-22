"""Market data loader for daily (regime) and intraday (execution) bars.

The engine distinguishes strictly between the two timeframes:

- ``fetch_historical_daily_bars`` returns *completed* daily bars used by the HMM.
- ``fetch_intraday_bars`` returns 5-minute execution bars.

Implementation priorities follow Spec A:

1. Support a pluggable "provider" so backtests and unit tests can use CSV or
   synthetic data without reaching out to Alpaca.
2. Only return *completed* bars; callers never receive the in-flight bar.
3. Cache results on disk to keep backtests reproducible.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol

import pandas as pd

LOG = logging.getLogger(__name__)

DAILY_TIMEFRAME = "1Day"
INTRADAY_TIMEFRAME = "5Min"
REQUIRED_OHLCV = ("open", "high", "low", "close", "volume")


class DataProvider(Protocol):
    """Minimal interface the market data manager uses to fetch bars."""

    def daily_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame: ...
    def intraday_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame: ...


@dataclass
class CsvProvider:
    """Reads OHLCV bars from a directory of ``<symbol>_<timeframe>.csv`` files."""

    root: Path
    date_column: str = "date"

    def _load(self, symbol: str, timeframe: str) -> pd.DataFrame:
        path = self.root / f"{symbol}_{timeframe}.csv"
        if not path.exists():
            raise FileNotFoundError(f"No CSV bars for {symbol} [{timeframe}] at {path}")
        df = pd.read_csv(path, parse_dates=[self.date_column])
        df = df.rename(columns={self.date_column: "timestamp"})
        df = df.set_index("timestamp").sort_index()
        return _normalize_ohlcv(df)

    def daily_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return _slice(self._load(symbol, DAILY_TIMEFRAME), start, end)

    def intraday_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return _slice(self._load(symbol, INTRADAY_TIMEFRAME), start, end)


@dataclass
class InMemoryProvider:
    """Keyed by ``(symbol, timeframe)`` for deterministic unit tests."""

    frames: dict[tuple[str, str], pd.DataFrame] = field(default_factory=dict)

    def _lookup(self, symbol: str, timeframe: str) -> pd.DataFrame:
        key = (symbol, timeframe)
        if key not in self.frames:
            raise KeyError(f"InMemoryProvider missing {key}")
        return _normalize_ohlcv(self.frames[key].copy())

    def daily_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return _slice(self._lookup(symbol, DAILY_TIMEFRAME), start, end)

    def intraday_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return _slice(self._lookup(symbol, INTRADAY_TIMEFRAME), start, end)


@dataclass
class AlpacaDataProvider:
    """Fetches OHLCV bars from Alpaca Market Data (REST, IEX feed by default).

    Works for both paper and live accounts (historical data is account-agnostic).
    Results are cached on disk under ``cache_dir/<symbol>_<timeframe>.parquet``
    to keep repeated fetches fast and reproducible.
    """

    api_key: str
    secret_key: str
    cache_dir: Path | None = None
    feed: str = "iex"  # "iex" is free; "sip" requires a paid data plan
    _client: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            from alpaca.data.historical import StockHistoricalDataClient  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "alpaca-py is required for AlpacaDataProvider. Install with: pip install alpaca-py"
            ) from exc
        self._client = StockHistoricalDataClient(api_key=self.api_key, secret_key=self.secret_key)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def daily_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self._fetch(symbol, DAILY_TIMEFRAME, start, end)

    def intraday_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return self._fetch(symbol, INTRADAY_TIMEFRAME, start, end)

    def _fetch(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest  # type: ignore
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit  # type: ignore

        cached = self._read_cache(symbol, timeframe)
        tf = self._map_timeframe(timeframe, TimeFrame, TimeFrameUnit)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            feed=self.feed,
        )
        try:
            response = self._client.get_stock_bars(request)  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - network path
            LOG.warning("Alpaca data fetch failed for %s [%s]: %s", symbol, timeframe, exc)
            if cached is not None and not cached.empty:
                return _slice(cached, start, end)
            raise

        df = self._response_to_df(response, symbol)
        merged = df if cached is None or cached.empty else _merge_frames(cached, df)
        self._write_cache(symbol, timeframe, merged)
        return _slice(_normalize_ohlcv(merged), start, end)

    @staticmethod
    def _map_timeframe(timeframe: str, TimeFrame, TimeFrameUnit):  # type: ignore[no-untyped-def]
        if timeframe == DAILY_TIMEFRAME:
            return TimeFrame.Day
        if timeframe == INTRADAY_TIMEFRAME:
            return TimeFrame(5, TimeFrameUnit.Minute)
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    @staticmethod
    def _response_to_df(response, symbol: str) -> pd.DataFrame:  # type: ignore[no-untyped-def]
        df = getattr(response, "df", None)
        if df is None or df.empty:
            return pd.DataFrame(columns=list(REQUIRED_OHLCV))
        # Alpaca returns a MultiIndex (symbol, timestamp). Drop the symbol level.
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol") if "symbol" in df.index.names else df.droplevel(0)
        df = df.rename_axis("timestamp")
        # Keep only required OHLCV columns (Alpaca also returns trade_count, vwap).
        keep = [c for c in REQUIRED_OHLCV if c in df.columns]
        return df[keep].copy()

    def _cache_path(self, symbol: str, timeframe: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{symbol}_{timeframe}.parquet"

    def _read_cache(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        path = self._cache_path(symbol, timeframe)
        if path is None or not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception as exc:  # pragma: no cover - corrupt cache
            LOG.warning("Failed to read cache %s: %s", path, exc)
            return None

    def _write_cache(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        path = self._cache_path(symbol, timeframe)
        if path is None or df.empty:
            return
        try:
            df.to_parquet(path)
        except Exception as exc:  # pragma: no cover - pyarrow optional
            LOG.debug("Parquet cache unavailable (%s); falling back to CSV", exc)
            df.to_csv(path.with_suffix(".csv"))


def _merge_frames(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Union two OHLCV frames deterministically; later rows win on conflicts."""
    if existing is None or existing.empty:
        return new.copy()
    if new is None or new.empty:
        return existing.copy()
    merged = pd.concat([existing, new])
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_OHLCV if col not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing}")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df[list(REQUIRED_OHLCV)].astype(float, copy=False)


def _slice(df: pd.DataFrame, start: datetime | None, end: datetime | None) -> pd.DataFrame:
    if df.empty:
        return df
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


@dataclass
class MarketDataManager:
    """Coordinates daily/intraday data loads and freshness bookkeeping."""

    provider: DataProvider
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))
    _last_daily_ts: dict[str, pd.Timestamp] = field(default_factory=dict, init=False)
    _last_intraday_ts: dict[str, pd.Timestamp] = field(default_factory=dict, init=False)

    def fetch_historical_daily_bars(
        self,
        symbol: str,
        lookback_bars: int = 504,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        end = end or self.clock()
        # Approximate calendar span; downstream code only uses bars that actually exist.
        start = end - timedelta(days=int(lookback_bars * 1.6 + 10))
        bars = self.provider.daily_bars(symbol, start, end)
        bars = _drop_incomplete_tail(bars, end, is_daily=True)
        if len(bars) == 0:
            LOG.warning("No daily bars returned for %s", symbol)
            return bars
        if len(bars) < lookback_bars:
            LOG.warning(
                "Only %d completed daily bars available for %s (<%d)",
                len(bars),
                symbol,
                lookback_bars,
            )
        bars = bars.tail(lookback_bars)
        self._last_daily_ts[symbol] = bars.index[-1]
        return bars

    def fetch_intraday_bars(
        self,
        symbol: str,
        lookback_bars: int = 390,  # roughly one NYSE session of 5-min bars
        end: datetime | None = None,
    ) -> pd.DataFrame:
        end = end or self.clock()
        start = end - timedelta(days=10)
        bars = self.provider.intraday_bars(symbol, start, end)
        bars = _drop_incomplete_tail(bars, end, is_daily=False)
        if len(bars) == 0:
            return bars
        bars = bars.tail(lookback_bars)
        self._last_intraday_ts[symbol] = bars.index[-1]
        return bars

    def last_completed_daily_bar_time(self, symbol: str) -> pd.Timestamp | None:
        return self._last_daily_ts.get(symbol)

    def last_completed_intraday_bar_time(self, symbol: str) -> pd.Timestamp | None:
        return self._last_intraday_ts.get(symbol)

    def freshness_snapshot(self, symbols: Iterable[str]) -> dict[str, dict[str, str | None]]:
        """Structured freshness payload used by the API and agent layers."""
        snapshot: dict[str, dict[str, str | None]] = {}
        for symbol in symbols:
            daily = self._last_daily_ts.get(symbol)
            intraday = self._last_intraday_ts.get(symbol)
            snapshot[symbol] = {
                "last_completed_daily_bar_time": daily.isoformat() if daily is not None else None,
                "last_completed_intraday_bar_time": intraday.isoformat() if intraday is not None else None,
            }
        return snapshot


def _drop_incomplete_tail(df: pd.DataFrame, now: datetime, *, is_daily: bool) -> pd.DataFrame:
    """Remove any in-flight bar whose period has not completed yet."""
    if df.empty:
        return df
    tail_ts = df.index[-1]
    now_ts = pd.Timestamp(now)
    if is_daily:
        # A daily bar is complete when we are strictly past the bar's day.
        if tail_ts.normalize() >= now_ts.normalize():
            df = df.iloc[:-1]
    else:
        # A 5-minute bar is complete when at least 5 minutes past its start have elapsed.
        if now_ts - tail_ts < pd.Timedelta(minutes=5):
            df = df.iloc[:-1]
    return df


def build_provider(config: Mapping[str, object]) -> DataProvider:
    """Factory used by the application bootstrap.

    Selection order:

    1. If ``provider == 'alpaca'`` and Alpaca credentials are present,
       use :class:`AlpacaDataProvider` with an on-disk cache under ``cache_dir``.
    2. Else if a CSV directory exists at ``data_dir``, use :class:`CsvProvider`.
    3. Else fall back to an empty :class:`InMemoryProvider` (useful for tests
       and offline smoke runs).
    """
    provider_name = str(config.get("provider", "auto")).lower()
    api_key = config.get("alpaca_api_key")
    secret_key = config.get("alpaca_secret_key")
    cache_dir = config.get("cache_dir")
    feed = str(config.get("alpaca_feed", "iex"))

    want_alpaca = provider_name == "alpaca" or (provider_name == "auto" and api_key and secret_key)
    if want_alpaca and api_key and secret_key:
        cache_path = Path(str(cache_dir)) if cache_dir else None
        LOG.info("Using AlpacaDataProvider (feed=%s cache=%s)", feed, cache_path)
        return AlpacaDataProvider(
            api_key=str(api_key),
            secret_key=str(secret_key),
            cache_dir=cache_path,
            feed=feed,
        )

    root = Path(str(config.get("data_dir", Path(__file__).resolve().parent / "bars")))
    if root.exists():
        return CsvProvider(root=root)
    LOG.warning("No data directory at %s; using empty in-memory provider", root)
    return InMemoryProvider()
