"""Exchange calendar / session state helpers (Phase A11).

Deliberately lightweight so the engine does not depend on ``pandas_market_calendars``.
Only the rules we actually use are encoded here:

- NYSE trading session runs 09:30 - 16:00 America/New_York on weekdays.
- A small set of fixed US holidays is hard-coded; more robust calendars can be
  plugged in by constructing ``ExchangeCalendar`` with a larger holiday set.
- Bars are "complete" strictly *after* their period end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - py3.8 fallback
    from backports.zoneinfo import ZoneInfo  # type: ignore


DEFAULT_TIMEZONE = "America/New_York"
SESSION_OPEN = time(9, 30)
SESSION_CLOSE = time(16, 0)


FIXED_HOLIDAYS: frozenset[tuple[int, int]] = frozenset(
    {
        (1, 1),   # New Year's Day
        (7, 4),   # Independence Day
        (12, 25), # Christmas Day
    }
)


class SessionState(str):
    """Typed strings for session status (stored as strings for JSON-friendliness)."""

    PRE_MARKET = "pre_market"
    OPEN = "open"
    POST_MARKET = "post_market"
    CLOSED = "closed"
    HOLIDAY = "holiday"


@dataclass
class ExchangeCalendar:
    """Minimal NYSE-style calendar."""

    timezone: str = DEFAULT_TIMEZONE
    holidays: frozenset[date] = field(default_factory=frozenset)
    _tz: ZoneInfo = field(init=False)

    def __post_init__(self) -> None:
        self._tz = ZoneInfo(self.timezone)

    def localize(self, dt: datetime) -> datetime:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc).astimezone(self._tz)
        return dt.astimezone(self._tz)

    def is_holiday(self, day: date) -> bool:
        if (day.month, day.day) in FIXED_HOLIDAYS:
            return True
        return day in self.holidays

    def is_trading_day(self, day: date) -> bool:
        if day.weekday() >= 5:  # Sat, Sun
            return False
        return not self.is_holiday(day)

    def get_exchange_session_state(self, now: datetime) -> str:
        local = self.localize(now)
        if not self.is_trading_day(local.date()):
            return SessionState.HOLIDAY if self.is_holiday(local.date()) else SessionState.CLOSED
        current = local.time()
        if current < SESSION_OPEN:
            return SessionState.PRE_MARKET
        if current >= SESSION_CLOSE:
            return SessionState.POST_MARKET
        return SessionState.OPEN

    def is_market_open(self, now: datetime) -> bool:
        return self.get_exchange_session_state(now) == SessionState.OPEN

    def previous_trading_day(self, day: date) -> date:
        probe = day - timedelta(days=1)
        while not self.is_trading_day(probe):
            probe -= timedelta(days=1)
        return probe


def is_bar_complete(bar_timestamp: datetime, now: datetime, *, is_daily: bool) -> bool:
    """A bar is complete when ``now`` is strictly past its end."""
    if is_daily:
        bar_day = bar_timestamp.date() if isinstance(bar_timestamp, datetime) else bar_timestamp
        now_day = now.date() if isinstance(now, datetime) else now
        return now_day > bar_day
    bar_end = bar_timestamp + timedelta(minutes=5)
    return now >= bar_end


def is_data_stale(
    last_bar: datetime | None,
    now: datetime,
    *,
    max_age: timedelta,
) -> bool:
    if last_bar is None:
        return True
    delta = now - last_bar if now.tzinfo == last_bar.tzinfo else now.astimezone(timezone.utc) - last_bar.astimezone(timezone.utc)
    return delta >= max_age


def get_effective_intraday_regime(
    current_daily_regime_date: date | None,
    now: datetime,
    calendar: ExchangeCalendar,
) -> date | None:
    """Daily regime stays fixed intraday; update only after the previous day closes."""
    local_today = calendar.localize(now).date()
    if current_daily_regime_date is None:
        return None
    if current_daily_regime_date >= local_today:
        return current_daily_regime_date
    return calendar.previous_trading_day(local_today)


def enumerate_trading_days(
    start: date,
    end: date,
    calendar: ExchangeCalendar,
) -> Iterable[date]:
    day = start
    while day <= end:
        if calendar.is_trading_day(day):
            yield day
        day += timedelta(days=1)


def freshness_payload(
    last_daily_bar: datetime | None,
    last_intraday_bar: datetime | None,
    now: datetime,
    *,
    calendar: ExchangeCalendar,
    intraday_stale_minutes: int = 15,
) -> dict[str, object]:
    """Payload used by the API/agent layers (Phase B8)."""
    local_now = calendar.localize(now)
    daily_stale = is_data_stale(last_daily_bar, now, max_age=timedelta(days=2))
    intraday_stale = is_data_stale(
        last_intraday_bar,
        now,
        max_age=timedelta(minutes=intraday_stale_minutes),
    )
    return {
        "exchange_timezone": calendar.timezone,
        "exchange_session_state": calendar.get_exchange_session_state(now),
        "now": local_now.isoformat(),
        "last_completed_daily_bar_time": last_daily_bar.isoformat() if last_daily_bar else None,
        "last_completed_intraday_bar_time": last_intraday_bar.isoformat() if last_intraday_bar else None,
        "data_freshness_status": "stale" if (daily_stale or intraday_stale) else "fresh",
        "daily_data_stale": daily_stale,
        "intraday_data_stale": intraday_stale,
        "regime_effective_session_date": (
            last_daily_bar.date().isoformat() if last_daily_bar else None
        ),
        "stale_data_blocked": daily_stale or intraday_stale,
    }
