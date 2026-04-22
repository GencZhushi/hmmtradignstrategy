"""Phase A11 - exchange calendar + freshness + daily/intraday timing."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from data.exchange_calendar import (
    ExchangeCalendar,
    SessionState,
    freshness_payload,
    get_effective_intraday_regime,
    is_bar_complete,
    is_data_stale,
)


def test_session_state_weekday_open_and_closed() -> None:
    cal = ExchangeCalendar()
    monday_open = datetime(2024, 1, 8, 14, 30, tzinfo=timezone.utc)  # 09:30 NY
    monday_after = datetime(2024, 1, 8, 21, 30, tzinfo=timezone.utc)  # 16:30 NY
    assert cal.get_exchange_session_state(monday_open) == SessionState.OPEN
    assert cal.get_exchange_session_state(monday_after) == SessionState.POST_MARKET


def test_weekend_and_holiday_are_non_trading() -> None:
    cal = ExchangeCalendar()
    saturday = datetime(2024, 1, 6, 14, 30, tzinfo=timezone.utc)
    christmas = datetime(2024, 12, 25, 14, 30, tzinfo=timezone.utc)
    assert cal.get_exchange_session_state(saturday) != SessionState.OPEN
    assert cal.get_exchange_session_state(christmas) == SessionState.HOLIDAY


def test_is_bar_complete_daily_vs_intraday() -> None:
    now = datetime(2024, 1, 8, 20, 30, tzinfo=timezone.utc)
    friday = datetime(2024, 1, 5, 21, 0, tzinfo=timezone.utc)
    assert is_bar_complete(friday, now, is_daily=True)
    fresh_intraday = now - timedelta(minutes=1)
    assert not is_bar_complete(fresh_intraday, now, is_daily=False)
    stale_intraday = now - timedelta(minutes=6)
    assert is_bar_complete(stale_intraday, now, is_daily=False)


def test_is_data_stale_respects_max_age() -> None:
    now = datetime(2024, 1, 8, 20, 30, tzinfo=timezone.utc)
    assert is_data_stale(None, now, max_age=timedelta(minutes=15))
    fresh = now - timedelta(minutes=5)
    assert not is_data_stale(fresh, now, max_age=timedelta(minutes=15))


def test_effective_intraday_regime_stays_fixed_intraday() -> None:
    cal = ExchangeCalendar()
    today = datetime(2024, 1, 8, 15, 0, tzinfo=timezone.utc)
    current_date = date(2024, 1, 8)
    assert get_effective_intraday_regime(current_date, today, cal) == current_date
    # If no regime has been computed yet, return None.
    assert get_effective_intraday_regime(None, today, cal) is None


def test_freshness_payload_flags_stale_data() -> None:
    cal = ExchangeCalendar()
    now = datetime(2024, 1, 8, 20, 30, tzinfo=timezone.utc)
    payload = freshness_payload(
        last_daily_bar=now - timedelta(days=3),
        last_intraday_bar=now - timedelta(minutes=30),
        now=now,
        calendar=cal,
    )
    assert payload["stale_data_blocked"] is True
    assert payload["daily_data_stale"] is True
    assert payload["intraday_data_stale"] is True
    assert payload["exchange_timezone"] == cal.timezone
