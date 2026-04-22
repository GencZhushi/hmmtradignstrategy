"""Phase A11 - missing / incomplete bar detection.

Incomplete or stale intraday bars must never feed execution decisions. The
helpers under test here are the boundary guards that:

- identify which timestamps are absent from a session's expected grid
- mark a bar as "complete" only strictly after its period end
- flag intraday and daily staleness once a configurable age window is exceeded
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from data.adjustments import detect_missing_bars
from data.exchange_calendar import ExchangeCalendar, is_bar_complete, is_data_stale


def test_detect_missing_bars_flags_gaps_inside_session() -> None:
    expected = pd.date_range("2024-05-15 09:30", "2024-05-15 10:00", freq="5min")
    # Drop the 09:45 bar to simulate a feed glitch.
    observed = expected.drop(pd.Timestamp("2024-05-15 09:45"))
    missing = detect_missing_bars(observed, freq="5min")
    assert list(missing) == [pd.Timestamp("2024-05-15 09:45")]


def test_detect_missing_bars_returns_empty_when_fully_populated() -> None:
    idx = pd.date_range("2024-05-15 09:30", "2024-05-15 10:00", freq="5min")
    assert len(detect_missing_bars(idx, freq="5min")) == 0


def test_detect_missing_bars_handles_empty_index() -> None:
    assert len(detect_missing_bars(pd.DatetimeIndex([]), freq="5min")) == 0


def test_intraday_bar_is_not_complete_until_period_end_passes() -> None:
    bar = datetime(2024, 5, 15, 9, 30, tzinfo=timezone.utc)
    # At the exact boundary the bar is considered complete (inclusive semantics).
    assert is_bar_complete(bar, bar + timedelta(minutes=5), is_daily=False) is True
    # Anything before the boundary is incomplete and must not drive decisions.
    assert is_bar_complete(bar, bar + timedelta(minutes=4), is_daily=False) is False


def test_daily_bar_completes_only_after_the_session_date_flips() -> None:
    bar = datetime(2024, 5, 15, 16, 0, tzinfo=timezone.utc)
    # Same calendar day -> not complete.
    assert is_bar_complete(bar, bar + timedelta(hours=1), is_daily=True) is False
    # Next calendar day -> complete.
    assert is_bar_complete(bar, bar + timedelta(days=1), is_daily=True) is True


def test_is_data_stale_with_no_last_bar_is_stale() -> None:
    now = datetime(2024, 5, 15, 10, 0, tzinfo=timezone.utc)
    assert is_data_stale(None, now, max_age=timedelta(minutes=15)) is True


def test_is_data_stale_within_window_is_fresh() -> None:
    last = datetime(2024, 5, 15, 9, 55, tzinfo=timezone.utc)
    now = datetime(2024, 5, 15, 10, 0, tzinfo=timezone.utc)
    assert is_data_stale(last, now, max_age=timedelta(minutes=15)) is False


def test_is_data_stale_outside_window_is_stale() -> None:
    last = datetime(2024, 5, 15, 9, 30, tzinfo=timezone.utc)
    now = datetime(2024, 5, 15, 10, 0, tzinfo=timezone.utc)
    assert is_data_stale(last, now, max_age=timedelta(minutes=15)) is True


def test_calendar_blocks_weekend_as_closed_not_holiday() -> None:
    cal = ExchangeCalendar()
    saturday = datetime(2024, 5, 18, 12, 0, tzinfo=timezone.utc)
    assert cal.get_exchange_session_state(saturday) == "closed"


def test_calendar_flags_july_4_as_holiday() -> None:
    cal = ExchangeCalendar()
    july_4 = datetime(2024, 7, 4, 14, 0, tzinfo=timezone.utc)
    assert cal.get_exchange_session_state(july_4) == "holiday"
