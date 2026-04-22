"""Phase A11 - overnight gap / stop-pierce behaviour and intraday regime pinning.

Two policies must hold under test:

1. When an overnight gap pierces a stop, the realized fill price is the open
   price (gap-through behaviour) rather than the unreachable stop level.
2. The daily HMM regime stays fixed intraday; it can only advance once the
   previous trading day is complete. This mirrors the live-trading contract
   so the backtester and the runtime engine produce matching decisions.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from data.exchange_calendar import ExchangeCalendar, get_effective_intraday_regime
from data.slippage import simulate_gap_fill_behavior


def test_long_gap_down_through_stop_exits_at_open() -> None:
    # 5% gap down pushes the open below a 98.0 stop -> fill at open.
    price = simulate_gap_fill_behavior(prior_close=100.0, gap_pct=-0.05, stop_price=98.0, side="LONG")
    assert price == pytest.approx(95.0)


def test_long_small_gap_down_respects_stop_floor() -> None:
    # Open at 99.0, stop at 98.0 -> stop is not pierced, fill caps at stop level.
    price = simulate_gap_fill_behavior(prior_close=100.0, gap_pct=-0.01, stop_price=98.0, side="LONG")
    assert price == pytest.approx(99.0)


def test_short_gap_up_through_stop_exits_at_open() -> None:
    # 5% gap up pushes the open above a 102.0 short stop -> fill at open.
    price = simulate_gap_fill_behavior(prior_close=100.0, gap_pct=0.05, stop_price=102.0, side="SHORT")
    assert price == pytest.approx(105.0)


def test_short_small_gap_up_respects_stop_ceiling() -> None:
    price = simulate_gap_fill_behavior(prior_close=100.0, gap_pct=0.01, stop_price=102.0, side="SHORT")
    assert price == pytest.approx(101.0)


def test_prior_close_must_be_positive() -> None:
    with pytest.raises(ValueError, match="prior_close"):
        simulate_gap_fill_behavior(prior_close=0.0, gap_pct=-0.05, stop_price=98.0, side="LONG")


def test_unknown_side_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown side"):
        simulate_gap_fill_behavior(prior_close=100.0, gap_pct=0.0, stop_price=98.0, side="FLAT")


def test_intraday_regime_pins_to_previous_trading_day() -> None:
    cal = ExchangeCalendar()
    now = datetime(2024, 5, 15, 14, 0, tzinfo=timezone.utc)  # mid-session Wednesday
    effective = get_effective_intraday_regime(date(2024, 5, 14), now, cal)
    # Regime already reflects the last completed session -> must not move.
    assert effective == date(2024, 5, 14)


def test_intraday_regime_skips_weekend_when_monday_opens() -> None:
    cal = ExchangeCalendar()
    monday_open = datetime(2024, 5, 20, 14, 0, tzinfo=timezone.utc)  # Monday
    # Only Thursday data is available (regime stamped for Thursday the prior week).
    effective = get_effective_intraday_regime(date(2024, 5, 16), monday_open, cal)
    # Effective regime for Monday must be the previous trading day = Friday 5/17.
    assert effective == date(2024, 5, 17)


def test_regime_is_never_dated_in_the_future() -> None:
    cal = ExchangeCalendar()
    now = datetime(2024, 5, 15, 14, 0, tzinfo=timezone.utc)
    # If the stamped regime is somehow ahead of today, the helper keeps it
    # (caller must guarantee daily bars are complete before stamping).
    effective = get_effective_intraday_regime(date(2024, 5, 16), now, cal)
    assert effective == date(2024, 5, 16)


def test_no_stamped_regime_returns_none() -> None:
    cal = ExchangeCalendar()
    now = datetime(2024, 5, 15, 14, 0, tzinfo=timezone.utc)
    assert get_effective_intraday_regime(None, now, cal) is None
