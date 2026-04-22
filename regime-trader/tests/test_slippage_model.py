"""Phase A11 - volatility / spread / participation-aware slippage model.

The slippage estimator must be:

- deterministic: identical inputs produce identical cost
- monotonically non-decreasing in volatility, spread, and participation
- side-symmetric: BUY adds cost, SELL subtracts it, both scaled by the rate
"""
from __future__ import annotations

import math

import pytest

from data.slippage import SlippageModel


def test_base_rate_applies_without_extras() -> None:
    model = SlippageModel(base_pct=0.0005)
    rate = model.estimate_slippage(reference_price=100.0)
    assert rate == pytest.approx(0.0005)


def test_higher_vol_increases_slippage() -> None:
    model = SlippageModel()
    low = model.estimate_slippage(reference_price=100.0, realized_vol=0.01)
    high = model.estimate_slippage(reference_price=100.0, realized_vol=0.05)
    assert high > low


def test_wider_spread_increases_slippage() -> None:
    model = SlippageModel()
    tight = model.estimate_slippage(reference_price=100.0, spread_pct=0.0005)
    wide = model.estimate_slippage(reference_price=100.0, spread_pct=0.003)
    assert wide > tight


def test_higher_participation_increases_slippage() -> None:
    model = SlippageModel()
    small = model.estimate_slippage(
        reference_price=100.0,
        notional=10_000.0,
        average_daily_notional=10_000_000.0,
    )
    big = model.estimate_slippage(
        reference_price=100.0,
        notional=1_000_000.0,
        average_daily_notional=10_000_000.0,
    )
    assert big > small


def test_participation_cost_grows_as_square_root() -> None:
    model = SlippageModel(base_pct=0.0, spread_multiplier=0.0, vol_multiplier=0.0)
    single = model.estimate_slippage(
        reference_price=100.0,
        notional=100_000.0,
        average_daily_notional=10_000_000.0,
    )
    quadruple = model.estimate_slippage(
        reference_price=100.0,
        notional=400_000.0,
        average_daily_notional=10_000_000.0,
    )
    # 4x participation -> 2x impact under the sqrt-impact model.
    assert quadruple == pytest.approx(single * 2.0, rel=1e-9)


def test_apply_buy_adds_cost_sell_subtracts_cost() -> None:
    model = SlippageModel(base_pct=0.002, spread_multiplier=0.0, vol_multiplier=0.0)
    buy = model.apply(reference_price=100.0, side="BUY")
    sell = model.apply(reference_price=100.0, side="SELL")
    assert buy == pytest.approx(100.2)
    assert sell == pytest.approx(99.8)


def test_apply_rejects_unknown_side() -> None:
    model = SlippageModel()
    with pytest.raises(ValueError, match="Unknown side"):
        model.apply(reference_price=100.0, side="HOLD")


def test_reference_price_must_be_positive() -> None:
    model = SlippageModel()
    with pytest.raises(ValueError, match="reference_price"):
        model.estimate_slippage(reference_price=0.0)


def test_participation_capped_at_100_percent() -> None:
    model = SlippageModel(base_pct=0.0, spread_multiplier=0.0, vol_multiplier=0.0, impact_coefficient=0.2)
    capped = model.estimate_slippage(
        reference_price=100.0,
        notional=5_000_000.0,
        average_daily_notional=1_000_000.0,  # 500% naive participation
    )
    # impact_coefficient * sqrt(1.0) = 0.2 because participation clamps at 1.0.
    assert capped == pytest.approx(0.2)


def test_deterministic_for_identical_inputs() -> None:
    model = SlippageModel()
    kwargs = dict(
        reference_price=100.0,
        realized_vol=0.02,
        spread_pct=0.001,
        notional=50_000.0,
        average_daily_notional=5_000_000.0,
    )
    first = model.estimate_slippage(**kwargs)
    second = model.estimate_slippage(**kwargs)
    assert first == second
    assert not math.isnan(first)
