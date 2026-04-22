"""Volatility/spread-aware slippage model (Phase A11).

Used by the backtester and the risk manager's projected post-trade checks.
Deterministic: given the same inputs, the model returns the same cost so
backtests stay reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass
class SlippageModel:
    """Simple three-component slippage: base + spread-aware + volatility-aware."""

    base_pct: float = 0.0005          # 5 bps baseline
    spread_multiplier: float = 0.5    # half of the spread is captured as cost
    vol_multiplier: float = 0.10      # 10% of realized vol
    impact_coefficient: float = 0.20  # square-root market-impact term

    def estimate_slippage(
        self,
        *,
        reference_price: float,
        realized_vol: float = 0.0,
        spread_pct: float = 0.0,
        notional: float = 0.0,
        average_daily_notional: float = 0.0,
    ) -> float:
        """Return slippage cost as a fraction of ``reference_price``."""
        if reference_price <= 0:
            raise ValueError("reference_price must be > 0")
        spread_component = self.spread_multiplier * max(spread_pct, 0.0)
        vol_component = self.vol_multiplier * max(realized_vol, 0.0)
        impact_component = 0.0
        if notional > 0 and average_daily_notional > 0:
            participation = min(notional / average_daily_notional, 1.0)
            impact_component = self.impact_coefficient * sqrt(participation)
        return max(self.base_pct + spread_component + vol_component + impact_component, 0.0)

    def apply(
        self,
        *,
        reference_price: float,
        side: str,
        **kwargs,
    ) -> float:
        """Return an execution price after penalizing the reference price by slippage."""
        rate = self.estimate_slippage(reference_price=reference_price, **kwargs)
        if side.upper() == "BUY":
            return reference_price * (1.0 + rate)
        if side.upper() == "SELL":
            return reference_price * (1.0 - rate)
        raise ValueError(f"Unknown side: {side}")


def simulate_gap_fill_behavior(
    *,
    prior_close: float,
    gap_pct: float,
    stop_price: float,
    side: str = "LONG",
) -> float:
    """Return the realized fill price when an overnight gap pierces a stop."""
    if prior_close <= 0:
        raise ValueError("prior_close must be > 0")
    open_price = prior_close * (1.0 + gap_pct)
    if side.upper() == "LONG":
        if open_price <= stop_price:
            return open_price  # gap through stop -> exit at open
        return max(open_price, stop_price)
    if side.upper() == "SHORT":
        if open_price >= stop_price:
            return open_price
        return min(open_price, stop_price)
    raise ValueError(f"Unknown side: {side}")
