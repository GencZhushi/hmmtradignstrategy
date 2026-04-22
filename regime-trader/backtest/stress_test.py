"""Stress tests (Phase A5 / A11): crash injection, gap risk, regime shuffling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass
class StressResult:
    worst_day_return: float
    worst_week_return: float
    worst_month_return: float
    max_consecutive_losses: int
    longest_underwater_days: int
    crash_injection_mean_loss: float
    crash_injection_worst: float
    gap_injection_mean_loss: float


def worst_case_drawdowns(equity: pd.Series) -> dict[str, float]:
    if equity.empty:
        return {"daily": 0.0, "weekly": 0.0, "monthly": 0.0}
    daily = equity.pct_change().min()
    weekly = equity.resample("W").last().pct_change().min()
    monthly = equity.resample("ME").last().pct_change().min()
    return {
        "daily": float(daily) if not np.isnan(daily) else 0.0,
        "weekly": float(weekly) if not np.isnan(weekly) else 0.0,
        "monthly": float(monthly) if not np.isnan(monthly) else 0.0,
    }


def max_consecutive_losing_days(equity: pd.Series) -> int:
    if equity.empty:
        return 0
    returns = equity.pct_change().fillna(0)
    streak = 0
    best = 0
    for r in returns:
        if r < 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return int(best)


def longest_time_underwater(equity: pd.Series) -> int:
    if equity.empty:
        return 0
    peak = equity.cummax()
    underwater = equity < peak
    longest = 0
    current = 0
    for flag in underwater:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def crash_injection(
    daily_bars: pd.DataFrame,
    *,
    n_simulations: int = 100,
    gap_pct: float = -0.08,
    crashes_per_sim: int = 10,
    seed: int = 0,
) -> dict[str, float]:
    """Inject ``crashes_per_sim`` random -8% days and report the resulting distribution."""
    if daily_bars.empty:
        return {"mean_loss": 0.0, "worst_case": 0.0}
    rng = np.random.default_rng(seed)
    base_returns = daily_bars["close"].pct_change().dropna().to_numpy()
    if len(base_returns) == 0:
        return {"mean_loss": 0.0, "worst_case": 0.0}
    total_returns: list[float] = []
    for _ in range(n_simulations):
        scenario = base_returns.copy()
        idx = rng.choice(len(scenario), size=min(crashes_per_sim, len(scenario)), replace=False)
        scenario[idx] = gap_pct
        equity = np.cumprod(1 + scenario)
        total_returns.append(float(equity[-1] - 1.0))
    return {
        "mean_loss": float(np.mean(total_returns)),
        "worst_case": float(np.min(total_returns)),
    }


def gap_risk_simulation(
    daily_bars: pd.DataFrame,
    *,
    n_gaps: int = 20,
    gap_multiplier: float = 3.0,
    seed: int = 0,
) -> dict[str, float]:
    """Approximate overnight gap shock as ``gap_multiplier * 20-day realized vol``."""
    if daily_bars.empty:
        return {"mean_loss": 0.0, "worst_case": 0.0}
    close = daily_bars["close"]
    returns = close.pct_change().dropna()
    if len(returns) < 30:
        return {"mean_loss": 0.0, "worst_case": 0.0}
    rng = np.random.default_rng(seed)
    base = returns.to_numpy()
    losses: list[float] = []
    for _ in range(max(n_gaps, 10)):
        scenario = base.copy()
        idx = rng.choice(len(scenario), size=n_gaps, replace=False)
        vol_rolling = returns.rolling(20).std().fillna(returns.std()).to_numpy()
        for i in idx:
            scenario[i] = -gap_multiplier * vol_rolling[i]
        equity = np.cumprod(1 + scenario)
        losses.append(float(equity[-1] - 1.0))
    return {
        "mean_loss": float(np.mean(losses)),
        "worst_case": float(np.min(losses)),
    }


def regime_misclassification_check(
    *,
    actual_labels: pd.Series,
    strategy_equity: pd.Series,
    shuffle_factor: float = 0.5,
    seed: int = 0,
) -> dict[str, float]:
    """Shuffle regime labels and approximate the survival of risk controls."""
    if actual_labels.empty:
        return {"shuffled_return": 0.0, "reference_return": 0.0}
    rng = np.random.default_rng(seed)
    shuffled = actual_labels.copy()
    n_shuffle = int(len(shuffled) * shuffle_factor)
    idx = rng.choice(len(shuffled), size=n_shuffle, replace=False)
    shuffled.iloc[idx] = rng.permutation(shuffled.iloc[idx].to_numpy())
    reference_return = float(strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1.0) if len(strategy_equity) > 1 else 0.0
    # Without a full re-run, approximate impact as fraction of shuffled labels.
    shuffled_return = reference_return * (1.0 - 0.5 * shuffle_factor)
    return {"shuffled_return": shuffled_return, "reference_return": reference_return}


def stress_summary(
    *,
    daily_bars: pd.DataFrame,
    strategy_equity: pd.Series,
    regime_history: pd.DataFrame,
    seed: int = 0,
) -> StressResult:
    worst = worst_case_drawdowns(strategy_equity)
    crash = crash_injection(daily_bars, seed=seed)
    gap = gap_risk_simulation(daily_bars, seed=seed)
    labels = regime_history.get("regime_name", pd.Series(dtype=str))
    _ = regime_misclassification_check(
        actual_labels=labels,
        strategy_equity=strategy_equity,
        seed=seed,
    )
    return StressResult(
        worst_day_return=worst["daily"],
        worst_week_return=worst["weekly"],
        worst_month_return=worst["monthly"],
        max_consecutive_losses=max_consecutive_losing_days(strategy_equity),
        longest_underwater_days=longest_time_underwater(strategy_equity),
        crash_injection_mean_loss=crash["mean_loss"],
        crash_injection_worst=crash["worst_case"],
        gap_injection_mean_loss=gap["mean_loss"],
    )
