"""Rolling return-correlation risk helpers (Phase A12).

The engine already blocks a trade that is very tightly correlated with an open
position. Here we also support *projected* correlated-exposure checks so the
risk manager can refuse a candidate before it is executed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass
class CorrelationBreach:
    """Structured correlation finding for a candidate trade."""

    symbol: str
    counterparty: str
    correlation: float
    scope: str  # "reduce" | "reject"

    def as_reason_code(self) -> str:
        return f"correlation_{self.scope}:{self.symbol}~{self.counterparty}"


def compute_rolling_return_correlation(
    returns: pd.DataFrame,
    lookback: int = 60,
) -> pd.DataFrame:
    """Return the most-recent ``lookback`` cross-correlation matrix."""
    if returns.empty:
        return pd.DataFrame()
    window = returns.tail(lookback)
    if len(window) < max(10, lookback // 4):
        return pd.DataFrame(index=returns.columns, columns=returns.columns, dtype=float)
    corr = window.corr().clip(-1.0, 1.0)
    values = corr.to_numpy(copy=True)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)


def check_correlation_limit(
    correlations: pd.DataFrame,
    *,
    candidate: str,
    open_symbols: Iterable[str],
    reduce_threshold: float = 0.70,
    reject_threshold: float = 0.85,
) -> list[CorrelationBreach]:
    breaches: list[CorrelationBreach] = []
    if correlations.empty or candidate not in correlations.columns:
        return breaches
    for other in open_symbols:
        if other == candidate or other not in correlations.index:
            continue
        value = float(correlations.loc[candidate, other])
        if abs(value) >= reject_threshold:
            breaches.append(CorrelationBreach(candidate, other, value, "reject"))
        elif abs(value) >= reduce_threshold:
            breaches.append(CorrelationBreach(candidate, other, value, "reduce"))
    return breaches


def project_joint_breach(
    *,
    candidate_allocations: Mapping[str, float],
    current_sector_exposure: Mapping[str, float],
    sector_limit: float,
    sector_of,
) -> dict[str, float]:
    """Return the sectors whose projected exposure would breach ``sector_limit``."""
    projected: dict[str, float] = {k: float(v) for k, v in current_sector_exposure.items()}
    for sym, change in candidate_allocations.items():
        bucket = sector_of(sym)
        projected[bucket] = projected.get(bucket, 0.0) + float(change)
    return {bucket: weight for bucket, weight in projected.items() if weight > sector_limit}


def resolve_joint_breach(
    candidate_allocations: Mapping[str, float],
    breaches: Mapping[str, float],
    *,
    sector_limit: float,
    sector_of,
) -> dict[str, float]:
    """Scale down symbols that contribute to each breached sector proportionally.

    Deterministic: symbols are sorted alphabetically so repeated calls yield the
    same scaled allocation map.
    """
    if not breaches:
        return {k: float(v) for k, v in candidate_allocations.items()}
    scaled = {k: float(v) for k, v in candidate_allocations.items()}
    for bucket, projected in sorted(breaches.items()):
        symbols = sorted(
            [sym for sym in scaled if sector_of(sym) == bucket],
            key=str,
        )
        contribution = sum(scaled[s] for s in symbols)
        if contribution <= 0:
            continue
        if contribution <= sector_limit:
            continue
        reduction_factor = sector_limit / contribution
        for sym in symbols:
            scaled[sym] = scaled[sym] * reduction_factor
    return scaled
