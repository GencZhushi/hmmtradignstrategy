"""Volatility-rank-driven allocation strategies (Spec A3).

Design rules:

- Always long or flat; never short.
- Low vol -> highest allocation; high vol -> reduced but still long.
- Labels like "BULL" are for humans; the orchestrator sorts regimes by
  ``expected_volatility`` independently when selecting a strategy.
- Uncertainty mode halves size and forces leverage to 1.0x.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from core.hmm_engine import RegimeInfo
from core.types import Direction, Signal

LOG = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Runtime knobs matching ``strategy:`` in ``settings.yaml``."""

    low_vol_allocation: float = 0.95
    mid_vol_allocation_trend: float = 0.95
    mid_vol_allocation_no_trend: float = 0.60
    high_vol_allocation: float = 0.60
    low_vol_leverage: float = 1.25
    rebalance_threshold: float = 0.10
    uncertainty_size_mult: float = 0.50

    @classmethod
    def from_config(cls, cfg: Mapping[str, float]) -> "StrategyConfig":
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in cfg.items() if k in fields})


class BaseStrategy(ABC):
    name: str = "base"

    def __init__(self, config: StrategyConfig):
        self.config = config

    @abstractmethod
    def generate_signal(
        self,
        *,
        symbol: str,
        bars: pd.DataFrame,
        regime_info: RegimeInfo,
        regime_probability: float,
    ) -> Signal | None: ...

    def _ema(self, close: pd.Series, span: int) -> float:
        return float(close.ewm(span=span, adjust=False).mean().iloc[-1])

    def _atr(self, bars: pd.DataFrame, period: int = 14) -> float:
        high = bars["high"]
        low = bars["low"]
        close = bars["close"]
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])


class LowVolBullStrategy(BaseStrategy):
    name = "LowVolBullStrategy"

    def generate_signal(self, *, symbol, bars, regime_info, regime_probability):  # type: ignore[override]
        if len(bars) < 50:
            return None
        close = bars["close"]
        price = float(close.iloc[-1])
        ema50 = self._ema(close, 50)
        atr = self._atr(bars, 14)
        stop = max(price - 3.0 * atr, ema50 - 0.5 * atr)
        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            target_allocation_pct=self.config.low_vol_allocation,
            leverage=self.config.low_vol_leverage,
            entry_price=price,
            stop_loss=stop,
            regime_id=regime_info.regime_id,
            regime_name=regime_info.regime_name,
            regime_probability=regime_probability,
            confidence=regime_probability,
            strategy_name=self.name,
            reasoning=[f"Low-vol regime {regime_info.regime_name}: full allocation + modest leverage"],
            metadata={"atr_14": atr, "ema_50": ema50},
        )


class MidVolCautiousStrategy(BaseStrategy):
    name = "MidVolCautiousStrategy"

    def generate_signal(self, *, symbol, bars, regime_info, regime_probability):  # type: ignore[override]
        if len(bars) < 50:
            return None
        close = bars["close"]
        price = float(close.iloc[-1])
        ema50 = self._ema(close, 50)
        atr = self._atr(bars, 14)
        trend_intact = price > ema50
        allocation = (
            self.config.mid_vol_allocation_trend
            if trend_intact
            else self.config.mid_vol_allocation_no_trend
        )
        stop = ema50 - 0.5 * atr
        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            target_allocation_pct=allocation,
            leverage=1.0,
            entry_price=price,
            stop_loss=stop,
            regime_id=regime_info.regime_id,
            regime_name=regime_info.regime_name,
            regime_probability=regime_probability,
            confidence=regime_probability,
            strategy_name=self.name,
            reasoning=[
                f"Mid-vol regime {regime_info.regime_name}: "
                f"{'trend intact' if trend_intact else 'trend broken'} -> allocation {allocation:.0%}"
            ],
            metadata={"atr_14": atr, "ema_50": ema50, "trend_intact": trend_intact},
        )


class HighVolDefensiveStrategy(BaseStrategy):
    name = "HighVolDefensiveStrategy"

    def generate_signal(self, *, symbol, bars, regime_info, regime_probability):  # type: ignore[override]
        if len(bars) < 50:
            return None
        close = bars["close"]
        price = float(close.iloc[-1])
        ema50 = self._ema(close, 50)
        atr = self._atr(bars, 14)
        stop = ema50 - 1.0 * atr
        return Signal(
            symbol=symbol,
            direction=Direction.LONG,
            target_allocation_pct=self.config.high_vol_allocation,
            leverage=1.0,
            entry_price=price,
            stop_loss=stop,
            regime_id=regime_info.regime_id,
            regime_name=regime_info.regime_name,
            regime_probability=regime_probability,
            confidence=regime_probability,
            strategy_name=self.name,
            reasoning=[f"High-vol regime {regime_info.regime_name}: reduced allocation, still long"],
            metadata={"atr_14": atr, "ema_50": ema50},
        )


# Backward-compatible aliases for readability in prompts/agents.
CrashDefensiveStrategy = HighVolDefensiveStrategy
BearTrendStrategy = HighVolDefensiveStrategy
MeanReversionStrategy = MidVolCautiousStrategy
BullTrendStrategy = LowVolBullStrategy
EuphoriaCautiousStrategy = LowVolBullStrategy


LABEL_TO_STRATEGY: dict[str, type[BaseStrategy]] = {
    "CRASH": HighVolDefensiveStrategy,
    "STRONG_BEAR": HighVolDefensiveStrategy,
    "BEAR": HighVolDefensiveStrategy,
    "WEAK_BEAR": MidVolCautiousStrategy,
    "NEUTRAL": MidVolCautiousStrategy,
    "WEAK_BULL": MidVolCautiousStrategy,
    "BULL": LowVolBullStrategy,
    "STRONG_BULL": LowVolBullStrategy,
    "EUPHORIA": LowVolBullStrategy,
}


@dataclass
class StrategyOrchestrator:
    """Maps ``regime_id`` to a strategy based on ascending volatility rank."""

    config: StrategyConfig
    regime_infos: list[RegimeInfo] = field(default_factory=list)
    _strategy_by_regime: dict[int, BaseStrategy] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.regime_infos:
            self.update_regime_infos(self.regime_infos)

    def update_regime_infos(self, regime_infos: Iterable[RegimeInfo]) -> None:
        infos = list(regime_infos)
        if not infos:
            self._strategy_by_regime = {}
            return
        ordered = sorted(infos, key=lambda r: r.expected_volatility)
        n = len(ordered)
        mapping: dict[int, BaseStrategy] = {}
        for rank, regime in enumerate(ordered):
            position = rank / max(n - 1, 1)
            if position <= 0.33:
                mapping[regime.regime_id] = LowVolBullStrategy(self.config)
            elif position >= 0.67:
                mapping[regime.regime_id] = HighVolDefensiveStrategy(self.config)
            else:
                mapping[regime.regime_id] = MidVolCautiousStrategy(self.config)
        self.regime_infos = infos
        self._strategy_by_regime = mapping

    def strategy_for(self, regime_id: int) -> BaseStrategy:
        if regime_id not in self._strategy_by_regime:
            raise KeyError(f"No strategy registered for regime_id={regime_id}")
        return self._strategy_by_regime[regime_id]

    def generate_signals(
        self,
        *,
        symbols: Sequence[str],
        bars_by_symbol: Mapping[str, pd.DataFrame],
        regime_id: int,
        regime_probability: float,
        is_flickering: bool,
    ) -> list[Signal]:
        if regime_id not in self._strategy_by_regime:
            raise KeyError(f"Unknown regime_id {regime_id}")
        regime_info = next(r for r in self.regime_infos if r.regime_id == regime_id)
        strategy = self._strategy_by_regime[regime_id]
        out: list[Signal] = []
        for symbol in symbols:
            bars = bars_by_symbol.get(symbol)
            if bars is None or bars.empty:
                continue
            signal = strategy.generate_signal(
                symbol=symbol,
                bars=bars,
                regime_info=regime_info,
                regime_probability=regime_probability,
            )
            if signal is None:
                continue
            if is_flickering or regime_probability < regime_info.min_confidence_to_act:
                signal = apply_uncertainty_mode(signal, self.config)
            out.append(signal)
        return out


def apply_uncertainty_mode(signal: Signal, config: StrategyConfig) -> Signal:
    return signal.with_modifications(
        target_allocation_pct=signal.target_allocation_pct * config.uncertainty_size_mult,
        leverage=min(signal.leverage, 1.0),
        reasoning=signal.reasoning + ["UNCERTAINTY - size halved"],
        metadata={**signal.metadata, "uncertainty_mode": True},
    )


def generate_target_allocation(
    *,
    current_alloc_pct: float,
    target_alloc_pct: float,
    rebalance_threshold: float,
) -> float:
    """Return the target allocation to submit after filtering by threshold."""
    if abs(target_alloc_pct - current_alloc_pct) < rebalance_threshold:
        return current_alloc_pct
    return target_alloc_pct
