"""Combines HMM regime state with the strategy orchestrator to produce Signals.

This module owns the daily-regime/intraday-execution timing rule:

- The HMM is invoked only with *completed* daily feature rows.
- The resulting daily regime is cached and reused across intraday calls until a
  new completed daily bar arrives.
- Flicker/confidence diagnostics feed the orchestrator's uncertainty mode.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from core.hmm_engine import RegimeState, VolatilityRegimeHMM
from core.regime_strategies import StrategyOrchestrator
from core.types import Signal

LOG = logging.getLogger(__name__)


@dataclass
class RegimeDiagnostics:
    """Per-symbol runtime diagnostics shared with the API/agent layer."""

    state: RegimeState | None = None
    is_flickering: bool = False
    flicker_rate: float = 0.0
    consecutive_bars: int = 0
    last_regime_id: int | None = None
    last_update_session_date: date | None = None


@dataclass
class SignalGenerator:
    """Bridge between the HMM (daily) and strategy orchestrator (signals)."""

    hmm: VolatilityRegimeHMM
    orchestrator: StrategyOrchestrator
    stability_bars: int = 3
    flicker_window: int = 20
    flicker_threshold: int = 4
    min_confidence: float = 0.55
    _recent_states: deque[int] = field(default_factory=lambda: deque(maxlen=20), init=False)
    _current_state_id: int | None = field(default=None, init=False)
    _consecutive_bars: int = field(default=0, init=False)
    _candidate_state_id: int | None = field(default=None, init=False)
    _candidate_run_length: int = field(default=0, init=False)
    _last_effective_date: date | None = field(default=None, init=False)
    diagnostics: RegimeDiagnostics = field(default_factory=RegimeDiagnostics)

    # -------------------------------------------------- regime update
    def on_new_daily_features(
        self,
        features: pd.DataFrame,
        *,
        as_of: datetime | None = None,
    ) -> RegimeState:
        """Update the cached regime state with the latest completed daily features."""
        if features.empty:
            raise ValueError("No features provided for HMM update")
        probabilities = self.hmm.filtered_probabilities(features)
        proba_row = probabilities.iloc[-1].to_numpy()
        new_state = int(np.argmax(proba_row))
        now = as_of or datetime.now(timezone.utc)
        self._update_flicker_window(new_state)
        confirmed = self._apply_stability_filter(new_state)
        regime_info = self.hmm.regimes[new_state]
        state = RegimeState(
            regime_id=self._current_state_id if self._current_state_id is not None else new_state,
            regime_name=regime_info.regime_name,
            probability=float(proba_row[new_state]),
            state_probabilities=[float(p) for p in proba_row],
            timestamp=pd.Timestamp(now),
            is_confirmed=confirmed,
            consecutive_bars=self._consecutive_bars,
            flicker_rate=self._flicker_rate(),
        )
        self.diagnostics = RegimeDiagnostics(
            state=state,
            is_flickering=self._is_flickering(),
            flicker_rate=state.flicker_rate,
            consecutive_bars=state.consecutive_bars,
            last_regime_id=state.regime_id,
            last_update_session_date=features.index[-1].date(),
        )
        self._last_effective_date = features.index[-1].date()
        return state

    # -------------------------------------------------- signal production
    def generate_signals(
        self,
        *,
        symbols: Sequence[str],
        bars_by_symbol: Mapping[str, pd.DataFrame],
    ) -> list[Signal]:
        if self.diagnostics.state is None:
            LOG.debug("No regime state yet; skipping signal generation")
            return []
        state = self.diagnostics.state
        return self.orchestrator.generate_signals(
            symbols=symbols,
            bars_by_symbol=bars_by_symbol,
            regime_id=state.regime_id,
            regime_probability=state.probability,
            is_flickering=self.diagnostics.is_flickering,
        )

    # -------------------------------------------------- diagnostics helpers
    def _update_flicker_window(self, new_state: int) -> None:
        self._recent_states.append(new_state)
        if len(self._recent_states) > self.flicker_window:
            self._recent_states.popleft()

    def _apply_stability_filter(self, new_state: int) -> bool:
        if self._current_state_id is None:
            self._current_state_id = new_state
            self._consecutive_bars = 1
            self._candidate_state_id = None
            self._candidate_run_length = 0
            return True
        if new_state == self._current_state_id:
            self._consecutive_bars += 1
            self._candidate_state_id = None
            self._candidate_run_length = 0
            return True
        if self._candidate_state_id == new_state:
            self._candidate_run_length += 1
        else:
            self._candidate_state_id = new_state
            self._candidate_run_length = 1
        if self._candidate_run_length >= self.stability_bars:
            LOG.warning(
                "Regime change confirmed %s -> %s",
                self._current_state_id,
                new_state,
            )
            self._current_state_id = new_state
            self._consecutive_bars = self._candidate_run_length
            self._candidate_state_id = None
            self._candidate_run_length = 0
            return True
        return False

    def _flicker_rate(self) -> float:
        if len(self._recent_states) < 2:
            return 0.0
        changes = sum(1 for prev, cur in zip(self._recent_states, list(self._recent_states)[1:]) if prev != cur)
        return changes / max(len(self._recent_states) - 1, 1)

    def _is_flickering(self) -> bool:
        if not self._recent_states:
            return False
        changes = sum(
            1
            for prev, cur in zip(self._recent_states, list(self._recent_states)[1:])
            if prev != cur
        )
        return changes >= self.flicker_threshold

    # -------------------------------------------------- accessors
    @property
    def current_state_id(self) -> int | None:
        return self._current_state_id

    @property
    def last_effective_date(self) -> date | None:
        return self._last_effective_date

    def reset(self) -> None:
        self._recent_states.clear()
        self._current_state_id = None
        self._consecutive_bars = 0
        self._candidate_state_id = None
        self._candidate_run_length = 0
        self._last_effective_date = None
        self.diagnostics = RegimeDiagnostics()
