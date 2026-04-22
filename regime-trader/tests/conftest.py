"""Test fixtures shared across the suite."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture()
def synthetic_daily_ohlcv() -> pd.DataFrame:
    """Deterministic synthetic daily OHLCV with ~1500 trading days and two vol regimes.

    The feature pipeline layers a 252-bar rolling z-score on top of indicators
    that themselves already roll over ~252 bars (e.g. ``rsi_z``), so roughly
    504 bars of warmup are consumed before the first complete feature row
    appears. 1500 bars leave >=504 HMM training bars for look-ahead tests and
    generous overlap for prefix/causality assertions.
    """
    rng = np.random.default_rng(42)
    n = 1500
    # Build two alternating vol regimes: low vol (0.7%) and high vol (2.2%)
    regimes = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        block = rng.integers(60, 180)
        end = min(i + block, n)
        regimes[i:end] = 0 if (i // max(block, 1)) % 2 == 0 else 1
        i = end
    daily_vol = np.where(regimes == 0, 0.007, 0.022)
    drift = np.where(regimes == 0, 0.0005, -0.0002)
    returns = rng.normal(loc=drift, scale=daily_vol)
    close = 100.0 * np.exp(np.cumsum(returns))
    intra_range = daily_vol * 1.5
    high = close * (1 + rng.uniform(0, intra_range))
    low = close * (1 - rng.uniform(0, intra_range))
    open_ = close * (1 + rng.normal(0, daily_vol / 2))
    volume = rng.integers(1_000_000, 5_000_000, size=n)
    idx = pd.bdate_range("2019-01-02", periods=n)
    df = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([open_, close, high]),
            "low": np.minimum.reduce([open_, close, low]),
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    df.index.name = "date"
    return df


@pytest.fixture()
def tmp_state_dir(tmp_path: Path) -> Path:
    state = tmp_path / "state"
    (state / "models").mkdir(parents=True)
    (state / "audit").mkdir(parents=True)
    (state / "snapshots").mkdir(parents=True)
    (state / "approvals").mkdir(parents=True)
    return state
