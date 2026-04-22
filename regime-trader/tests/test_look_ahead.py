"""MUST: feature pipeline + HMM filtered inference must not peek at future data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import VolatilityRegimeHMM
from data.feature_engineering import FeatureEngine, detect_future_leakage


def test_daily_features_do_not_use_future_data(synthetic_daily_ohlcv: pd.DataFrame) -> None:
    engine = FeatureEngine(zscore_window=252)
    full = engine.build_daily_features(synthetic_daily_ohlcv).to_numpy()
    prefix = engine.build_daily_features(synthetic_daily_ohlcv.iloc[:600]).to_numpy()
    assert not detect_future_leakage([full, prefix]), "features must match on overlap"


@pytest.mark.slow
def test_filtered_hmm_regimes_match_on_prefix(synthetic_daily_ohlcv: pd.DataFrame) -> None:
    # hmmlearn is an optional scientific dependency. Skip cleanly when it is
    # absent so minimal CI environments stay green; this still runs the
    # no-look-ahead contract when the library is available.
    pytest.importorskip("hmmlearn")
    engine = FeatureEngine(zscore_window=252)
    features = engine.build_daily_features(synthetic_daily_ohlcv)
    assert len(features) >= 504, "need at least 504 completed bars for HMM training"
    train = features.iloc[:504]

    model = VolatilityRegimeHMM(n_candidates=(3,), n_init=2, random_state=0)
    model.fit(train)

    proba_short = model.filtered_probabilities(features.iloc[:600])
    proba_long = model.filtered_probabilities(features.iloc[:700])
    length = len(proba_short)

    short_states = np.argmax(proba_short.to_numpy(), axis=1)
    long_states = np.argmax(proba_long.to_numpy()[:length], axis=1)
    np.testing.assert_array_equal(short_states, long_states)

    np.testing.assert_allclose(
        proba_short.to_numpy(),
        proba_long.to_numpy()[:length],
        rtol=1e-6,
        atol=1e-6,
    )
