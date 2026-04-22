"""Phase A3 / A13 - HMM training, BIC selection, persistence, registry."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.hmm_engine import VolatilityRegimeHMM
from core.model_registry import (
    ModelRegistry,
    PromotionRejected,
    RollbackRejected,
    register_model_version,
)
from data.feature_engineering import FeatureEngine

# hmmlearn is an optional scientific dependency; every test in this module
# fits a real HMM and therefore cannot run when the library is missing. Skip
# cleanly in that case so CI on minimal environments stays green without
# losing phase-level coverage when hmmlearn IS installed.
pytest.importorskip("hmmlearn")


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory: pytest.TempPathFactory, synthetic_ohlcv_session) -> VolatilityRegimeHMM:
    engine = FeatureEngine(zscore_window=252)
    features = engine.build_daily_features(synthetic_ohlcv_session).iloc[:504]
    model = VolatilityRegimeHMM(n_candidates=(3, 4), n_init=2, random_state=7)
    model.fit(features)
    return model


@pytest.fixture(scope="module")
def synthetic_ohlcv_session() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n = 900
    drift = np.where(np.arange(n) // 200 % 2 == 0, 0.0005, -0.0002)
    scale = np.where(np.arange(n) // 200 % 2 == 0, 0.008, 0.02)
    rets = rng.normal(drift, scale)
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 5_000_000, n)
    idx = pd.bdate_range("2019-01-02", periods=n)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([open_, close, high]),
            "low": np.minimum.reduce([open_, close, low]),
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def test_bic_model_selection_covers_candidates(trained_model: VolatilityRegimeHMM) -> None:
    meta = trained_model.metadata
    assert meta.bic_scores, "BIC scores must be recorded for all candidates"
    assert set(meta.bic_scores.keys()) == {3, 4}
    assert meta.selected_bic == min(meta.bic_scores.values())


def test_filtered_probabilities_sum_to_one(trained_model: VolatilityRegimeHMM, synthetic_ohlcv_session: pd.DataFrame) -> None:
    engine = FeatureEngine(zscore_window=252)
    features = engine.build_daily_features(synthetic_ohlcv_session)
    proba = trained_model.filtered_probabilities(features)
    sums = proba.sum(axis=1).to_numpy()
    np.testing.assert_allclose(sums, np.ones(len(sums)), atol=1e-9)
    assert proba.shape[1] == trained_model.metadata.n_states


def test_predict_is_not_available(trained_model: VolatilityRegimeHMM, synthetic_ohlcv_session: pd.DataFrame) -> None:
    engine = FeatureEngine(zscore_window=252)
    features = engine.build_daily_features(synthetic_ohlcv_session)
    with pytest.raises(NotImplementedError):
        trained_model.predict_proba(features)


def test_regime_labels_follow_return_rank(trained_model: VolatilityRegimeHMM) -> None:
    regimes = trained_model.regimes
    by_return = sorted(regimes, key=lambda r: r.expected_return)
    for rank, regime in enumerate(by_return):
        assert regime.label_return_rank == rank


def test_model_persists_and_reloads(tmp_path: Path, trained_model: VolatilityRegimeHMM, synthetic_ohlcv_session: pd.DataFrame) -> None:
    artifact = trained_model.save_model(tmp_path)
    reloaded = VolatilityRegimeHMM().load_model(artifact)
    assert reloaded.metadata.n_states == trained_model.metadata.n_states
    assert reloaded.metadata.dataset_hash == trained_model.metadata.dataset_hash
    engine = FeatureEngine(zscore_window=252)
    features = engine.build_daily_features(synthetic_ohlcv_session)
    original = trained_model.filtered_probabilities(features).to_numpy()
    after = reloaded.filtered_probabilities(features).to_numpy()
    np.testing.assert_allclose(original, after, atol=1e-9)


def test_registry_registers_and_promotes(tmp_path: Path, trained_model: VolatilityRegimeHMM) -> None:
    registry = ModelRegistry(tmp_path / "models")
    entry = register_model_version(registry, trained_model, notes="unit test")
    assert entry.model_version in {e.model_version for e in registry.list_versions()}
    promoted = registry.promote_model(entry.model_version, actor="test", enforce_comparison=False)
    assert registry.active_version == promoted.model_version


def test_registry_rejects_worse_candidate(tmp_path: Path, trained_model: VolatilityRegimeHMM) -> None:
    registry = ModelRegistry(tmp_path / "models")
    first = register_model_version(registry, trained_model)
    registry.promote_model(first.model_version, enforce_comparison=False)

    # Copy first artifact on disk but register a worse BIC value for the candidate.
    worse = registry.register(
        artifact_dir=Path(first.path),
        metadata=trained_model.metadata.__class__(
            **{**trained_model.metadata.__dict__, "model_version": "worse-candidate", "selected_bic": trained_model.metadata.selected_bic + 100.0, "dataset_hash": "different-hash"},
        ),
    )
    with pytest.raises(PromotionRejected):
        registry.promote_model(worse.model_version, enforce_comparison=True, min_improvement_bic=0.0)


def test_registry_rollback(tmp_path: Path, trained_model: VolatilityRegimeHMM) -> None:
    registry = ModelRegistry(tmp_path / "models")
    first = register_model_version(registry, trained_model)
    registry.promote_model(first.model_version, enforce_comparison=False)
    second_meta = trained_model.metadata.__class__(
        **{**trained_model.metadata.__dict__, "model_version": "second-version", "dataset_hash": "hash-2", "selected_bic": trained_model.metadata.selected_bic - 10.0},
    )
    second = registry.register(artifact_dir=Path(first.path), metadata=second_meta)
    registry.promote_model(second.model_version, enforce_comparison=False)
    rolled = registry.rollback_model(actor="test")
    assert rolled.model_version == first.model_version
    assert registry.active_version == first.model_version


def test_registry_rollback_requires_history(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    with pytest.raises(RollbackRejected):
        registry.rollback_model()


def test_fallback_returns_previous_active(tmp_path: Path, trained_model: VolatilityRegimeHMM) -> None:
    registry = ModelRegistry(tmp_path / "models")
    first = register_model_version(registry, trained_model)
    registry.promote_model(first.model_version, enforce_comparison=False)
    fallback = registry.fallback_entry()
    assert fallback is not None
    assert fallback.model_version == first.model_version
