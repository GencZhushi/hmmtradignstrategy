"""Phase A13 - candidate-vs-active promotion decision logic.

The registry must:

- approve the first promotion when there is no active model yet
- reject candidates trained on the same dataset_hash as the active model
- approve candidates with strictly better BIC, respecting
  ``min_improvement_bic``
- raise ``PromotionRejected`` via ``promote_model`` when governance says no
- always flip previous active models to ``fallback`` status on promotion
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.hmm_engine import ModelMetadata
from core.model_registry import (
    ModelRegistry,
    PromotionRejected,
)


def _metadata(
    version: str,
    *,
    dataset_hash: str = "hash-1",
    selected_bic: float = -500.0,
) -> ModelMetadata:
    return ModelMetadata(
        model_version=version,
        trained_at=f"2024-05-0{min(len(version), 9)}T12:00:00+00:00",
        n_states=3,
        n_samples=504,
        feature_columns=["log_return", "realized_vol_20"],
        bic_scores={3: selected_bic},
        selected_bic=selected_bic,
        log_likelihood=300.0,
        dataset_hash=dataset_hash,
        random_state=42,
        regimes=[{"regime_id": 0, "regime_name": "low_vol"}],
    )


def _register(registry: ModelRegistry, version: str, **overrides) -> None:
    meta = _metadata(version, **overrides)
    artifact = registry.root / version
    artifact.mkdir(parents=True, exist_ok=True)
    registry.register(artifact_dir=artifact, metadata=meta)


def test_first_promotion_is_always_approved(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1")
    decision = registry.compare_candidate_vs_active("v1")
    assert decision.approved is True
    assert decision.active_version is None
    assert "no active model" in decision.reason


def test_promote_model_sets_active_status_and_metadata(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1")
    promoted = registry.promote_model("v1", actor="alice", enforce_comparison=False)
    assert promoted.status == "active"
    assert promoted.promoted_by == "alice"
    assert promoted.promoted_at is not None
    assert registry.active_version == "v1"


def test_same_dataset_hash_is_rejected(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", dataset_hash="hash-x")
    registry.promote_model("v1", enforce_comparison=False)
    _register(registry, "v2", dataset_hash="hash-x", selected_bic=-700.0)
    decision = registry.compare_candidate_vs_active("v2")
    assert decision.approved is False
    assert decision.reason.startswith("candidate trained on same dataset")


def test_better_bic_is_approved(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", selected_bic=-500.0)
    registry.promote_model("v1", enforce_comparison=False)
    _register(registry, "v2", dataset_hash="hash-2", selected_bic=-600.0)
    decision = registry.compare_candidate_vs_active("v2")
    assert decision.approved is True
    assert decision.delta_bic == pytest.approx(100.0)


def test_worse_bic_is_rejected(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", selected_bic=-500.0)
    registry.promote_model("v1", enforce_comparison=False)
    _register(registry, "v2", dataset_hash="hash-2", selected_bic=-400.0)
    decision = registry.compare_candidate_vs_active("v2")
    assert decision.approved is False
    assert decision.delta_bic == pytest.approx(-100.0)


def test_min_improvement_threshold_gates_promotion(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", selected_bic=-500.0)
    registry.promote_model("v1", enforce_comparison=False)
    _register(registry, "v2", dataset_hash="hash-2", selected_bic=-510.0)
    decision = registry.compare_candidate_vs_active("v2", min_improvement_bic=20.0)
    assert decision.approved is False  # 10 < 20 threshold
    decision_loose = registry.compare_candidate_vs_active("v2", min_improvement_bic=5.0)
    assert decision_loose.approved is True


def test_promote_model_raises_when_governance_rejects(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", selected_bic=-500.0)
    registry.promote_model("v1", enforce_comparison=False)
    _register(registry, "v2", dataset_hash="hash-2", selected_bic=-400.0)
    with pytest.raises(PromotionRejected):
        registry.promote_model("v2")


def test_previous_active_is_demoted_to_fallback(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", selected_bic=-500.0)
    registry.promote_model("v1", enforce_comparison=False)
    _register(registry, "v2", dataset_hash="hash-2", selected_bic=-600.0)
    registry.promote_model("v2", enforce_comparison=True)
    assert registry.get("v1").status == "fallback"
    assert registry.get("v2").status == "active"
    assert registry.get("v2").previous_active == "v1"


def test_unknown_candidate_version_raises_keyerror(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    with pytest.raises(KeyError):
        registry.compare_candidate_vs_active("does-not-exist")
