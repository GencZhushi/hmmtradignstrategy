"""Phase A13 - rollback of the active model to its previous version.

The rollback path must:

- refuse when no history exists (no active model, or active has no previous)
- revert ``active_version`` to the predecessor
- mark the rolled-back version as ``retired``
- restore the previous model to ``active`` status
- persist across registry reloads
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.hmm_engine import ModelMetadata
from core.model_registry import (
    ModelRegistry,
    RollbackRejected,
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
    artifact = registry.root / version
    artifact.mkdir(parents=True, exist_ok=True)
    registry.register(artifact_dir=artifact, metadata=_metadata(version, **overrides))


def test_rollback_without_active_raises(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    with pytest.raises(RollbackRejected):
        registry.rollback_model()


def test_rollback_without_history_raises(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1")
    registry.promote_model("v1", enforce_comparison=False)
    with pytest.raises(RollbackRejected):
        registry.rollback_model()


def test_rollback_restores_previous_active(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", selected_bic=-500.0)
    registry.promote_model("v1", enforce_comparison=False)
    _register(registry, "v2", dataset_hash="hash-2", selected_bic=-600.0)
    registry.promote_model("v2", enforce_comparison=True)
    restored = registry.rollback_model(actor="operator")
    assert restored is not None
    assert registry.active_version == "v1"
    assert restored.status == "active"
    assert restored.promoted_by == "operator"


def test_rollback_marks_demoted_version_retired(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", selected_bic=-500.0)
    registry.promote_model("v1", enforce_comparison=False)
    _register(registry, "v2", dataset_hash="hash-2", selected_bic=-600.0)
    registry.promote_model("v2")
    registry.rollback_model()
    assert registry.get("v2").status == "retired"


def test_rollback_persists_across_reload(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", selected_bic=-500.0)
    registry.promote_model("v1", enforce_comparison=False)
    _register(registry, "v2", dataset_hash="hash-2", selected_bic=-600.0)
    registry.promote_model("v2")
    registry.rollback_model()
    reopened = ModelRegistry(tmp_path / "models")
    assert reopened.active_version == "v1"
    assert reopened.get("v1").status == "active"
    assert reopened.get("v2").status == "retired"


def test_double_rollback_raises(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", selected_bic=-500.0)
    registry.promote_model("v1", enforce_comparison=False)
    _register(registry, "v2", dataset_hash="hash-2", selected_bic=-600.0)
    registry.promote_model("v2")
    registry.rollback_model()
    # v1 has no ``previous_active`` pointer - second rollback must refuse.
    with pytest.raises(RollbackRejected):
        registry.rollback_model()


def test_rollback_clears_previous_active_pointer_on_restored_entry(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", selected_bic=-500.0)
    registry.promote_model("v1", enforce_comparison=False)
    _register(registry, "v2", dataset_hash="hash-2", selected_bic=-600.0)
    registry.promote_model("v2")
    restored = registry.rollback_model()
    # After rollback the restored model has no ``previous_active`` so further
    # rollbacks cannot cascade backwards through deleted history.
    assert restored is not None
    assert restored.previous_active is None
