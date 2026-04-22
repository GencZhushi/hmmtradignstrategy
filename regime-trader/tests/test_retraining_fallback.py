"""Phase A13 - fallback model lookup when retraining fails.

If retraining fails or produces a rejected candidate, the engine must keep
trading on the last-known-good model. The registry surfaces this via:

- ``active_entry()`` : the model currently used for inference
- ``fallback_entry()`` : the most recent active/fallback model distinct from
  the live one, used when the active version artifact becomes unusable
- promotion policy must never silently down-rank a working active model
"""
from __future__ import annotations

from pathlib import Path

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
    trained_at: str = "2024-05-01T12:00:00+00:00",
) -> ModelMetadata:
    return ModelMetadata(
        model_version=version,
        trained_at=trained_at,
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


def test_fallback_is_none_when_registry_empty(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    assert registry.fallback_entry() is None


def test_fallback_returns_active_when_only_one_model_exists(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1")
    registry.promote_model("v1", enforce_comparison=False)
    entry = registry.fallback_entry()
    assert entry is not None and entry.model_version == "v1"


def test_fallback_returns_most_recent_non_active_model(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", selected_bic=-500.0, trained_at="2024-01-01T00:00:00+00:00")
    registry.promote_model("v1", enforce_comparison=False)
    _register(
        registry,
        "v2",
        dataset_hash="hash-2",
        selected_bic=-600.0,
        trained_at="2024-02-01T00:00:00+00:00",
    )
    registry.promote_model("v2")
    entry = registry.fallback_entry()
    # v1 was demoted to "fallback" during v2 promotion -> becomes the fallback choice.
    assert entry is not None and entry.model_version == "v1"


def test_fallback_prefers_most_recent_when_multiple_exist(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", selected_bic=-500.0, trained_at="2024-01-01T00:00:00+00:00")
    registry.promote_model("v1", enforce_comparison=False)
    _register(
        registry,
        "v2",
        dataset_hash="hash-2",
        selected_bic=-600.0,
        trained_at="2024-02-01T00:00:00+00:00",
    )
    registry.promote_model("v2")
    _register(
        registry,
        "v3",
        dataset_hash="hash-3",
        selected_bic=-700.0,
        trained_at="2024-03-01T00:00:00+00:00",
    )
    registry.promote_model("v3")
    # After v3 promotion: v1 and v2 are both "fallback"; registry picks most recent.
    entry = registry.fallback_entry()
    assert entry is not None and entry.model_version == "v2"


def test_rejected_candidate_does_not_displace_active(tmp_path: Path) -> None:
    """Retraining failure path: a worse candidate must never reach ``active``."""
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", selected_bic=-500.0)
    registry.promote_model("v1", enforce_comparison=False)
    _register(registry, "v2", dataset_hash="hash-2", selected_bic=-400.0)
    try:
        registry.promote_model("v2")
    except PromotionRejected:
        pass
    # Active pointer must still be v1; inference continues on the last good model.
    assert registry.active_version == "v1"
    assert registry.get("v1").status == "active"
    assert registry.get("v2").status == "candidate"


def test_fallback_survives_registry_reopen(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register(registry, "v1", selected_bic=-500.0)
    registry.promote_model("v1", enforce_comparison=False)
    _register(registry, "v2", dataset_hash="hash-2", selected_bic=-600.0)
    registry.promote_model("v2")
    reopened = ModelRegistry(tmp_path / "models")
    entry = reopened.fallback_entry()
    assert entry is not None and entry.model_version == "v1"


def test_get_fallback_model_returns_none_when_registry_empty(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    # No models exist -> no fallback model loadable.
    assert registry.get_fallback_model() is None
