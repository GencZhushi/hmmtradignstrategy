"""Phase A13 - model registry versioning, persistence, and metadata capture.

Tests confirm the filesystem-backed registry:

- records every registered candidate with its dataset hash and BIC
- persists metadata between instances of the same registry (reload safe)
- stores a stand-alone training report via ``store_training_metadata``
- never auto-promotes: registrations start in ``candidate`` status
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from core.hmm_engine import ModelMetadata
from core.model_registry import (
    ModelRegistry,
    store_training_metadata,
)


def _metadata(
    version: str,
    *,
    dataset_hash: str = "hash-1",
    selected_bic: float = -500.0,
    log_likelihood: float = 300.0,
) -> ModelMetadata:
    return ModelMetadata(
        model_version=version,
        trained_at="2024-05-01T12:00:00+00:00",
        n_states=3,
        n_samples=504,
        feature_columns=["log_return", "realized_vol_20"],
        bic_scores={3: selected_bic, 4: selected_bic + 50, 5: selected_bic + 120},
        selected_bic=selected_bic,
        log_likelihood=log_likelihood,
        dataset_hash=dataset_hash,
        random_state=42,
        regimes=[{"regime_id": 0, "regime_name": "low_vol"}],
        notes="unit test",
    )


def _register_candidate(registry: ModelRegistry, version: str, **overrides) -> None:
    meta = _metadata(version, **overrides)
    artifact = registry.root / version
    artifact.mkdir(parents=True, exist_ok=True)
    registry.register(artifact_dir=artifact, metadata=meta, notes=meta.notes)


def test_register_records_candidate_with_metadata(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register_candidate(registry, "v1")
    entry = registry.get("v1")
    assert entry.model_version == "v1"
    assert entry.dataset_hash == "hash-1"
    assert entry.status == "candidate"
    assert entry.selected_bic == -500.0
    assert registry.active_version is None


def test_registry_reloads_state_from_disk(tmp_path: Path) -> None:
    first = ModelRegistry(tmp_path / "models")
    _register_candidate(first, "v1")
    _register_candidate(first, "v2", dataset_hash="hash-2", selected_bic=-520.0)
    first.promote_model("v1", actor="alice", enforce_comparison=False)

    # Reopen the registry - metadata + active pointer must survive.
    reopened = ModelRegistry(tmp_path / "models")
    versions = {e.model_version for e in reopened.list_versions()}
    assert versions == {"v1", "v2"}
    assert reopened.active_version == "v1"
    assert reopened.get("v1").status == "active"
    assert reopened.get("v2").status == "candidate"


def test_store_training_metadata_writes_side_car_report(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    meta = _metadata("v1")
    path = store_training_metadata(registry, meta, extras={"dataset_rows": 504})
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["model_version"] == "v1"
    assert payload["extras"]["dataset_rows"] == 504


def test_dataset_hash_is_recorded_on_each_entry(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register_candidate(registry, "v1", dataset_hash="hash-a")
    _register_candidate(registry, "v2", dataset_hash="hash-b")
    hashes = {e.model_version: e.dataset_hash for e in registry.list_versions()}
    assert hashes == {"v1": "hash-a", "v2": "hash-b"}


def test_registry_preserves_bic_and_log_likelihood(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register_candidate(registry, "v1", selected_bic=-500.0, log_likelihood=250.0)
    entry = registry.get("v1")
    assert entry.selected_bic == -500.0
    assert entry.log_likelihood == 250.0


def test_registry_index_file_is_json(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    _register_candidate(registry, "v1")
    index = tmp_path / "models" / "registry.json"
    assert index.exists()
    payload = json.loads(index.read_text(encoding="utf-8"))
    assert "models" in payload and "v1" in payload["models"]


def test_register_different_versions_with_same_dataset_stays_deduplicated(tmp_path: Path) -> None:
    # Two candidates trained on the same dataset must both be registered; it is
    # the promotion path that rejects same-hash promotions, not registration.
    registry = ModelRegistry(tmp_path / "models")
    _register_candidate(registry, "v1", dataset_hash="hash-x")
    _register_candidate(registry, "v2", dataset_hash="hash-x")
    versions = {e.model_version for e in registry.list_versions()}
    assert versions == {"v1", "v2"}


def test_metadata_replace_via_dataclass(tmp_path: Path) -> None:
    # ``replace`` is the pattern the runtime uses to build a second candidate
    # from an existing metadata object without mutating the original.
    original = _metadata("v1")
    clone = replace(original, model_version="v2", dataset_hash="hash-z")
    assert original.model_version == "v1"
    assert clone.model_version == "v2"
    assert clone.dataset_hash == "hash-z"
