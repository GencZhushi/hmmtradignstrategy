"""Model registry (Phase A13).

Tracks trained HMM artifacts, captures promotion/rollback decisions, and provides
a fallback-model lookup so a failed retraining never strands the engine with no
usable model.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from core.hmm_engine import ModelMetadata, VolatilityRegimeHMM

LOG = logging.getLogger(__name__)

REGISTRY_FILENAME = "registry.json"


@dataclass
class RegistryEntry:
    """Metadata for a single registered model version."""

    model_version: str
    path: str
    trained_at: str
    dataset_hash: str
    n_states: int
    selected_bic: float
    log_likelihood: float
    notes: str = ""
    status: str = "candidate"  # candidate | active | fallback | retired
    promoted_at: str | None = None
    promoted_by: str | None = None
    previous_active: str | None = None


@dataclass
class PromotionDecision:
    """Result returned when comparing a candidate against the active model."""

    candidate_version: str
    active_version: str | None
    approved: bool
    reason: str
    delta_bic: float | None = None


@dataclass
class ModelRegistry:
    """Filesystem-backed registry of HMM model versions."""

    root: Path
    _index: dict[str, RegistryEntry] = field(default_factory=dict, init=False)
    _active_version: str | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._load()

    # ---------------------------------------------------------- persistence
    def _registry_path(self) -> Path:
        return self.root / REGISTRY_FILENAME

    def _load(self) -> None:
        path = self._registry_path()
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._index = {
            key: RegistryEntry(**entry) for key, entry in payload.get("models", {}).items()
        }
        self._active_version = payload.get("active")

    def _save(self) -> None:
        payload = {
            "active": self._active_version,
            "models": {k: asdict(v) for k, v in self._index.items()},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self._registry_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._registry_path())

    # ---------------------------------------------------------- registration
    def register(
        self,
        artifact_dir: Path,
        metadata: ModelMetadata,
        notes: str = "",
    ) -> RegistryEntry:
        entry = RegistryEntry(
            model_version=metadata.model_version,
            path=str(Path(artifact_dir).resolve()),
            trained_at=metadata.trained_at,
            dataset_hash=metadata.dataset_hash,
            n_states=metadata.n_states,
            selected_bic=metadata.selected_bic,
            log_likelihood=metadata.log_likelihood,
            notes=notes,
        )
        self._index[entry.model_version] = entry
        self._save()
        LOG.info("Registered model %s (status=candidate)", entry.model_version)
        return entry

    # ---------------------------------------------------------- queries
    def list_versions(self) -> list[RegistryEntry]:
        return list(self._index.values())

    def get(self, version: str) -> RegistryEntry:
        return self._index[version]

    @property
    def active_version(self) -> str | None:
        return self._active_version

    def active_entry(self) -> RegistryEntry | None:
        if self._active_version is None:
            return None
        return self._index.get(self._active_version)

    def fallback_entry(self) -> RegistryEntry | None:
        active = self.active_entry()
        candidates = [
            entry
            for entry in self._index.values()
            if entry.status in {"active", "fallback"} and entry.model_version != (active.model_version if active else None)
        ]
        if not candidates:
            return active
        candidates.sort(key=lambda e: e.trained_at, reverse=True)
        return candidates[0]

    def get_fallback_model(self) -> "VolatilityRegimeHMM | None":
        entry = self.fallback_entry()
        if entry is None:
            return None
        hmm = VolatilityRegimeHMM()
        hmm.load_model(Path(entry.path))
        return hmm

    # ---------------------------------------------------------- governance
    def compare_candidate_vs_active(
        self,
        candidate_version: str,
        *,
        min_improvement_bic: float = 0.0,
    ) -> PromotionDecision:
        candidate = self._index.get(candidate_version)
        if candidate is None:
            raise KeyError(f"Unknown candidate version: {candidate_version}")
        active = self.active_entry()
        if active is None:
            return PromotionDecision(
                candidate_version=candidate_version,
                active_version=None,
                approved=True,
                reason="no active model; first promotion allowed",
            )
        if candidate.dataset_hash == active.dataset_hash:
            return PromotionDecision(
                candidate_version=candidate_version,
                active_version=active.model_version,
                approved=False,
                reason="candidate trained on same dataset as active",
                delta_bic=0.0,
            )
        delta = active.selected_bic - candidate.selected_bic
        approved = delta > min_improvement_bic
        reason = (
            f"candidate BIC improved by {delta:.3f}"
            if approved
            else f"candidate BIC not better than active (delta={delta:.3f})"
        )
        return PromotionDecision(
            candidate_version=candidate_version,
            active_version=active.model_version,
            approved=approved,
            reason=reason,
            delta_bic=delta,
        )

    def promote_model(
        self,
        version: str,
        actor: str = "system",
        *,
        enforce_comparison: bool = True,
        min_improvement_bic: float = 0.0,
    ) -> RegistryEntry:
        entry = self._index[version]
        if enforce_comparison:
            decision = self.compare_candidate_vs_active(version, min_improvement_bic=min_improvement_bic)
            if not decision.approved:
                raise PromotionRejected(decision)
        previous = self._active_version
        for other in self._index.values():
            if other.status == "active" and other.model_version != version:
                other.status = "fallback"
        entry.status = "active"
        entry.promoted_at = datetime.now(timezone.utc).isoformat()
        entry.promoted_by = actor
        entry.previous_active = previous
        self._active_version = version
        self._save()
        LOG.info("Promoted model %s (previous=%s actor=%s)", version, previous, actor)
        return entry

    def rollback_model(self, actor: str = "system") -> RegistryEntry | None:
        active = self.active_entry()
        if active is None or not active.previous_active:
            raise RollbackRejected("no previous active model to roll back to")
        previous_version = active.previous_active
        previous = self._index.get(previous_version)
        if previous is None:
            raise RollbackRejected(f"previous version {previous_version} missing from registry")
        active.status = "retired"
        previous.status = "active"
        previous.promoted_at = datetime.now(timezone.utc).isoformat()
        previous.promoted_by = actor
        previous.previous_active = None
        self._active_version = previous.model_version
        self._save()
        LOG.info("Rolled back to model %s (retired %s)", previous.model_version, active.model_version)
        return previous


class PromotionRejected(RuntimeError):
    """Raised when a promotion attempt is blocked by governance policy."""

    def __init__(self, decision: PromotionDecision):
        super().__init__(decision.reason)
        self.decision = decision


class RollbackRejected(RuntimeError):
    pass


def register_model_version(
    registry: ModelRegistry,
    hmm: VolatilityRegimeHMM,
    *,
    notes: str = "",
) -> RegistryEntry:
    """Convenience helper: persist a fitted HMM into the registry."""
    artifact_dir = hmm.save_model(registry.root)
    return registry.register(artifact_dir=artifact_dir, metadata=hmm.metadata, notes=notes)


def store_training_metadata(
    registry: ModelRegistry,
    metadata: ModelMetadata,
    *,
    extras: Mapping[str, object] | None = None,
) -> Path:
    """Persist a stand-alone training report for audit/governance."""
    filename = f"{metadata.model_version}.training.json"
    path = registry.root / filename
    payload: dict[str, object] = {"metadata": asdict(metadata)}
    if extras:
        payload["extras"] = dict(extras)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def compute_training_dataset_hash(features) -> str:
    """Re-export of the dataset hash helper so callers do not reach into internals."""
    from core.hmm_engine import _hash_dataset  # local import to avoid cycle

    return _hash_dataset(features)


def _generate_version_id() -> str:
    return f"hmm-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def atomic_copy(src: Path, dst: Path) -> Path:
    """Copy ``src`` into ``dst`` atomically (used by tests and recovery)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(dst.parent)) as tmp:
        staging = Path(tmp) / src.name
        if src.is_dir():
            shutil.copytree(src, staging)
        else:
            shutil.copy2(src, staging)
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        shutil.move(str(staging), str(dst))
    return dst
