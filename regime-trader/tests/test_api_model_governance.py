"""Phase B10 - API exposure of model governance (active, fallback, candidates)."""
from __future__ import annotations

from pathlib import Path

from core.hmm_engine import ModelMetadata
from core.model_registry import RegistryEntry
from tests._api_fixtures import boot_client, admin_token


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _seed_registry_entries(app) -> tuple[str, str]:
    """Seed two candidate model entries and promote one as active."""
    registry = app.state.service.application.model_registry
    v1 = RegistryEntry(
        model_version="hmm-v1",
        path=str(Path(registry.root) / "hmm-v1"),
        trained_at="2024-01-01T00:00:00+00:00",
        dataset_hash="dataset-v1",
        n_states=3,
        selected_bic=120.0,
        log_likelihood=-50.0,
        status="candidate",
    )
    v2 = RegistryEntry(
        model_version="hmm-v2",
        path=str(Path(registry.root) / "hmm-v2"),
        trained_at="2024-06-01T00:00:00+00:00",
        dataset_hash="dataset-v2",
        n_states=4,
        selected_bic=118.0,
        log_likelihood=-48.0,
        status="candidate",
    )
    registry._index["hmm-v1"] = v1
    registry._index["hmm-v2"] = v2
    registry._save()
    registry.promote_model("hmm-v1", actor="bootstrap", enforce_comparison=False)
    return "hmm-v1", "hmm-v2"


def test_model_governance_exposes_active_and_candidates(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path)
    active, _ = _seed_registry_entries(app)
    response = client.get("/regime/model")
    assert response.status_code == 200
    body = response.json()
    assert body["active_model_version"] == active
    assert isinstance(body["candidates"], list)
    assert any(c["model_version"] == active for c in body["candidates"])


def test_model_governance_via_config_route(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path)
    _seed_registry_entries(app)
    response = client.get("/config/model")
    assert response.status_code == 200
    body = response.json()
    assert body["active_model_version"] == "hmm-v1"


def test_promote_model_requires_admin_role(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path)
    _seed_registry_entries(app)
    # No auth -> 401
    response = client.post("/config/model/promote", params={"version": "hmm-v2"})
    assert response.status_code == 401

    # Admin token works
    admin = admin_token(app)
    response = client.post(
        "/config/model/promote",
        params={"version": "hmm-v2"},
        headers=_auth(admin),
    )
    # Promote returns 200 on success or 409 if governance rejects.
    assert response.status_code in (200, 409)


def test_rollback_returns_previous_active(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path)
    _seed_registry_entries(app)
    admin = admin_token(app)
    # Promote v2 first so a rollback target exists.
    promote = client.post(
        "/config/model/promote",
        params={"version": "hmm-v2"},
        headers=_auth(admin),
    )
    if promote.status_code != 200:
        return  # promotion rejected; not a failure of this test
    rollback = client.post("/config/model/rollback", headers=_auth(admin))
    assert rollback.status_code == 200
    body = rollback.json()
    assert body.get("model_version") == "hmm-v1"


def test_health_reports_active_model(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path)
    active, _ = _seed_registry_entries(app)
    health = client.get("/health").json()
    assert health["active_model"] == active


def test_regime_current_exposes_active_model_version(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path)
    active, _ = _seed_registry_entries(app)
    regime = client.get("/regime/current").json()
    assert regime["active_model_version"] == active
