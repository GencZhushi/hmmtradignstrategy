"""Phase B1 - boot the FastAPI app + basic auth + health routing."""
from __future__ import annotations

from pathlib import Path

from tests._api_fixtures import boot_client


def test_health_route_returns_ok(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["trading_mode"] == "paper"


def test_info_route_contains_mode(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    response = client.get("/info")
    assert response.status_code == 200
    assert response.json()["trading_mode"] == "paper"


def test_protected_route_requires_authentication(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    response = client.post("/orders/preview", json={
        "symbol": "SPY",
        "direction": "LONG",
        "allocation_pct": 0.05,
    })
    assert response.status_code == 401


def test_login_issues_jwt_for_bootstrap_admin(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    response = client.post("/auth/login", json={"username": "admin", "password": "admin-pw"})
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "admin"
    assert data["access_token"]


def test_openclaw_router_mounted(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    response = client.get("/agent/tools")
    assert response.status_code == 200
    tools = response.json()["tools"]
    names = {tool["name"] for tool in tools}
    assert {"get_regime", "preview_trade", "submit_trade_intent"} <= names
