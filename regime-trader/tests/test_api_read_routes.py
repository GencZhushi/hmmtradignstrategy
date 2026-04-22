"""Phase B2 - read-only state API returns engine-backed payloads."""
from __future__ import annotations

from pathlib import Path

from tests._api_fixtures import boot_client


def test_portfolio_and_regime_reflect_engine_state(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path)
    service = app.state.service
    service.application.position_tracker.apply_fill(
        symbol="SPY", side="BUY", qty=3.0, price=100.0, stop_price=95.0, regime_name="BULL"
    )

    portfolio = client.get("/portfolio").json()
    assert portfolio["equity"] > 0
    assert any(pos["symbol"] == "SPY" for pos in portfolio["positions"])

    positions = client.get("/positions").json()
    assert any(pos["symbol"] == "SPY" for pos in positions)

    regime = client.get("/regime/current").json()
    assert regime["regime_name"] is None or isinstance(regime["regime_name"], str)

    risk = client.get("/risk/status").json()
    assert "breaker_state" in risk and "active_constraints" in risk

    concentration = client.get("/concentration").json()
    assert "sector_exposure" in concentration


def test_freshness_endpoint_exposes_session_payload(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    payload = client.get("/freshness").json()
    assert "exchange_timezone" in payload
    assert "exchange_session_state" in payload
    assert "stale_data_blocked" in payload


def test_model_governance_endpoint_returns_schema(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    payload = client.get("/regime/model").json()
    assert "candidates" in payload
    assert "active_model_version" in payload


def test_orders_history_empty_initially(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    response = client.get("/orders/history")
    assert response.status_code == 200
    assert response.json() == []


def test_audit_logs_return_json_list(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    response = client.get("/audit/logs")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_error_paths_do_not_leak_internal_traces(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    response = client.get("/market/bars/daily", params={"symbol": "NO_SUCH"})
    assert response.status_code in (200, 404)
    if response.status_code == 404:
        payload = response.json()
        assert "detail" in payload
        assert "Traceback" not in str(payload)
