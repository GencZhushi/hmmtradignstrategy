"""Phase B9 - API exposure of sector/correlation concentration metrics."""
from __future__ import annotations

from pathlib import Path

from tests._api_fixtures import boot_client


def test_concentration_payload_contains_sector_exposure(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    body = client.get("/concentration").json()
    assert "sector_exposure" in body
    assert "projected_post_trade_exposure" in body
    assert "correlation_metrics" in body
    assert "blocked_reasons" in body


def test_portfolio_exposes_sector_exposure_buckets(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path)
    service = app.state.service
    service.application.position_tracker.apply_fill(
        symbol="SPY", side="BUY", qty=10, price=100.0, stop_price=95.0, regime_name="BULL"
    )
    portfolio = client.get("/portfolio").json()
    assert isinstance(portfolio["sector_exposure"], dict)
    # One of the positions should be tagged with an explicit sector or ETF bucket.
    positions = portfolio["positions"]
    sample = next(pos for pos in positions if pos["symbol"] == "SPY")
    assert sample["sector_bucket"] is not None or sample["etf_bucket"] is not None


def test_risk_status_exposes_sector_limit_and_correlation_thresholds(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    risk = client.get("/risk/status").json()
    assert "active_constraints" in risk
    constraints = risk["active_constraints"]
    for key in ("max_exposure", "max_single_position", "max_sector_exposure"):
        assert key in constraints


def test_preview_response_includes_projected_exposure(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    # Login as operator
    token_resp = client.post(
        "/auth/login", json={"username": "admin", "password": "admin-pw"}
    ).json()
    token = token_resp["access_token"]
    response = client.post(
        "/orders/preview",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    body = response.json()
    # Contract: projected_exposure + projected_sector_exposure keys must be present
    # (they may be None when the risk manager has no baseline yet).
    assert "projected_exposure" in body
    assert "projected_sector_exposure" in body
    assert "sector_bucket" in body
    assert "etf_bucket" in body


def test_concentration_correlation_metrics_are_dict(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    body = client.get("/concentration").json()
    assert isinstance(body["correlation_metrics"], dict)
