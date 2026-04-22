"""Phase B8 - API exposure of data freshness, session state, and regime timing."""
from __future__ import annotations

from pathlib import Path

from tests._api_fixtures import boot_client


def test_freshness_payload_exposes_required_keys(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    body = client.get("/freshness").json()
    required = {
        "exchange_timezone",
        "exchange_session_state",
        "data_freshness_status",
        "daily_data_stale",
        "intraday_data_stale",
        "stale_data_blocked",
    }
    assert required <= set(body.keys())


def test_session_state_is_one_of_known_values(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    body = client.get("/freshness").json()
    assert body["exchange_session_state"] in {
        "pre_market",
        "open",
        "post_market",
        "closed",
        "holiday",
    }


def test_regime_endpoint_exposes_effective_session_date(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    body = client.get("/regime/current").json()
    # Even with no training yet, the contract requires the field to be present.
    assert "effective_session_date" in body
    assert "active_model_version" in body


def test_stale_data_flag_matches_blocked_flag(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    body = client.get("/freshness").json()
    # If either daily or intraday is stale, blocked must be True.
    if body["daily_data_stale"] or body["intraday_data_stale"]:
        assert body["stale_data_blocked"] is True


def test_health_route_includes_active_model(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    body = client.get("/health").json()
    # Contract: active_model key must exist even if no model is promoted yet.
    assert "active_model" in body
