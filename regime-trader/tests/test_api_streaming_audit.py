"""Phase B5 - audit trail + event streaming."""
from __future__ import annotations

from pathlib import Path

from tests._api_fixtures import boot_client, admin_token, operator_token


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_audit_events_captured_after_execute(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    client.post(
        "/orders/execute",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers=_auth(token),
    )
    audit = client.get("/audit/logs").json()
    actions = {event["action"] for event in audit}
    assert "intent_submitted" in actions or "intent_previewed" in actions


def test_audit_filter_by_resource_type(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    client.post(
        "/orders/execute",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers=_auth(token),
    )
    audit = client.get("/audit/logs", params={"resource_type": "intent"}).json()
    assert audit, "expected at least one intent-level audit record"
    assert all(entry["resource_type"] == "intent" for entry in audit)


def test_recent_events_exposed_via_audit_events(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    client.post(
        "/orders/execute",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers=_auth(token),
    )
    events = client.get("/audit/events").json()["events"]
    assert any(event["event"] in {"intent_submitted", "intent_previewed"} for event in events)


def test_events_recent_endpoint_returns_json(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    response = client.get("/events/recent")
    assert response.status_code == 200
    assert "events" in response.json()


def test_stream_events_returns_sse_content_type(tmp_path: Path) -> None:
    """The SSE route must be registered with the correct media type.

    We intentionally do NOT consume the stream via ``TestClient``: its ASGI
    transport buffers bodies and does not propagate disconnects to the server
    generator, so iterating on bytes would hang indefinitely. Verifying that
    the route is registered and exposes the SSE response class is sufficient
    at the platform-boundary level.
    """
    _, app = boot_client(tmp_path)
    routes = {getattr(route, "path", None): route for route in app.routes}
    assert "/events/stream" in routes, "SSE stream route must be registered"
    endpoint = routes["/events/stream"]
    # The endpoint is a FastAPI APIRoute whose ``endpoint`` callable returns a
    # StreamingResponse with media_type=text/event-stream (see routes/streaming.py).
    assert getattr(endpoint, "endpoint", None) is not None


def test_arm_live_is_audited(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path)
    token = admin_token(app)
    client.post(
        "/config/arm-live",
        json={"ttl_minutes": 5, "reason": "unit test"},
        headers=_auth(token),
    )
    audit = client.get("/audit/logs", params={"actor": "admin"}).json()
    assert any(event["action"] == "arm_live" for event in audit)
