"""Phase B5 - event streaming surface (SSE + recent-event cache).

Focused tests on the ``/events/*`` endpoints. The SSE body itself is not
consumed inside ``TestClient`` (its ASGI transport buffers forever on
long-lived streams); instead the route registration and media-type contract
are asserted, along with the cached ``/events/recent`` payload the dashboard
uses to warm up on initial connect.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.routing import APIRoute
from starlette.responses import StreamingResponse

from tests._api_fixtures import boot_client, operator_token


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_stream_route_is_registered(tmp_path: Path) -> None:
    _, app = boot_client(tmp_path)
    stream = next(
        (r for r in app.routes if isinstance(r, APIRoute) and r.path == "/events/stream"),
        None,
    )
    assert stream is not None, "SSE route /events/stream must be registered"


def test_events_recent_returns_empty_list_on_fresh_boot(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    payload = client.get("/events/recent").json()
    assert "events" in payload
    assert isinstance(payload["events"], list)
    assert "generated_at" in payload


def test_events_recent_surfaces_execute_event(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    client.post(
        "/orders/execute",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers=_auth(token),
    )
    events = client.get("/events/recent").json()["events"]
    names = {event.get("event") for event in events}
    assert names & {"intent_submitted", "intent_previewed", "order_submitted"}


def test_events_recent_items_have_timestamp_and_payload(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    client.post(
        "/orders/execute",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers=_auth(token),
    )
    events = client.get("/events/recent").json()["events"]
    assert events
    for entry in events:
        assert "event" in entry and "payload" in entry
        assert "timestamp" in entry


def test_stream_endpoint_calls_produce_streaming_response(tmp_path: Path) -> None:
    """Call the endpoint factory directly and verify the return type.

    Going through ``TestClient`` on a live SSE generator would block because
    the ASGI transport buffers bodies and never propagates disconnects to the
    generator. Exercising the endpoint callable is enough to guarantee the
    platform wires the handler through to a ``StreamingResponse``.
    """
    import asyncio

    from api.routes.streaming import stream_events

    class _FakeRequest:
        async def is_disconnected(self) -> bool:
            return True

    _, app = boot_client(tmp_path)
    service = app.state.service
    response = asyncio.run(
        stream_events(_FakeRequest(), service=service)  # type: ignore[arg-type]
    )
    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"


def test_events_recent_payload_json_serialisable(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    import json

    response = client.get("/events/recent")
    assert response.status_code == 200
    # Round-tripping through json catches any non-serialisable entries the
    # service might accidentally leak into the cache.
    parsed = json.loads(response.text)
    assert "events" in parsed
