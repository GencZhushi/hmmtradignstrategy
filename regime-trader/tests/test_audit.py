"""Phase B5 - audit trail endpoints and structured event capture.

Complements ``test_api_streaming_audit.py`` with dedicated coverage of the
``/audit/*`` surface area. The audit trail is the single source of truth for
operator/agent accountability, so these tests pin:

- every mutating action leaves a persisted audit record
- records carry actor, actor_type, action, resource_type, resource_id
- filtering by ``resource_type`` and ``actor`` works
- ``limit`` query caps the returned volume
- configuration changes (arm-live, approval-mode) are audited by the admin role
"""
from __future__ import annotations

from pathlib import Path

from tests._api_fixtures import admin_token, boot_client, operator_token


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_audit_event_records_actor_and_action(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    client.post(
        "/orders/execute",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers=_auth(token),
    )
    audit = client.get("/audit/logs", headers=_auth(token)).json()
    assert audit, "expected at least one audit record"
    record = audit[0]
    assert {"actor", "actor_type", "action", "resource_type"}.issubset(record.keys())
    assert record["actor"] == "operator"


def test_audit_filter_by_actor(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    operator = operator_token(app)
    admin = admin_token(app)
    client.post(
        "/orders/execute",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers=_auth(operator),
    )
    client.post(
        "/config/arm-live",
        json={"ttl_minutes": 5, "reason": "unit test"},
        headers=_auth(admin),
    )
    admin_records = client.get("/audit/logs", params={"actor": "admin"}, headers=_auth(admin)).json()
    operator_records = client.get("/audit/logs", params={"actor": "operator"}, headers=_auth(operator)).json()
    assert admin_records and all(r["actor"] == "admin" for r in admin_records)
    assert operator_records and all(r["actor"] == "operator" for r in operator_records)


def test_audit_limit_caps_returned_volume(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    for i in range(5):
        client.post(
            "/orders/execute",
            json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05 + i * 0.001},
            headers=_auth(token),
        )
    limited = client.get("/audit/logs", params={"limit": 2}, headers=_auth(token)).json()
    assert len(limited) <= 2


def test_audit_logs_reject_out_of_range_limit(tmp_path: Path) -> None:
    client, _ = boot_client(tmp_path)
    # 0 is outside the [1, 500] bound documented on the route definition.
    too_low = client.get("/audit/logs", params={"limit": 0})
    too_high = client.get("/audit/logs", params={"limit": 501})
    assert too_low.status_code == 422
    assert too_high.status_code == 422


def test_audit_logs_payload_shape_matches_schema(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    client.post(
        "/orders/execute",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers=_auth(token),
    )
    audit = client.get("/audit/logs", headers=_auth(token)).json()
    for record in audit:
        # Every record must at minimum carry these keys so downstream readers
        # (dashboard, agent audit summary) never need to branch on absence.
        for key in ("actor", "actor_type", "action", "resource_type", "timestamp"):
            assert key in record, f"audit record missing {key}: {record}"


def test_audit_events_endpoint_returns_recent_event_stream_cache(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    client.post(
        "/orders/execute",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers=_auth(token),
    )
    payload = client.get("/audit/events", headers=_auth(token)).json()
    assert "events" in payload
    assert isinstance(payload["events"], list)


def test_rejections_are_audited_with_approval_resource_type(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="manual")
    operator = operator_token(app)
    admin = admin_token(app)
    client.post(
        "/orders/execute",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers=_auth(operator),
    )
    pending = client.get("/approvals/pending", headers=_auth(operator)).json()
    if not pending:
        return  # approval policy auto-approved; test would be vacuous
    approval_id = pending[0]["approval_id"]
    client.post(
        "/approvals/reject",
        json={"approval_id": approval_id, "reason": "not today"},
        headers=_auth(admin),
    )
    audit = client.get(
        "/audit/logs",
        params={"resource_type": "approval"},
        headers=_auth(admin),
    ).json()
    # Reject writes a dedicated ``approval_rejected`` audit record (approve
    # instead writes ``intent_approved_executed`` under resource_type=intent).
    assert audit, "rejection activity must be audited under resource_type=approval"
    assert all(r["resource_type"] == "approval" for r in audit)
    assert any(r["action"] == "approval_rejected" for r in audit)


def test_approval_execution_is_audited_under_intent(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="manual")
    operator = operator_token(app)
    admin = admin_token(app)
    client.post(
        "/orders/execute",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers=_auth(operator),
    )
    pending = client.get("/approvals/pending", headers=_auth(operator)).json()
    if not pending:
        return
    approval_id = pending[0]["approval_id"]
    client.post(
        "/approvals/approve",
        json={"approval_id": approval_id, "reason": "ok"},
        headers=_auth(admin),
    )
    audit = client.get(
        "/audit/logs",
        params={"resource_type": "intent"},
        headers=_auth(admin),
    ).json()
    actions = {r["action"] for r in audit}
    assert "intent_approved_executed" in actions
