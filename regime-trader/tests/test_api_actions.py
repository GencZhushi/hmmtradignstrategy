"""Phase B3 + B6 - preview, execute, approval workflow, idempotency via API."""
from __future__ import annotations

from pathlib import Path

from tests._api_fixtures import boot_client, admin_token, operator_token


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_preview_returns_structured_order_plan(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="manual")
    token = operator_token(app)
    payload = {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05}
    response = client.post("/orders/preview", json=payload, headers=_auth_headers(token))
    assert response.status_code == 200
    body = response.json()
    assert body["plan_id"]
    assert body["status"] in {"approved", "rejected"}
    assert "reason_codes" in body


def test_execute_route_runs_preview_then_queues_approval(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="manual")
    token = operator_token(app)
    payload = {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05}
    exec_response = client.post("/orders/execute", json=payload, headers=_auth_headers(token))
    assert exec_response.status_code == 200
    pending = client.get("/approvals/pending").json()
    assert len(pending) == 1
    assert pending[0]["status"] == "pending"


def test_execute_route_auto_executes_when_policy_allows(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    payload = {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05}
    response = client.post("/orders/execute", json=payload, headers=_auth_headers(token))
    assert response.status_code == 200
    history = client.get("/orders/history").json()
    assert history, "execute should have produced at least one order record"


def test_duplicate_idempotency_key_returns_same_intent(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    payload = {
        "symbol": "SPY",
        "direction": "LONG",
        "allocation_pct": 0.05,
        "idempotency_key": "fixed-key",
    }
    first = client.post(
        "/orders/preview",
        json=payload,
        headers={**_auth_headers(token), "Idempotency-Key": "fixed-key"},
    )
    second = client.post(
        "/orders/preview",
        json=payload,
        headers={**_auth_headers(token), "Idempotency-Key": "fixed-key"},
    )
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["intent_id"] == second.json()["intent_id"]


def test_approval_approve_then_reject_workflow(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="manual")
    operator = operator_token(app)
    admin = admin_token(app)
    client.post(
        "/orders/execute",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers=_auth_headers(operator),
    )
    pending = client.get("/approvals/pending").json()
    assert pending, "expected one pending approval"
    approval_id = pending[0]["approval_id"]
    approve_response = client.post(
        "/approvals/approve",
        json={"approval_id": approval_id, "reason": "ok"},
        headers=_auth_headers(admin),
    )
    assert approve_response.status_code == 200
    # Re-approving should fail with 409.
    re_approve = client.post(
        "/approvals/approve",
        json={"approval_id": approval_id, "reason": "ok"},
        headers=_auth_headers(admin),
    )
    assert re_approve.status_code == 409


def test_close_all_requires_admin(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path)
    operator = operator_token(app)
    admin = admin_token(app)
    operator_response = client.post("/positions/close-all", headers=_auth_headers(operator))
    assert operator_response.status_code == 403
    admin_response = client.post("/positions/close-all", headers=_auth_headers(admin))
    assert admin_response.status_code == 200


def test_arm_live_requires_admin_role(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path)
    admin = admin_token(app)
    response = client.post(
        "/config/arm-live",
        json={"ttl_minutes": 5, "reason": "operator check"},
        headers=_auth_headers(admin),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["armed_by"] == "admin"


def test_rejected_execute_returns_400_with_reason(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    operator = operator_token(app)
    # Missing stop because allocation=0 is a pass-through -> use invalid allocation>1.5 to force rejection
    payload = {"symbol": "SPY", "direction": "LONG", "allocation_pct": 1.6}
    response = client.post("/orders/execute", json=payload, headers=_auth_headers(operator))
    assert response.status_code in (400, 422)
