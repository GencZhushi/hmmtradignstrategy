"""Phase B6 - API idempotency middleware + duplicate-write handling."""
from __future__ import annotations

from pathlib import Path

from tests._api_fixtures import boot_client, admin_token, operator_token


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_missing_idempotency_key_is_tolerated(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    payload = {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05}
    response = client.post("/orders/preview", json=payload, headers=_auth(token))
    assert response.status_code == 200
    # Server should synthesize a deterministic key via stable_idempotency_key.
    body = response.json()
    assert body["intent_id"]


def test_same_idempotency_key_returns_same_intent_on_retry(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    payload = {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05}
    headers = {**_auth(token), "Idempotency-Key": "retry-abc"}
    first = client.post("/orders/preview", json=payload, headers=headers).json()
    second = client.post("/orders/preview", json=payload, headers=headers).json()
    assert first["intent_id"] == second["intent_id"]


def test_execute_idempotency_does_not_double_submit(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    payload = {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05}
    headers = {**_auth(token), "Idempotency-Key": "exec-key-1"}
    first = client.post("/orders/execute", json=payload, headers=headers)
    second = client.post("/orders/execute", json=payload, headers=headers)
    assert first.status_code == 200
    assert second.status_code == 200
    # Both responses should refer to the same intent even though execute was called twice.
    assert first.json()["intent_id"] == second.json()["intent_id"]
    orders = client.get("/orders/history").json()
    intent_ids = [o["intent_id"] for o in orders]
    # At most one order should have been created for the shared intent id.
    assert intent_ids.count(first.json()["intent_id"]) <= 1


def test_read_endpoints_do_not_mutate_execution_state(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    # Seed one executed intent so the state machine is non-empty.
    client.post(
        "/orders/execute",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers={**_auth(token), "Idempotency-Key": "seed"},
    )
    first_orders = client.get("/orders/history").json()
    for _ in range(3):
        client.get("/orders/history")
        client.get("/portfolio")
        client.get("/regime/current")
        client.get("/freshness")
    second_orders = client.get("/orders/history").json()
    assert len(first_orders) == len(second_orders)


def test_conflicting_payload_with_same_key_still_returns_original(tmp_path: Path) -> None:
    """Replaying a retry with a mutated payload must not create a second intent."""
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    headers = {**_auth(token), "Idempotency-Key": "fixed-key"}
    first = client.post(
        "/orders/preview",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers=headers,
    ).json()
    # Send a different payload with the same idempotency key - the server must
    # still respond with the original intent id and NOT create a second record.
    second = client.post(
        "/orders/preview",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.10},
        headers=headers,
    ).json()
    assert first["intent_id"] == second["intent_id"]
