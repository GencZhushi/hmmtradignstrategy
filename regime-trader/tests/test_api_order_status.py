"""Phase B7 - API exposure of order lifecycle (attempts, partial fills, retries)."""
from __future__ import annotations

from pathlib import Path

from core.types import OrderStatus
from tests._api_fixtures import boot_client, operator_token


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _seed_order(client, token: str) -> str:
    response = client.post(
        "/orders/execute",
        json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
        headers={**_auth(token), "Idempotency-Key": "seed"},
    )
    assert response.status_code == 200
    intent_id = response.json()["intent_id"]
    return intent_id


def test_order_history_exposes_lifecycle_fields(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    _seed_order(client, token)
    orders = client.get("/orders/history").json()
    assert orders, "expected at least one order record"
    sample = orders[0]
    for key in (
        "order_id",
        "trade_id",
        "intent_id",
        "symbol",
        "side",
        "quantity",
        "filled_qty",
        "status",
        "protective_stop_status",
        "attempts",
    ):
        assert key in sample, f"missing lifecycle field '{key}'"


def test_partial_fill_distinguishable_from_filled(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    _seed_order(client, token)
    orders = client.get("/orders/history").json()
    assert orders
    order_id = orders[0]["order_id"]

    # Simulate a partial fill directly on the state machine and confirm the
    # API status reflects the intermediate state.
    state_machine = app.state.service.application.state_machine
    state_machine.handle_partial_fill(order_id, filled_qty=0.5, fill_price=100.0)
    app.state.service.repository.upsert_order(state_machine.summary(order_id))

    partial = next(o for o in client.get("/orders/history").json() if o["order_id"] == order_id)
    assert partial["status"] == OrderStatus.PARTIALLY_FILLED.value
    assert partial["filled_qty"] > 0

    state_machine.handle_partial_fill(order_id, filled_qty=state_machine.orders[order_id].remaining_qty(), fill_price=100.0)
    app.state.service.repository.upsert_order(state_machine.summary(order_id))
    filled = next(o for o in client.get("/orders/history").json() if o["order_id"] == order_id)
    assert filled["status"] == OrderStatus.FILLED.value


def test_attempts_list_returned_for_retries(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    _seed_order(client, token)
    orders = client.get("/orders/history").json()
    assert orders
    sample = orders[0]
    assert isinstance(sample["attempts"], list)
    # An executed order should have at least one attempt.
    assert len(sample["attempts"]) >= 1


def test_status_filter_returns_only_matching_orders(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    _seed_order(client, token)
    response = client.get("/orders/history", params={"status": "definitely_not_a_status"})
    assert response.status_code == 200
    assert response.json() == []


def test_order_payload_contains_trade_id_for_audit_trail(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    _seed_order(client, token)
    orders = client.get("/orders/history").json()
    assert all(o["trade_id"] for o in orders), "trade_id must be present for retry history"
