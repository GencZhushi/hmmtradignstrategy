"""Phase B6 - API-level concurrency safety against parallel writes."""
from __future__ import annotations

import threading
from pathlib import Path

from tests._api_fixtures import boot_client, operator_token


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_concurrent_executes_with_same_key_dedupe(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    payload = {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05}
    headers = {**_auth(token), "Idempotency-Key": "parallel-key"}

    responses: list = []

    def _submit() -> None:
        responses.append(client.post("/orders/execute", json=payload, headers=headers))

    threads = [threading.Thread(target=_submit) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(r.status_code == 200 for r in responses)
    intent_ids = {r.json()["intent_id"] for r in responses}
    # Every request must collapse to a single intent id.
    assert len(intent_ids) == 1

    orders = client.get("/orders/history").json()
    matching = [o for o in orders if o["intent_id"] == intent_ids.pop()]
    assert len(matching) <= 1, "duplicate orders were created for concurrent retries"


def test_concurrent_different_keys_produce_independent_intents(tmp_path: Path) -> None:
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)
    responses: list = []

    def _submit(idx: int) -> None:
        headers = {**_auth(token), "Idempotency-Key": f"key-{idx}"}
        body = {"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05}
        responses.append(client.post("/orders/preview", json=body, headers=headers))

    threads = [threading.Thread(target=_submit, args=(i,)) for i in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    intent_ids = {r.json()["intent_id"] for r in responses if r.status_code == 200}
    # Different keys must create distinct intents.
    assert len(intent_ids) == len(responses)


def test_read_routes_never_block_on_mutation(tmp_path: Path) -> None:
    """Concurrent reads alongside writes must not deadlock or raise."""
    client, app = boot_client(tmp_path, approval_mode="auto_paper")
    token = operator_token(app)

    def _write() -> None:
        for i in range(5):
            client.post(
                "/orders/preview",
                json={"symbol": "SPY", "direction": "LONG", "allocation_pct": 0.05},
                headers={**_auth(token), "Idempotency-Key": f"mix-{i}"},
            )

    def _read() -> None:
        for _ in range(5):
            assert client.get("/portfolio").status_code == 200
            assert client.get("/risk/status").status_code == 200

    threads = [threading.Thread(target=_write), threading.Thread(target=_read)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
