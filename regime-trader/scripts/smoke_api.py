"""Hit the live API and print structured state for verification."""
from __future__ import annotations

import json

import httpx

BASE = "http://127.0.0.1:8000"


def main() -> int:
    # Log in as bootstrapped admin (password comes from REGIME_TRADER_ADMIN_BOOTSTRAP_PASSWORD).
    import os

    from dotenv import load_dotenv

    load_dotenv()
    pw = os.getenv("REGIME_TRADER_ADMIN_BOOTSTRAP_PASSWORD", "regime-admin")
    login = httpx.post(f"{BASE}/auth/login", json={"username": "admin", "password": pw})
    print(f"POST /auth/login -> {login.status_code}")
    token = login.json().get("access_token")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    for path in ("/health", "/info", "/portfolio", "/positions", "/regime/current", "/freshness"):
        r = httpx.get(f"{BASE}{path}", headers=headers, timeout=10)
        print(f"\nGET {path} -> {r.status_code}")
        try:
            print(json.dumps(r.json(), indent=2, default=str)[:1200])
        except Exception:
            print(r.text[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
