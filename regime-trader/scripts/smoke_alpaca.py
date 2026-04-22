"""One-shot smoke test: verify .env credentials connect to Alpaca paper."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker.alpaca_client import AlpacaClient
from config import bootstrap_project


def main() -> int:
    cfg = bootstrap_project(strict_secrets=True)
    api_key, secret = cfg.secrets.credentials_for("paper")
    print(f"Key loaded: {bool(api_key)} | Secret loaded: {bool(secret)}")
    client = AlpacaClient(api_key=api_key, secret_key=secret, paper=True)
    print(f"SDK connected: {client.is_connected}")
    try:
        acct = client.get_account()
    except Exception as exc:  # pragma: no cover - network
        print(f"get_account() FAILED: {exc}")
        return 1
    print("--- Account ---")
    print(f"  status      : {acct.get('status')}")
    print(f"  cash        : {acct.get('cash')}")
    print(f"  buying_power: {acct.get('buying_power')}")
    print(f"  equity      : {acct.get('equity')}")
    print(f"  portfolio_value: {acct.get('portfolio_value')}")
    print(f"  pattern_day_trader: {acct.get('pattern_day_trader')}")
    print(f"Market open : {client.is_market_open()}")
    positions = client.list_positions()
    print(f"Positions   : {len(positions)}")
    for p in positions[:5]:
        print(f"  - {p.get('symbol')} qty={p.get('qty')} avg={p.get('avg_entry_price')}")
    open_orders = client.list_orders(status="open")
    print(f"Open orders : {len(open_orders)}")
    print("OK - paper broker reachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
