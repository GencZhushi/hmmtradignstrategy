"""Verify the dashboard static assets and the AlpacaDataProvider live fetch."""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from config import bootstrap_project
from data.market_data import build_provider
from monitoring.application import build_data_config

BASE = "http://127.0.0.1:8000"


def main() -> int:
    status_app_js = httpx.get(f"{BASE}/web/static/app.js").status_code
    html = httpx.get(f"{BASE}/").text
    has_script = "/web/static/app.js" in html
    has_all_tabs = all(f'data-tab="{t}"' in html for t in ("overview", "positions", "signals", "approvals", "audit", "settings"))
    print(f"GET /web/static/app.js -> {status_app_js}  (expect 200)")
    print(f"HTML references /web/static/app.js -> {has_script}")
    print(f"HTML has all six tab buttons       -> {has_all_tabs}")

    print("\n--- Live AlpacaDataProvider fetch ---")
    cfg = bootstrap_project(strict_secrets=True)
    state_dir = Path(cfg.source_path).parent.parent / "state"
    provider = build_provider(build_data_config(cfg, state_dir))
    print(f"Provider class: {type(provider).__name__}")
    end = datetime.now(timezone.utc) - timedelta(days=1)
    start = end - timedelta(days=800)
    daily = provider.daily_bars("SPY", start, end)
    print(f"SPY daily bars fetched: {len(daily)} rows")
    if not daily.empty:
        print(f"  first bar: {daily.index[0].date()}  close={float(daily.iloc[0]['close']):.2f}")
        print(f"  last  bar: {daily.index[-1].date()} close={float(daily.iloc[-1]['close']):.2f}")
    end_ix = datetime.now(timezone.utc) - timedelta(minutes=20)
    start_ix = end_ix - timedelta(hours=6)
    intraday = provider.intraday_bars("SPY", start_ix, end_ix)
    print(f"SPY 5Min bars fetched:  {len(intraday)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
