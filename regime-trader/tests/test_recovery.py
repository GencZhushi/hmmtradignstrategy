"""Phase A8 - orchestrator state snapshot + restart safety."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from broker.alpaca_client import SimulatedBroker
from broker.position_tracker import PositionTracker
from config.loader import bootstrap_project
from monitoring.application import TradingApplication


def _write_settings(tmp_path: Path) -> Path:
    path = tmp_path / "settings.yaml"
    payload = {
        "broker": {
            "trading_mode": "paper",
            "execution_enabled": False,
            "symbols": ["SPY"],
            "regime_timeframe": "1Day",
            "execution_timeframe": "5Min",
            "exchange_timezone": "America/New_York",
        },
        "hmm": {
            "n_candidates": [3, 4],
            "n_init": 2,
            "covariance_type": "full",
            "min_train_bars": 504,
            "stability_bars": 3,
            "flicker_window": 20,
            "flicker_threshold": 4,
            "min_confidence": 0.55,
            "zscore_window": 252,
        },
        "strategy": {
            "low_vol_allocation": 0.95,
            "mid_vol_allocation_trend": 0.95,
            "mid_vol_allocation_no_trend": 0.60,
            "high_vol_allocation": 0.60,
            "low_vol_leverage": 1.25,
            "rebalance_threshold": 0.10,
            "uncertainty_size_mult": 0.50,
        },
        "risk": {
            "max_risk_per_trade": 0.01,
            "max_exposure": 0.80,
            "max_leverage": 1.25,
            "max_single_position": 0.15,
            "max_concurrent": 5,
            "max_daily_trades": 20,
            "daily_dd_reduce": 0.02,
            "daily_dd_halt": 0.03,
            "weekly_dd_reduce": 0.05,
            "weekly_dd_halt": 0.07,
            "max_dd_from_peak": 0.10,
            "max_sector_exposure": 0.30,
            "correlation_reduce_threshold": 0.70,
            "correlation_reject_threshold": 0.85,
            "correlation_lookback_days": 60,
        },
        "backtest": {
            "slippage_pct": 0.0005,
            "initial_capital": 100_000,
            "train_window": 504,
            "test_window": 126,
            "step_size": 126,
            "risk_free_rate": 0.045,
        },
        "monitoring": {"dashboard_refresh_seconds": 5, "alert_rate_limit_minutes": 15},
        "platform": {
            "api_host": "127.0.0.1",
            "api_port": 8000,
            "approval_mode": "manual",
            "live_arming_ttl_minutes": 15,
            "storage_backend": "sqlite",
            "sqlite_path": str(tmp_path / "state" / "test.db"),
            "state_dir": str(tmp_path / "state"),
            "audit_dir": str(tmp_path / "state" / "audit"),
            "snapshot_dir": str(tmp_path / "state" / "snapshots"),
            "approval_dir": str(tmp_path / "state" / "approvals"),
        },
        "governance": {"model_registry_path": str(tmp_path / "state" / "models")},
        "agent": {"default_permission": "agent_preview"},
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_application_boots_and_saves_snapshot(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path)
    cfg = bootstrap_project(settings_path=settings_path, env_path=tmp_path / ".env.missing")
    app = TradingApplication(cfg, dry_run=True)
    app.position_tracker.apply_fill(symbol="SPY", side="BUY", qty=5, price=100.0, stop_price=95.0, regime_name="BULL")
    snapshot_path = app.save_state_snapshot()
    assert snapshot_path.exists()
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["portfolio"]["positions"]["SPY"]["quantity"] == pytest.approx(5)


def test_restart_rehydrates_positions(tmp_path: Path) -> None:
    settings_path = _write_settings(tmp_path)
    cfg = bootstrap_project(settings_path=settings_path, env_path=tmp_path / ".env.missing")
    first = TradingApplication(cfg, dry_run=True)
    first.position_tracker.apply_fill(symbol="SPY", side="BUY", qty=3, price=100.0, stop_price=95.0, regime_name="BULL")
    first.save_state_snapshot()
    # Restart with the same config - second instance should inherit positions.
    second = TradingApplication(cfg, dry_run=True)
    state = second.position_tracker.snapshot()
    assert state.positions["SPY"].quantity == pytest.approx(3)


def test_reconcile_after_restart_does_not_double_enter(tmp_path: Path) -> None:
    tracker = PositionTracker(broker=None, initial_equity=100_000.0)
    tracker.apply_fill(symbol="SPY", side="BUY", qty=4, price=100.0, stop_price=95.0)
    broker = SimulatedBroker()
    broker.positions["SPY"] = {"symbol": "SPY", "qty": 4.0, "avg_entry_price": 100.0, "current_price": 101.0}
    tracker.broker = broker
    tracker.sync_positions()
    assert tracker.snapshot().positions["SPY"].quantity == pytest.approx(4.0)
