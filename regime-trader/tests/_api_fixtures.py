"""Shared fixtures to boot the FastAPI app inside a tmp workspace."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from api.app import create_app
from api.auth import issue_access_token
from api.services import ApprovalPolicy
from config.loader import bootstrap_project
from monitoring.application import TradingApplication


def _settings_payload(tmp_path: Path, approval_mode: str = "manual") -> dict:
    return {
        "broker": {
            "trading_mode": "paper",
            "execution_enabled": False,
            "symbols": ["SPY", "QQQ"],
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
            "initial_capital": 100000,
            "train_window": 504,
            "test_window": 126,
            "step_size": 126,
            "risk_free_rate": 0.045,
        },
        "monitoring": {"dashboard_refresh_seconds": 5, "alert_rate_limit_minutes": 15},
        "platform": {
            "api_host": "127.0.0.1",
            "api_port": 8000,
            "approval_mode": approval_mode,
            "live_arming_ttl_minutes": 15,
            "storage_backend": "sqlite",
            "sqlite_path": str(tmp_path / "state" / "test.db"),
            "state_dir": str(tmp_path / "state"),
            "audit_dir": str(tmp_path / "state" / "audit"),
            "snapshot_dir": str(tmp_path / "state" / "snapshots"),
            "approval_dir": str(tmp_path / "state" / "approvals"),
            "require_approval_in_paper": approval_mode == "manual",
        },
        "governance": {"model_registry_path": str(tmp_path / "state" / "models")},
        "agent": {"default_permission": "agent_preview"},
    }


def boot_client(
    tmp_path: Path,
    *,
    approval_mode: str = "manual",
    fresh_data: bool = True,
) -> tuple[TestClient, object]:
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(yaml.safe_dump(_settings_payload(tmp_path, approval_mode=approval_mode)), encoding="utf-8")
    env_path = tmp_path / ".env.empty"
    env_path.write_text(
        "REGIME_TRADER_JWT_SECRET=test-secret\nREGIME_TRADER_ADMIN_BOOTSTRAP_PASSWORD=admin-pw\n",
        encoding="utf-8",
    )
    cfg = bootstrap_project(settings_path=settings_path, env_path=env_path)
    application = TradingApplication(cfg, dry_run=True)
    app = create_app(cfg, application=application)
    # Ensure deterministic policy regardless of yaml value.
    service = app.state.service
    service.approval_policy = ApprovalPolicy(
        mode=approval_mode,
        require_approval_in_paper=approval_mode == "manual",
    )
    if fresh_data:
        # The default in-memory market-data provider reports no bars, which makes
        # the freshness payload always "stale" and therefore blocks every agent
        # write in tests. Tests that need stale behaviour override this on the
        # service instance explicitly.
        _install_fresh_freshness_stub(service)
    client = TestClient(app)
    return client, app


def _install_fresh_freshness_stub(service) -> None:
    """Replace ``service.get_freshness`` with a fresh-market stub.

    Tests that want to exercise stale-data behaviour reassign
    ``service.get_freshness`` after the fixture runs.
    """
    from datetime import datetime, timezone

    def _fresh():
        now = datetime.now(timezone.utc).isoformat()
        return {
            "exchange_timezone": "America/New_York",
            "exchange_session_state": "open",
            "last_completed_daily_bar_time": now,
            "last_completed_intraday_bar_time": now,
            "data_freshness_status": "fresh",
            "daily_data_stale": False,
            "intraday_data_stale": False,
            "regime_effective_session_date": now[:10],
            "stale_data_blocked": False,
            "now": now,
        }

    service.get_freshness = _fresh  # type: ignore[assignment]


def admin_token(app) -> str:
    auth_settings = app.state.auth_settings
    return issue_access_token(auth_settings, "admin", role="admin")


def operator_token(app) -> str:
    auth_settings = app.state.auth_settings
    return issue_access_token(auth_settings, "operator", role="operator")
