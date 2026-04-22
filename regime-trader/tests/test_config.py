"""Phase A1 - config loader tests."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from config.loader import (
    ConfigError,
    bootstrap_project,
    load_secrets,
    load_settings,
    validate_config,
)


def _write_settings(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _baseline_settings() -> dict:
    return {
        "broker": {
            "trading_mode": "paper",
            "execution_enabled": False,
            "symbols": ["SPY"],
            "regime_timeframe": "1Day",
            "execution_timeframe": "5Min",
        },
        "hmm": {
            "n_candidates": [3, 4, 5],
            "n_init": 5,
            "covariance_type": "full",
            "min_train_bars": 504,
            "stability_bars": 3,
            "flicker_window": 20,
            "flicker_threshold": 4,
            "min_confidence": 0.55,
        },
        "strategy": {
            "low_vol_allocation": 0.95,
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
            "approval_mode": "manual",
            "live_arming_ttl_minutes": 15,
            "storage_backend": "sqlite",
            "sqlite_path": "state/test.db",
            "state_dir": "state",
        },
    }


def test_load_settings_reads_yaml(tmp_path: Path) -> None:
    path = _write_settings(tmp_path / "settings.yaml", _baseline_settings())
    data = load_settings(path)
    assert data["broker"]["trading_mode"] == "paper"
    assert data["hmm"]["min_train_bars"] == 504


def test_missing_secret_raises_when_execution_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _baseline_settings()
    settings["broker"]["execution_enabled"] = True
    _write_settings(tmp_path / "settings.yaml", settings)
    for key in ("ALPACA_PAPER_API_KEY", "ALPACA_PAPER_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    secrets = load_secrets(env_path=tmp_path / ".env.missing")
    with pytest.raises(ConfigError):
        validate_config(settings, secrets)


def test_runtime_override_takes_precedence(tmp_path: Path) -> None:
    path = _write_settings(tmp_path / "settings.yaml", _baseline_settings())
    data = load_settings(path, overrides={"broker": {"trading_mode": "live"}})
    assert data["broker"]["trading_mode"] == "live"
    # Other keys preserved
    assert data["broker"]["execution_enabled"] is False


def test_env_used_only_for_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("ALPACA_PAPER_API_KEY=abc\nALPACA_PAPER_SECRET_KEY=def\n", encoding="utf-8")
    for key in ("ALPACA_PAPER_API_KEY", "ALPACA_PAPER_SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    secrets = load_secrets(env_path=env_file)
    assert secrets.alpaca_paper_api_key == "abc"
    assert secrets.alpaca_paper_secret_key == "def"
    # Behavior config not leaking into secrets
    assert not hasattr(secrets, "symbols")


def test_validate_rejects_impossible_drawdown(tmp_path: Path) -> None:
    settings = _baseline_settings()
    settings["risk"]["daily_dd_halt"] = 0.01  # below reduce -> invalid
    with pytest.raises(ConfigError):
        validate_config(settings, load_secrets(env_path=tmp_path / ".env.none"))


def test_bootstrap_project_creates_state_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _baseline_settings()
    state_dir = tmp_path / "state"
    settings["platform"]["state_dir"] = str(state_dir)
    settings["platform"]["audit_dir"] = str(state_dir / "audit")
    settings["platform"]["snapshot_dir"] = str(state_dir / "snapshots")
    settings["platform"]["approval_dir"] = str(state_dir / "approvals")
    settings["governance"] = {"model_registry_path": str(state_dir / "models")}
    path = _write_settings(tmp_path / "settings.yaml", settings)
    cfg = bootstrap_project(settings_path=path, env_path=tmp_path / ".env.none")
    assert cfg.get("broker.trading_mode") == "paper"
    for sub in ("audit", "snapshots", "approvals", "models"):
        assert (state_dir / sub).is_dir()
