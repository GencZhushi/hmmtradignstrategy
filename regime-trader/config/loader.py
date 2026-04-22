"""Configuration precedence: .env (secrets) < settings.yaml < CLI overrides.

Only secrets live in .env. Runtime behavior lives in settings.yaml. CLI flags
override settings.yaml for the current process only.
"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml
from dotenv import load_dotenv


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "settings.yaml"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Secrets:
    """Secrets loaded from .env (never from settings.yaml)."""

    alpaca_paper_api_key: str | None = None
    alpaca_paper_secret_key: str | None = None
    alpaca_live_api_key: str | None = None
    alpaca_live_secret_key: str | None = None
    jwt_secret: str | None = None
    admin_bootstrap_password: str | None = None
    service_token: str | None = None
    openclaw_service_token: str | None = None

    def credentials_for(self, mode: str) -> tuple[str | None, str | None]:
        if mode == "live":
            return self.alpaca_live_api_key, self.alpaca_live_secret_key
        return self.alpaca_paper_api_key, self.alpaca_paper_secret_key


@dataclass
class AppConfig:
    """Typed view over loaded settings plus loaded secrets."""

    raw: dict[str, Any]
    secrets: Secrets = field(default_factory=Secrets)
    source_path: Path | None = None

    def section(self, name: str) -> dict[str, Any]:
        if name not in self.raw:
            raise KeyError(f"Missing config section: {name}")
        value = self.raw[name]
        if not isinstance(value, dict):
            raise TypeError(f"Config section '{name}' must be a mapping, got {type(value).__name__}")
        return value

    def get(self, dotted_path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted_path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, Mapping)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_settings(
    path: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load settings.yaml and apply optional CLI-style overrides (highest precedence)."""
    settings_path = Path(path) if path else DEFAULT_SETTINGS_PATH
    if not settings_path.exists():
        raise FileNotFoundError(f"settings.yaml not found at {settings_path}")
    with settings_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError("settings.yaml must contain a mapping at the top level")
    if overrides:
        data = _deep_merge(data, overrides)
    return data


def load_secrets(env_path: Path | str | None = None) -> Secrets:
    """Load secrets from a .env file without leaking them into process-wide env."""
    if env_path is not None:
        load_dotenv(dotenv_path=Path(env_path), override=False)
    else:
        load_dotenv(override=False)
    return Secrets(
        alpaca_paper_api_key=os.getenv("ALPACA_PAPER_API_KEY"),
        alpaca_paper_secret_key=os.getenv("ALPACA_PAPER_SECRET_KEY"),
        alpaca_live_api_key=os.getenv("ALPACA_LIVE_API_KEY"),
        alpaca_live_secret_key=os.getenv("ALPACA_LIVE_SECRET_KEY"),
        jwt_secret=os.getenv("REGIME_TRADER_JWT_SECRET"),
        admin_bootstrap_password=os.getenv("REGIME_TRADER_ADMIN_BOOTSTRAP_PASSWORD"),
        service_token=os.getenv("REGIME_TRADER_SERVICE_TOKEN"),
        openclaw_service_token=os.getenv("OPENCLAW_SERVICE_TOKEN"),
    )


REQUIRED_SECTIONS: tuple[str, ...] = (
    "broker",
    "hmm",
    "strategy",
    "risk",
    "backtest",
    "monitoring",
    "platform",
)

REQUIRED_NUMERIC: dict[str, tuple[str, ...]] = {
    "hmm": ("min_train_bars", "stability_bars", "flicker_window", "flicker_threshold", "min_confidence"),
    "strategy": ("low_vol_allocation", "rebalance_threshold", "uncertainty_size_mult"),
    "risk": (
        "max_risk_per_trade",
        "max_exposure",
        "max_leverage",
        "max_single_position",
        "max_concurrent",
        "max_daily_trades",
        "daily_dd_reduce",
        "daily_dd_halt",
        "weekly_dd_reduce",
        "weekly_dd_halt",
        "max_dd_from_peak",
    ),
    "backtest": ("train_window", "test_window", "step_size", "initial_capital"),
}


class ConfigError(ValueError):
    """Raised when the loaded configuration is invalid."""


def _require_positive(section: str, key: str, value: Any) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigError(f"{section}.{key} must be numeric, got {type(value).__name__}")
    if value <= 0:
        raise ConfigError(f"{section}.{key} must be > 0, got {value}")


def validate_config(settings: Mapping[str, Any], secrets: Secrets | None = None) -> None:
    """Raise ConfigError on any invalid settings or missing critical secrets.

    Secret checks only enforce paper credentials when execution is enabled in paper mode.
    """
    for section in REQUIRED_SECTIONS:
        if section not in settings:
            raise ConfigError(f"Missing required settings section: {section}")
        if not isinstance(settings[section], Mapping):
            raise ConfigError(f"Settings section '{section}' must be a mapping")

    for section, keys in REQUIRED_NUMERIC.items():
        for key in keys:
            if key not in settings[section]:
                raise ConfigError(f"Missing numeric setting {section}.{key}")
            _require_positive(section, key, settings[section][key])

    risk = settings["risk"]
    if risk["daily_dd_reduce"] >= risk["daily_dd_halt"]:
        raise ConfigError("risk.daily_dd_reduce must be < risk.daily_dd_halt")
    if risk["weekly_dd_reduce"] >= risk["weekly_dd_halt"]:
        raise ConfigError("risk.weekly_dd_reduce must be < risk.weekly_dd_halt")
    if risk["max_dd_from_peak"] <= 0:
        raise ConfigError("risk.max_dd_from_peak must be > 0")

    backtest = settings["backtest"]
    if backtest["train_window"] < 252:
        raise ConfigError("backtest.train_window must be >= 252 completed daily bars")
    if backtest["step_size"] <= 0 or backtest["test_window"] <= 0:
        raise ConfigError("backtest.step_size and test_window must be positive")

    hmm = settings["hmm"]
    candidates = hmm.get("n_candidates")
    if not isinstance(candidates, Iterable) or isinstance(candidates, (str, bytes)):
        raise ConfigError("hmm.n_candidates must be a list of integers")
    candidates_list = list(candidates)
    if not candidates_list or any(not isinstance(c, int) or c < 2 for c in candidates_list):
        raise ConfigError("hmm.n_candidates must contain integers >= 2")

    broker = settings["broker"]
    if broker.get("trading_mode") not in {"paper", "live"}:
        raise ConfigError("broker.trading_mode must be 'paper' or 'live'")

    if secrets is not None and broker.get("execution_enabled"):
        api_key, secret_key = secrets.credentials_for(broker["trading_mode"])
        if not api_key or not secret_key:
            raise ConfigError(
                f"Missing required Alpaca {broker['trading_mode']} credentials. "
                "Set ALPACA_*_API_KEY and ALPACA_*_SECRET_KEY in .env."
            )


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def bootstrap_project(
    settings_path: Path | str | None = None,
    env_path: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
    strict_secrets: bool = False,
) -> AppConfig:
    """Load config and secrets, validate, and create required state directories.

    When ``strict_secrets`` is True, missing secrets for the selected broker mode raise.
    When False (default), validation skips the secret requirement so unit tests and
    dry-runs can still bootstrap.
    """
    settings = load_settings(settings_path, overrides=overrides)
    secrets = load_secrets(env_path)
    validate_config(settings, secrets if strict_secrets else None)

    platform = settings.get("platform", {})
    state_root = Path(platform.get("state_dir", "state"))
    if not state_root.is_absolute():
        state_root = PROJECT_ROOT / state_root
    for subdir in (
        state_root,
        Path(platform.get("audit_dir", state_root / "audit")),
        Path(platform.get("snapshot_dir", state_root / "snapshots")),
        Path(platform.get("approval_dir", state_root / "approvals")),
    ):
        path = subdir if subdir.is_absolute() else PROJECT_ROOT / subdir
        _ensure_dir(path)

    governance = settings.get("governance", {})
    registry = Path(governance.get("model_registry_path", state_root / "models"))
    if not registry.is_absolute():
        registry = PROJECT_ROOT / registry
    _ensure_dir(registry)

    return AppConfig(
        raw=settings,
        secrets=secrets,
        source_path=Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH,
    )
