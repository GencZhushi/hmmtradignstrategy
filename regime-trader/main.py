"""Regime Trader CLI entrypoint.

Boots the app with the configured precedence: .env secrets -> settings.yaml ->
CLI overrides, then dispatches into a sub-command.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from config import bootstrap_project

LOG = logging.getLogger("regime_trader.main")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="regime-trader", description="HMM-regime trading engine")
    parser.add_argument("--settings", type=Path, default=None, help="Path to settings.yaml override")
    parser.add_argument("--env", type=Path, default=None, help="Path to .env override")
    parser.add_argument("--trading-mode", choices=["paper", "live"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Full pipeline without broker calls")
    parser.add_argument("--backtest", action="store_true", help="Run walk-forward backtester")
    parser.add_argument("--train-only", action="store_true", help="Train HMM then exit")
    parser.add_argument("--serve-api", action="store_true", help="Launch FastAPI web/API platform")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser


def _apply_cli_overrides(args: argparse.Namespace) -> dict | None:
    overrides: dict = {}
    if args.trading_mode:
        overrides.setdefault("broker", {})["trading_mode"] = args.trading_mode
    if args.host or args.port:
        overrides.setdefault("platform", {})
        if args.host:
            overrides["platform"]["api_host"] = args.host
        if args.port:
            overrides["platform"]["api_port"] = args.port
    return overrides or None


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    cfg = bootstrap_project(
        settings_path=args.settings,
        env_path=args.env,
        overrides=_apply_cli_overrides(args),
        strict_secrets=not (args.dry_run or args.backtest or args.train_only),
    )
    LOG.info(
        "Config loaded from %s | mode=%s execution_enabled=%s",
        cfg.source_path,
        cfg.get("broker.trading_mode"),
        cfg.get("broker.execution_enabled"),
    )

    if args.serve_api:
        from api.app import run_server

        host = cfg.get("platform.api_host", "127.0.0.1")
        port = int(cfg.get("platform.api_port", 8000))
        return run_server(cfg, host=host, port=port, dry_run=args.dry_run)

    if args.backtest:
        from backtest.backtester import run_backtest_cli

        return run_backtest_cli(cfg)

    if args.train_only:
        from core.hmm_engine import train_cli

        return train_cli(cfg)

    # Default: live/paper orchestration loop (dry-run skips broker mutations).
    from monitoring.application import TradingApplication

    app = TradingApplication(cfg, dry_run=args.dry_run)
    try:
        app.run()
    except KeyboardInterrupt:  # pragma: no cover - signal path
        LOG.warning("Interrupted - shutting down")
        app.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    sys.exit(main())
