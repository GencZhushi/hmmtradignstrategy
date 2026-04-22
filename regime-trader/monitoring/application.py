"""Top-level orchestration loop with startup reconciliation + state snapshots.

Ties together:

- Market data + features + HMM regime
- Strategy orchestrator
- Risk manager + execution coordinator
- Broker executor / position tracker
- Monitoring (logger, alerts, dashboard, snapshots)

The class is designed so it can be driven either by the CLI (``main.py``) or by
the FastAPI platform (Spec B) via the same interface.
"""
from __future__ import annotations

import json
import logging
import signal as _signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from broker.alpaca_client import AlpacaClient, SimulatedBroker
from broker.order_executor import OrderExecutor
from broker.position_tracker import PositionTracker
from config.loader import AppConfig
from core.execution_coordinator import ExecutionCoordinator
from core.hmm_engine import VolatilityRegimeHMM
from core.idempotency import IdempotencyStore
from core.lock_manager import LockManager
from core.model_registry import ModelRegistry
from core.order_state_machine import OrderStateMachine
from core.regime_strategies import StrategyConfig, StrategyOrchestrator
from core.risk_manager import RiskLimits, RiskManager
from core.sector_mapping import SectorClassifier
from core.signal_generator import SignalGenerator
from core.types import AuditEvent, PortfolioState
from data.exchange_calendar import ExchangeCalendar, freshness_payload
from data.feature_engineering import FeatureEngine
from data.market_data import MarketDataManager, build_provider
from monitoring.alerts import AlertDispatcher, console_sink
from monitoring.dashboard import build_snapshot, render_plain
from monitoring.logger import configure_logging, emit_event

LOG = logging.getLogger("regime_trader.main")


def build_data_config(config: AppConfig, state_dir: Path) -> dict[str, Any]:
    """Compose the market-data provider config by merging ``settings.yaml`` data
    section with the Alpaca credentials already loaded into ``Secrets``.

    Shared by ``TradingApplication`` and the CLI helpers
    (``core.hmm_engine.train_cli``, ``backtest.backtester.run_backtest_cli``)
    so they all pick the same provider for the active trading mode.

    Provider factory fields (all optional):
      - provider:          'alpaca' | 'csv' | 'auto' (default 'auto')
      - data_dir:          CSV fallback directory
      - cache_dir:         on-disk cache for Alpaca parquet files
      - alpaca_api_key:    pulled from .env paper/live credentials
      - alpaca_secret_key: pulled from .env paper/live credentials
      - alpaca_feed:       'iex' (free) | 'sip' (paid)
    """
    data_cfg: dict[str, Any] = dict(config.raw.get("data") or {})
    data_cfg.setdefault("data_dir", state_dir / "bars")
    data_cfg.setdefault("cache_dir", state_dir / "bars_cache")
    mode = str(config.get("broker.trading_mode", "paper"))
    api_key, secret_key = config.secrets.credentials_for(mode)
    if api_key and secret_key:
        data_cfg.setdefault("alpaca_api_key", api_key)
        data_cfg.setdefault("alpaca_secret_key", secret_key)
    return data_cfg


@dataclass
class TradingApplication:
    """Top-level orchestrator; safe to run or embed in the API platform."""

    config: AppConfig
    dry_run: bool = False
    poll_interval_seconds: float = 5.0
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _snapshot_path: Path = field(init=False)

    # runtime components (built in __post_init__)
    market_data: MarketDataManager = field(init=False)
    feature_engine: FeatureEngine = field(init=False)
    hmm: VolatilityRegimeHMM = field(init=False)
    orchestrator: StrategyOrchestrator = field(init=False)
    signal_generator: SignalGenerator = field(init=False)
    risk_manager: RiskManager = field(init=False)
    state_machine: OrderStateMachine = field(init=False)
    idempotency: IdempotencyStore = field(init=False)
    lock_manager: LockManager = field(init=False)
    position_tracker: PositionTracker = field(init=False)
    executor: OrderExecutor = field(init=False)
    coordinator: ExecutionCoordinator = field(init=False)
    alerts: AlertDispatcher = field(init=False)
    calendar: ExchangeCalendar = field(init=False)
    model_registry: ModelRegistry = field(init=False)
    audit_log: list[AuditEvent] = field(default_factory=list, init=False)
    event_listeners: list[Callable[[str, Mapping[str, Any]], None]] = field(default_factory=list, init=False)
    recent_signals: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        platform = self.config.section("platform")
        state_dir = Path(platform.get("state_dir", "state"))
        if not state_dir.is_absolute():
            state_dir = Path(self.config.source_path).parent.parent / state_dir if self.config.source_path else Path.cwd() / state_dir
        self._snapshot_path = Path(platform.get("snapshot_dir", state_dir / "snapshots")) / "engine_snapshot.json"
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        configure_logging(log_dir=state_dir / "logs")
        self.calendar = ExchangeCalendar(timezone=self.config.get("broker.exchange_timezone", "America/New_York"))

        self.market_data = MarketDataManager(provider=build_provider(self._build_data_config(state_dir)))
        self.feature_engine = FeatureEngine(zscore_window=int(self.config.get("hmm.zscore_window", 252)))

        self.model_registry = ModelRegistry(state_dir / "models")
        self.hmm = VolatilityRegimeHMM(
            n_candidates=tuple(self.config.get("hmm.n_candidates", (3, 4, 5, 6, 7))),
            n_init=int(self.config.get("hmm.n_init", 10)),
            covariance_type=str(self.config.get("hmm.covariance_type", "full")),
        )
        self._load_active_model()

        strategy_cfg = StrategyConfig.from_config(self.config.section("strategy"))
        try:
            initial_regimes = list(self.hmm.regimes)
        except RuntimeError:
            initial_regimes = []
        self.orchestrator = StrategyOrchestrator(config=strategy_cfg, regime_infos=initial_regimes)
        self.signal_generator = SignalGenerator(
            hmm=self.hmm,
            orchestrator=self.orchestrator,
            stability_bars=int(self.config.get("hmm.stability_bars", 3)),
            flicker_window=int(self.config.get("hmm.flicker_window", 20)),
            flicker_threshold=int(self.config.get("hmm.flicker_threshold", 4)),
            min_confidence=float(self.config.get("hmm.min_confidence", 0.55)),
        )

        sector_classifier = SectorClassifier()
        limits = RiskLimits.from_config(self.config.section("risk"))
        self.risk_manager = RiskManager(limits=limits, sector_classifier=sector_classifier)

        self.state_machine = OrderStateMachine()
        self.idempotency = IdempotencyStore(snapshot_path=state_dir / "idempotency.json")
        self.lock_manager = LockManager()
        self.position_tracker = PositionTracker(
            broker=None,
            initial_equity=float(self.config.get("backtest.initial_capital", 100_000)),
        )

        broker = self._build_broker()
        self.executor = OrderExecutor(broker=broker, dry_run=self.dry_run)
        self.position_tracker.broker = broker

        self.coordinator = ExecutionCoordinator(
            risk_manager=self.risk_manager,
            state_machine=self.state_machine,
            idempotency=self.idempotency,
            lock_manager=self.lock_manager,
            executor=self.executor,
            portfolio_provider=self.position_tracker.snapshot,
            market_price_provider=lambda symbol: self.position_tracker.current_prices().get(symbol, 0.0),
        )
        self.coordinator.register_listener(self._on_event)

        self.alerts = AlertDispatcher(rate_limit_minutes=int(self.config.get("monitoring.alert_rate_limit_minutes", 15)))
        self.alerts.register_sink(console_sink)

        if self._snapshot_path.exists():
            try:
                self.load_state_snapshot(self._snapshot_path)
                LOG.info("Recovered state from %s", self._snapshot_path)
            except Exception as exc:  # pragma: no cover - best effort
                LOG.warning("Failed to load state snapshot: %s", exc)

    # ---------------------------------------------------------- lifecycle
    def run(self) -> None:
        _install_signal_handlers(self)
        LOG.info("Starting engine (dry_run=%s mode=%s)", self.dry_run, self.config.get("broker.trading_mode"))
        self.reconcile_on_startup()
        poll = max(self.poll_interval_seconds, 0.5)
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # pragma: no cover - runtime safeguard
                LOG.exception("Engine tick failed: %s", exc)
                self.alerts.emit_alert("engine_tick_error", str(exc), severity="error")
            self._stop.wait(poll)
        self.shutdown()

    def shutdown(self) -> None:
        LOG.info("Shutting down engine")
        self._stop.set()
        try:
            self.save_state_snapshot(self._snapshot_path)
        except Exception as exc:  # pragma: no cover
            LOG.error("Failed to save snapshot on shutdown: %s", exc)
        emit_event("main", "engine_shutdown", trading_mode=self.config.get("broker.trading_mode"))

    def reconcile_on_startup(self) -> None:
        LOG.info("Performing startup reconciliation")
        try:
            self.position_tracker.sync_positions()
        except Exception as exc:
            LOG.warning("Position sync skipped: %s", exc)
        try:
            broker_orders = self.executor.broker.list_orders()
        except Exception as exc:
            LOG.debug("Order list unavailable during reconciliation: %s", exc)
            broker_orders = []
        self.coordinator.reconcile_after_reconnect(broker_orders=broker_orders)
        self._on_event("startup_reconciled", {
            "positions": len(self.position_tracker.snapshot().positions),
            "broker_orders": len(broker_orders),
            "active_model": self.model_registry.active_version,
        })

    def tick(self) -> None:
        """One orchestration step; keep lightweight so the API can call it too."""
        self._refresh_freshness()

    # ---------------------------------------------------------- model management
    def _load_active_model(self) -> None:
        active = self.model_registry.active_entry()
        if active is None:
            LOG.info("No active model registered yet; engine will train on demand")
            return
        try:
            self.hmm.load_model(Path(active.path))
        except Exception as exc:
            LOG.warning("Failed to load active model %s: %s", active.model_version, exc)
            fallback = self.model_registry.fallback_entry()
            if fallback is not None and fallback.model_version != active.model_version:
                LOG.info("Falling back to %s", fallback.model_version)
                self.hmm.load_model(Path(fallback.path))

    # ---------------------------------------------------------- snapshots
    def save_state_snapshot(self, path: Path | None = None) -> Path:
        path = path or self._snapshot_path
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio": self.position_tracker.dump_state(),
            "breaker_state": self.position_tracker.snapshot().breaker_state.value,
            "active_model": self.model_registry.active_version,
            "idempotency": self.idempotency.snapshot(),
            "orders": {
                order_id: self.state_machine.summary(order_id)
                for order_id in self.state_machine.orders
            },
            "recent_signals": self.recent_signals[-20:],
        }
        tmp = path.with_suffix(".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
        tmp.replace(path)
        return path

    def load_state_snapshot(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "portfolio" in payload:
            self.position_tracker.load_state(payload["portfolio"])
        if "idempotency" in payload:
            self.idempotency.restore(payload["idempotency"])
        self.recent_signals = list(payload.get("recent_signals", []))

    # ---------------------------------------------------------- helpers
    def emit_alert(self, event_type: str, message: str, *, severity: str = "info", context: Mapping[str, Any] | None = None) -> bool:
        return self.alerts.emit_alert(event_type, message, severity=severity, context=context)

    def _refresh_freshness(self) -> None:
        symbols = self.config.get("broker.symbols", [])
        payload = freshness_payload(
            last_daily_bar=_first_value(self.market_data.freshness_snapshot(symbols), "last_completed_daily_bar_time"),
            last_intraday_bar=_first_value(self.market_data.freshness_snapshot(symbols), "last_completed_intraday_bar_time"),
            now=datetime.now(timezone.utc),
            calendar=self.calendar,
        )
        self._on_event("freshness_refresh", payload)

    def _build_broker(self):
        mode = str(self.config.get("broker.trading_mode", "paper"))
        api_key, secret_key = self.config.secrets.credentials_for(mode)
        if self.dry_run or not api_key or not secret_key:
            LOG.info("Using SimulatedBroker (dry_run=%s has_keys=%s)", self.dry_run, bool(api_key))
            return SimulatedBroker(cash=float(self.config.get("backtest.initial_capital", 100_000)))
        return AlpacaClient(api_key=api_key, secret_key=secret_key, paper=mode == "paper")

    def _build_data_config(self, state_dir: Path) -> dict[str, Any]:
        return build_data_config(self.config, state_dir)

    def register_listener(self, listener: Callable[[str, Mapping[str, Any]], None]) -> None:
        self.event_listeners.append(listener)

    def _on_event(self, event: str, payload: Mapping[str, Any]) -> None:
        emit_event("main", event, **payload)
        for listener in list(self.event_listeners):
            try:
                listener(event, payload)
            except Exception as exc:  # pragma: no cover - listener isolation
                LOG.error("Listener failed on %s: %s", event, exc)


def _first_value(data: Mapping[str, Mapping[str, Any]], key: str) -> datetime | None:
    for entry in data.values():
        value = entry.get(key)
        if value:
            try:
                return datetime.fromisoformat(str(value))
            except ValueError:
                return None
    return None


def _install_signal_handlers(app: TradingApplication) -> None:
    def _handler(signum, frame):  # pragma: no cover - signal path
        app._stop.set()

    for sig_name in ("SIGINT", "SIGTERM"):
        if hasattr(_signal, sig_name):
            try:
                _signal.signal(getattr(_signal, sig_name), _handler)
            except ValueError:
                # signal only works in the main thread; API server starts us elsewhere
                pass
