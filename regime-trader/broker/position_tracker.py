"""Position tracker: mirrors broker positions locally and reconciles on startup.

Responsibilities:

- Maintain the authoritative ``PortfolioState`` object the engine consumes.
- Apply fills (full or partial) to local state atomically.
- Reconcile the local view against the broker on startup/reconnect.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from broker.alpaca_client import BrokerProtocol
from core.types import Position, PortfolioState

LOG = logging.getLogger(__name__)


@dataclass
class PositionTracker:
    """Keeps local portfolio/position state aligned with the broker."""

    broker: BrokerProtocol | None = None
    initial_equity: float = 100_000.0
    state: PortfolioState = field(init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    _price_cache: dict[str, float] = field(default_factory=dict, init=False)
    _daily_reset_ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc), init=False)
    _weekly_reset_ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc), init=False)

    def __post_init__(self) -> None:
        self.state = PortfolioState(
            equity=self.initial_equity,
            cash=self.initial_equity,
            buying_power=self.initial_equity,
        )

    # ------------------------------------------------------------------ snapshots
    def snapshot(self) -> PortfolioState:
        with self._lock:
            return self.state

    def current_prices(self) -> dict[str, float]:
        with self._lock:
            return dict(self._price_cache)

    # ------------------------------------------------------------------ fills
    def apply_fill(
        self,
        *,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        stop_price: float | None = None,
        regime_name: str | None = None,
    ) -> Position:
        if qty <= 0 or price <= 0:
            raise ValueError("Fill quantity and price must be positive")
        with self._lock:
            position = self.state.positions.get(symbol)
            sign = 1 if side.upper() == "BUY" else -1
            new_qty = (position.quantity if position else 0.0) + sign * qty
            if position is None and sign == -1:
                raise ValueError(f"Cannot sell {symbol}: no position on record")
            if new_qty < 0:
                raise ValueError(f"Fill would produce short position on {symbol}; engine is long-only")
            if new_qty == 0:
                self.state.positions.pop(symbol, None)
                self.state.cash += price * qty
            else:
                if position is None:
                    position = Position(
                        symbol=symbol,
                        quantity=new_qty,
                        avg_entry_price=price,
                        current_price=price,
                        stop_price=stop_price,
                        regime_at_entry=regime_name,
                    )
                else:
                    total_cost = position.avg_entry_price * position.quantity + sign * price * qty
                    position.avg_entry_price = total_cost / new_qty if new_qty else price
                    position.quantity = new_qty
                    position.current_price = price
                    if stop_price is not None:
                        position.stop_price = stop_price
                    if regime_name is not None and position.regime_at_entry is None:
                        position.regime_at_entry = regime_name
                self.state.positions[symbol] = position
                self.state.cash -= sign * price * qty
            self._price_cache[symbol] = price
            self.state.daily_trade_count += 1
            self._recompute_metrics()
            return self.state.positions.get(symbol, position)

    def update_price(self, symbol: str, price: float) -> None:
        with self._lock:
            self._price_cache[symbol] = price
            if symbol in self.state.positions:
                self.state.positions[symbol].current_price = price
            self._recompute_metrics()

    def reset_daily(self) -> None:
        with self._lock:
            self.state.daily_pnl = 0.0
            self.state.daily_trade_count = 0
            self._daily_reset_ts = datetime.now(timezone.utc)

    def reset_weekly(self) -> None:
        with self._lock:
            self.state.weekly_pnl = 0.0
            self._weekly_reset_ts = datetime.now(timezone.utc)

    # ------------------------------------------------------------------ reconciliation
    def sync_positions(self) -> list[Position]:
        if self.broker is None:
            return list(self.state.positions.values())
        with self._lock:
            broker_view = {pos.get("symbol"): pos for pos in self.broker.list_positions()}
            local_view = dict(self.state.positions)

            for symbol, pos in broker_view.items():
                qty = float(pos.get("qty", 0.0))
                price = float(pos.get("current_price") or pos.get("avg_entry_price") or 0.0)
                if qty <= 0:
                    local_view.pop(symbol, None)
                    continue
                avg_entry = float(pos.get("avg_entry_price", price or 0.0))
                position = Position(
                    symbol=symbol,
                    quantity=qty,
                    avg_entry_price=avg_entry,
                    current_price=price or avg_entry,
                    stop_price=pos.get("stop_price"),
                    regime_at_entry=pos.get("regime_at_entry"),
                )
                local_view[symbol] = position

            removed = [sym for sym in local_view if sym not in broker_view]
            for sym in removed:
                LOG.info("Position %s disappeared from broker - removing locally", sym)
                local_view.pop(sym, None)

            self.state.positions = local_view
            self._recompute_metrics()
            return list(self.state.positions.values())

    def reconcile_from_orders(self, broker_orders: Iterable[Mapping[str, Any]]) -> None:
        """Update last-known prices/stops from broker order records."""
        with self._lock:
            for order in broker_orders:
                symbol = order.get("symbol")
                fill_price = order.get("avg_fill_price") or order.get("limit_price")
                if symbol and fill_price:
                    self._price_cache[symbol] = float(fill_price)
            self._recompute_metrics()

    # ------------------------------------------------------------------ metrics
    def _recompute_metrics(self) -> None:
        equity = self.state.cash + sum(pos.market_value for pos in self.state.positions.values())
        self.state.equity = equity
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        self.state.drawdown = (
            (self.state.peak_equity - equity) / self.state.peak_equity
            if self.state.peak_equity > 0
            else 0.0
        )
        self.state.buying_power = max(self.state.cash, 0.0) * 4

    # ------------------------------------------------------------------ snapshots
    def dump_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "equity": self.state.equity,
                "cash": self.state.cash,
                "peak_equity": self.state.peak_equity,
                "daily_pnl": self.state.daily_pnl,
                "weekly_pnl": self.state.weekly_pnl,
                "drawdown": self.state.drawdown,
                "positions": {
                    sym: {
                        "symbol": pos.symbol,
                        "quantity": pos.quantity,
                        "avg_entry_price": pos.avg_entry_price,
                        "current_price": pos.current_price,
                        "stop_price": pos.stop_price,
                        "regime_at_entry": pos.regime_at_entry,
                    }
                    for sym, pos in self.state.positions.items()
                },
            }

    def load_state(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            self.state.equity = float(payload.get("equity", self.state.equity))
            self.state.cash = float(payload.get("cash", self.state.cash))
            self.state.peak_equity = float(payload.get("peak_equity", self.state.equity))
            self.state.daily_pnl = float(payload.get("daily_pnl", 0.0))
            self.state.weekly_pnl = float(payload.get("weekly_pnl", 0.0))
            self.state.drawdown = float(payload.get("drawdown", 0.0))
            self.state.positions = {
                sym: Position(
                    symbol=data["symbol"],
                    quantity=float(data["quantity"]),
                    avg_entry_price=float(data["avg_entry_price"]),
                    current_price=float(data.get("current_price", data["avg_entry_price"])),
                    stop_price=data.get("stop_price"),
                    regime_at_entry=data.get("regime_at_entry"),
                )
                for sym, data in dict(payload.get("positions", {})).items()
            }
            self._recompute_metrics()
