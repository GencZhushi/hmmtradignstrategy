"""Thin wrapper around alpaca-py so the engine stays decoupled from the SDK.

The wrapper exposes only the methods the executor/tracker need. If ``alpaca-py``
is not installed (e.g. during unit tests), the class falls back to raising a
clear ``BrokerUnavailable`` error so downstream code can short-circuit.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

LOG = logging.getLogger(__name__)


class BrokerUnavailable(RuntimeError):
    """Raised when the broker SDK is missing or credentials are not configured."""


class BrokerProtocol(Protocol):
    """Subset of broker operations the engine relies on."""

    def is_market_open(self) -> bool: ...
    def get_account(self) -> dict[str, Any]: ...
    def list_positions(self) -> list[dict[str, Any]]: ...
    def list_orders(self, *, status: str | None = None) -> list[dict[str, Any]]: ...
    def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]: ...
    def cancel_order(self, order_id: str) -> None: ...
    def replace_order(self, order_id: str, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class AlpacaClient:
    """Paper/live-aware Alpaca client with retry + reconnection logic."""

    api_key: str | None
    secret_key: str | None
    paper: bool = True
    base_url_override: str | None = None
    max_retries: int = 3
    backoff_seconds: float = 0.5
    _trading_client: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key or not self.secret_key:
            self._trading_client = None
            return
        try:
            from alpaca.trading.client import TradingClient  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            LOG.info("alpaca-py not installed; AlpacaClient in offline mode (%s)", exc)
            self._trading_client = None
            return
        self._trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=self.paper,
            url_override=self.base_url_override,
        )

    @property
    def is_connected(self) -> bool:
        return self._trading_client is not None

    def _retry(self, fn, *args, **kwargs):
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - network path
                last_exc = exc
                LOG.warning("Alpaca call failed (attempt %d/%d): %s", attempt, self.max_retries, exc)
                time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        raise BrokerUnavailable(f"Alpaca call failed after retries: {last_exc}")

    # -------------------------------------------------- read calls
    def is_market_open(self) -> bool:
        if not self.is_connected:
            raise BrokerUnavailable("Alpaca client not connected")
        clock = self._retry(self._trading_client.get_clock)  # pragma: no cover
        return bool(getattr(clock, "is_open", False))  # pragma: no cover

    def get_account(self) -> dict[str, Any]:
        if not self.is_connected:
            raise BrokerUnavailable("Alpaca client not connected")
        account = self._retry(self._trading_client.get_account)  # pragma: no cover
        return _safe_dict(account)  # pragma: no cover

    def list_positions(self) -> list[dict[str, Any]]:
        if not self.is_connected:
            raise BrokerUnavailable("Alpaca client not connected")
        positions = self._retry(self._trading_client.get_all_positions)  # pragma: no cover
        return [_safe_dict(p) for p in positions]  # pragma: no cover

    def list_orders(self, *, status: str | None = None) -> list[dict[str, Any]]:
        if not self.is_connected:
            raise BrokerUnavailable("Alpaca client not connected")
        from alpaca.trading.requests import GetOrdersRequest  # type: ignore  # pragma: no cover

        req = GetOrdersRequest(status=status) if status else GetOrdersRequest()  # pragma: no cover
        orders = self._retry(self._trading_client.get_orders, filter=req)  # pragma: no cover
        return [_safe_dict(o) for o in orders]  # pragma: no cover

    # -------------------------------------------------- mutating calls
    def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise BrokerUnavailable("Alpaca client not connected")
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, StopLossRequest, TakeProfitRequest  # type: ignore  # pragma: no cover
        from alpaca.trading.enums import OrderSide, TimeInForce  # type: ignore  # pragma: no cover

        side = OrderSide.BUY if str(payload["side"]).upper() == "BUY" else OrderSide.SELL  # pragma: no cover
        tif = TimeInForce.DAY  # pragma: no cover
        stop_req = None  # pragma: no cover
        if payload.get("stop_price"):  # pragma: no cover
            stop_req = StopLossRequest(stop_price=payload["stop_price"])  # pragma: no cover
        tp_req = None  # pragma: no cover
        if payload.get("take_profit"):  # pragma: no cover
            tp_req = TakeProfitRequest(limit_price=payload["take_profit"])  # pragma: no cover
        if payload.get("limit_price"):  # pragma: no cover
            request = LimitOrderRequest(  # pragma: no cover
                symbol=payload["symbol"],
                qty=payload["qty"],
                side=side,
                time_in_force=tif,
                limit_price=payload["limit_price"],
                stop_loss=stop_req,
                take_profit=tp_req,
            )
        else:  # pragma: no cover
            request = MarketOrderRequest(
                symbol=payload["symbol"],
                qty=payload["qty"],
                side=side,
                time_in_force=tif,
                stop_loss=stop_req,
                take_profit=tp_req,
            )
        order = self._retry(self._trading_client.submit_order, request)  # pragma: no cover
        return _safe_dict(order)  # pragma: no cover

    def cancel_order(self, order_id: str) -> None:
        if not self.is_connected:
            raise BrokerUnavailable("Alpaca client not connected")
        self._retry(self._trading_client.cancel_order_by_id, order_id)  # pragma: no cover

    def replace_order(self, order_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise BrokerUnavailable("Alpaca client not connected")
        from alpaca.trading.requests import ReplaceOrderRequest  # type: ignore  # pragma: no cover

        request = ReplaceOrderRequest(**payload)  # pragma: no cover
        order = self._retry(self._trading_client.replace_order_by_id, order_id, request)  # pragma: no cover
        return _safe_dict(order)  # pragma: no cover


def _safe_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()  # pragma: no cover - pydantic v2
    if hasattr(obj, "dict"):
        return obj.dict()  # pragma: no cover - pydantic v1
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_") and not callable(getattr(obj, k))}


# ---------------------------------------------------------- simulated broker
@dataclass
class SimulatedBroker:
    """Deterministic in-memory broker used by dry-run and the test suite."""

    orders: list[dict[str, Any]] = field(default_factory=list)
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    cash: float = 100_000.0
    accepted_states: tuple[str, ...] = ("accepted", "filled")
    _counter: int = field(default=0, init=False)

    def is_market_open(self) -> bool:
        return True

    def get_account(self) -> dict[str, Any]:
        return {"cash": self.cash, "buying_power": self.cash * 4}

    def list_positions(self) -> list[dict[str, Any]]:
        return list(self.positions.values())

    def list_orders(self, *, status: str | None = None) -> list[dict[str, Any]]:
        if status is None:
            return list(self.orders)
        return [o for o in self.orders if o.get("status") == status]

    def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._counter += 1
        broker_order_id = f"sim-{self._counter:06d}"
        order = {
            "broker_order_id": broker_order_id,
            "symbol": payload["symbol"],
            "side": payload["side"],
            "qty": float(payload.get("qty", 0.0)),
            "limit_price": payload.get("limit_price"),
            "stop_price": payload.get("stop_price"),
            "take_profit": payload.get("take_profit"),
            "status": "accepted",
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "filled_qty": 0.0,
            "avg_fill_price": None,
        }
        self.orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> None:
        for order in self.orders:
            if order["broker_order_id"] == order_id:
                order["status"] = "canceled"

    def replace_order(self, order_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        for order in self.orders:
            if order["broker_order_id"] == order_id:
                order.update(payload)
                return order
        raise KeyError(order_id)

    def simulate_fill(self, broker_order_id: str, *, fill_qty: float, fill_price: float) -> None:
        for order in self.orders:
            if order["broker_order_id"] == broker_order_id:
                order["filled_qty"] = order.get("filled_qty", 0.0) + fill_qty
                order["avg_fill_price"] = fill_price
                if order["filled_qty"] >= order["qty"]:
                    order["status"] = "filled"
                else:
                    order["status"] = "partially_filled"
                symbol = order["symbol"]
                sign = 1 if str(order["side"]).upper() == "BUY" else -1
                pos = self.positions.setdefault(
                    symbol,
                    {"symbol": symbol, "qty": 0.0, "avg_entry_price": fill_price},
                )
                new_qty = pos["qty"] + sign * fill_qty
                if new_qty == 0:
                    self.positions.pop(symbol, None)
                else:
                    pos["qty"] = new_qty
                    pos["avg_entry_price"] = fill_price
                return
