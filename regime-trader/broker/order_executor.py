"""Order executor: the only call site that mutates broker state (Spec A7).

Works against any ``BrokerProtocol`` implementation — the live ``AlpacaClient``
or the ``SimulatedBroker`` used by tests and ``--dry-run``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol

from broker.alpaca_client import BrokerProtocol
from core.order_state_machine import OrderRecord

LOG = logging.getLogger(__name__)


class ExecutorError(RuntimeError):
    pass


@dataclass
class OrderExecutor:
    """Translate ``OrderRecord``s into broker calls + track submission status."""

    broker: BrokerProtocol
    dry_run: bool = False

    def submit_order(self, order: OrderRecord) -> str:
        if self.dry_run:
            broker_order_id = f"dry-{order.order_id}"
            LOG.info("[dry-run] submit_order -> %s", broker_order_id)
            return broker_order_id
        payload = _order_payload(order)
        response = self.broker.submit_order(payload)
        broker_order_id = str(response.get("broker_order_id") or response.get("id") or "")
        if not broker_order_id:
            raise ExecutorError(f"Broker did not return an order id: {response}")
        LOG.info("Submitted order %s -> %s", order.order_id, broker_order_id)
        return broker_order_id

    def cancel_order(self, broker_order_id: str) -> None:
        if self.dry_run:
            LOG.info("[dry-run] cancel_order %s", broker_order_id)
            return
        self.broker.cancel_order(broker_order_id)

    def modify_stop(self, broker_order_id: str, new_stop: float) -> None:
        if self.dry_run:
            LOG.info("[dry-run] modify_stop %s -> %.2f", broker_order_id, new_stop)
            return
        self.broker.replace_order(broker_order_id, {"stop_price": new_stop})

    def close_position(self, *, symbol: str, qty: float) -> str:
        synthetic = OrderRecord(
            order_id=f"close-{symbol}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            trade_id="close",
            intent_id="system_close",
            symbol=symbol,
            side="SELL",
            quantity=qty,
            limit_price=None,
            stop_price=None,
            take_profit=None,
            idempotency_key=f"close:{symbol}",
        )
        return self.submit_order(synthetic)

    def close_all_positions(self, positions: Mapping[str, float]) -> list[str]:
        ids: list[str] = []
        for symbol, qty in positions.items():
            if qty <= 0:
                continue
            ids.append(self.close_position(symbol=symbol, qty=qty))
        return ids


def _order_payload(order: OrderRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "symbol": order.symbol,
        "side": order.side,
        "qty": order.quantity,
    }
    if order.limit_price is not None:
        payload["limit_price"] = float(order.limit_price)
    if order.stop_price is not None:
        payload["stop_price"] = float(order.stop_price)
    if order.take_profit is not None:
        payload["take_profit"] = float(order.take_profit)
    return payload
