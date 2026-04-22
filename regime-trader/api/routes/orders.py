"""Orders + positions action routes (Phase B3 + B6 + B7)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_service, idempotency_key_header, require_role
from api.schemas import OrderPlanSchema, OrderSchema, TradeIntentRequest
from api.services import PlatformService

router = APIRouter(tags=["orders"])


@router.get("/orders/history", response_model=list[OrderSchema])
def order_history(
    limit: int = Query(100, ge=1, le=500),
    status: str | None = Query(default=None),
    service: PlatformService = Depends(get_service),
) -> list[OrderSchema]:
    orders = service.list_orders(limit=limit)
    if status:
        orders = [o for o in orders if o["status"] == status]
    return [OrderSchema(**o) for o in orders]


@router.post("/orders/preview", response_model=OrderPlanSchema)
def preview_order(
    payload: TradeIntentRequest,
    service: PlatformService = Depends(get_service),
    principal=Depends(require_role("operator")),
    idempotency: str | None = Depends(idempotency_key_header),
) -> OrderPlanSchema:
    data = payload.model_dump()
    if idempotency:
        data["idempotency_key"] = idempotency
    outcome = service.preview_intent(data, actor=principal.subject, actor_type=principal.actor_type)
    return _plan_to_schema(outcome)


@router.post("/orders/execute", response_model=OrderPlanSchema)
def execute_order(
    payload: TradeIntentRequest,
    service: PlatformService = Depends(get_service),
    principal=Depends(require_role("operator")),
    idempotency: str | None = Depends(idempotency_key_header),
) -> OrderPlanSchema:
    data = payload.model_dump()
    data["requires_confirmation"] = False
    if idempotency:
        data["idempotency_key"] = idempotency
    outcome = service.submit_intent(data, actor=principal.subject, actor_type=principal.actor_type)
    if not outcome.decision.approved:
        raise HTTPException(status_code=400, detail={
            "reason": outcome.decision.reason_message,
            "reason_codes": outcome.decision.reason_codes,
        })
    return _plan_to_schema(outcome)


@router.post("/orders/cancel")
def cancel_order(
    order_id: str,
    service: PlatformService = Depends(get_service),
    principal=Depends(require_role("operator")),
) -> dict:
    record = service.application.state_machine.orders.get(order_id)
    if record is None or not record.attempts:
        raise HTTPException(status_code=404, detail="order not found or not submitted")
    broker_order_id = record.attempts[-1].broker_order_id
    if broker_order_id is None:
        raise HTTPException(status_code=400, detail="order has no broker reference")
    service.application.executor.cancel_order(broker_order_id)
    service._record_audit(
        action="order_cancel",
        resource_type="order",
        resource_id=order_id,
        actor=principal.subject,
        actor_type=principal.actor_type,
        reason="user cancel",
        after={"broker_order_id": broker_order_id},
    )
    return {"cancelled": True, "order_id": order_id}


@router.post("/positions/close")
def close_position(
    symbol: str,
    service: PlatformService = Depends(get_service),
    principal=Depends(require_role("operator")),
) -> dict:
    return service.close_position(symbol=symbol, actor=principal.subject, actor_type=principal.actor_type)


@router.post("/positions/close-all")
def close_all_positions(
    service: PlatformService = Depends(get_service),
    principal=Depends(require_role("admin")),
) -> dict:
    closed = service.close_all_positions(actor=principal.subject, actor_type=principal.actor_type)
    return {"closed": closed}


def _plan_to_schema(outcome) -> OrderPlanSchema:
    plan = outcome.plan
    return OrderPlanSchema(
        intent_id=plan.intent_id,
        plan_id=plan.plan_id,
        approved_signal=plan.approved_signal,
        symbol=plan.symbol,
        direction=plan.direction.value,
        risk_adjusted_size=plan.risk_adjusted_size,
        risk_adjusted_leverage=plan.risk_adjusted_leverage,
        entry_type=plan.entry_type,
        limit_price=plan.limit_price,
        stop_loss=plan.stop_loss,
        take_profit=plan.take_profit,
        status=plan.status,
        rejection_reason=plan.rejection_reason,
        reason_codes=list(plan.reason_codes),
        projected_exposure=plan.projected_exposure,
        projected_sector_exposure=plan.projected_sector_exposure,
        sector_bucket=plan.sector_bucket,
        etf_bucket=plan.etf_bucket,
    )
