"""Signal + intent preview read routes (Phase B2 + B3)."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service, require_role
from api.schemas import OrderPlanSchema, SignalSchema, TradeIntentRequest
from api.services import PlatformService

router = APIRouter(tags=["signals"])


@router.get("/signals/latest", response_model=list[SignalSchema])
def latest_signals(service: PlatformService = Depends(get_service)) -> list[SignalSchema]:
    return [SignalSchema(**s) for s in service.latest_signals()]


@router.post("/signals/preview", response_model=OrderPlanSchema)
def preview_signal(
    payload: TradeIntentRequest,
    service: PlatformService = Depends(get_service),
    principal=Depends(require_role("operator")),
) -> OrderPlanSchema:
    outcome = service.preview_intent(
        payload.model_dump(),
        actor=principal.subject,
        actor_type=principal.actor_type,
    )
    return OrderPlanSchema(
        intent_id=outcome.plan.intent_id,
        plan_id=outcome.plan.plan_id,
        approved_signal=outcome.plan.approved_signal,
        symbol=outcome.plan.symbol,
        direction=outcome.plan.direction.value,
        risk_adjusted_size=outcome.plan.risk_adjusted_size,
        risk_adjusted_leverage=outcome.plan.risk_adjusted_leverage,
        entry_type=outcome.plan.entry_type,
        limit_price=outcome.plan.limit_price,
        stop_loss=outcome.plan.stop_loss,
        take_profit=outcome.plan.take_profit,
        status=outcome.plan.status,
        rejection_reason=outcome.plan.rejection_reason,
        reason_codes=list(outcome.plan.reason_codes),
        projected_exposure=outcome.plan.projected_exposure,
        projected_sector_exposure=outcome.plan.projected_sector_exposure,
        sector_bucket=outcome.plan.sector_bucket,
        etf_bucket=outcome.plan.etf_bucket,
    )
