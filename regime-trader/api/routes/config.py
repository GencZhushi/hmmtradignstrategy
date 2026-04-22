"""Admin routes: config reload + arm-live + model governance (Phase B5 + B10)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_service, require_role
from api.schemas import ArmLiveRequest, ConfigReloadRequest, ModelGovernanceSchema
from api.services import PlatformService

router = APIRouter(tags=["admin"], prefix="/config")


@router.post("/reload")
def reload_config(
    payload: ConfigReloadRequest,
    service: PlatformService = Depends(get_service),
    principal=Depends(require_role("admin")),
) -> dict:
    return service.reload_config(actor=principal.subject, reason=payload.reason)


@router.post("/arm-live")
def arm_live(
    payload: ArmLiveRequest,
    service: PlatformService = Depends(get_service),
    principal=Depends(require_role("admin")),
) -> dict:
    return service.arm_live(actor=principal.subject, ttl_minutes=payload.ttl_minutes, reason=payload.reason)


@router.get("/model", response_model=ModelGovernanceSchema)
def model_governance(service: PlatformService = Depends(get_service)) -> ModelGovernanceSchema:
    return ModelGovernanceSchema(**service.get_model_governance())


@router.post("/model/promote")
def promote_model(
    version: str,
    service: PlatformService = Depends(get_service),
    principal=Depends(require_role("admin")),
) -> dict:
    outcome = service.promote_model(version=version, actor=principal.subject)
    if not outcome["approved"]:
        raise HTTPException(status_code=409, detail=outcome)
    return outcome


@router.post("/model/rollback")
def rollback_model(
    service: PlatformService = Depends(get_service),
    principal=Depends(require_role("admin")),
) -> dict:
    return service.rollback_model(actor=principal.subject)
