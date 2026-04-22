"""Regime + market-freshness read routes (Phase B2 + B8 + B10)."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas import ModelGovernanceSchema, RegimeSnapshot
from api.services import PlatformService

router = APIRouter(tags=["regime"])


@router.get("/regime/current", response_model=RegimeSnapshot)
def get_current_regime(service: PlatformService = Depends(get_service)) -> RegimeSnapshot:
    return RegimeSnapshot(**service.get_regime())


@router.get("/regime/model", response_model=ModelGovernanceSchema)
def get_model_governance(service: PlatformService = Depends(get_service)) -> ModelGovernanceSchema:
    return ModelGovernanceSchema(**service.get_model_governance())
