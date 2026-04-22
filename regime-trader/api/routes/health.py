"""Health + freshness read routes (Phase B2 + B8)."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas import FreshnessPayload, HealthResponse
from api.services import PlatformService

router = APIRouter(tags=["status"])


@router.get("/health", response_model=HealthResponse)
def get_health(service: PlatformService = Depends(get_service)) -> HealthResponse:
    return HealthResponse(**service.get_health())


@router.get("/freshness", response_model=FreshnessPayload)
def get_freshness(service: PlatformService = Depends(get_service)) -> FreshnessPayload:
    return FreshnessPayload(**service.get_freshness())
