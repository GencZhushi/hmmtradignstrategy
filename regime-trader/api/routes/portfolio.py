"""Portfolio, positions, and risk status read routes (Phase B2 + B9)."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from api.dependencies import get_service
from api.schemas import ConcentrationSchema, PortfolioSchema, PositionSchema
from api.services import PlatformService

router = APIRouter(tags=["portfolio"])


@router.get("/portfolio", response_model=PortfolioSchema)
def get_portfolio(service: PlatformService = Depends(get_service)) -> PortfolioSchema:
    return PortfolioSchema(**service.get_portfolio())


@router.get("/positions", response_model=list[PositionSchema])
def get_positions(service: PlatformService = Depends(get_service)) -> list[PositionSchema]:
    portfolio = service.get_portfolio()
    return [PositionSchema(**p) for p in portfolio["positions"]]


@router.get("/risk/status")
def get_risk_status(service: PlatformService = Depends(get_service)) -> dict:
    return service.get_risk_status()


@router.get("/concentration", response_model=ConcentrationSchema)
def get_concentration(service: PlatformService = Depends(get_service)) -> ConcentrationSchema:
    return ConcentrationSchema(**service.get_concentration())
