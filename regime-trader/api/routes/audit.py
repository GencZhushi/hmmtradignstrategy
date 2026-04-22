"""Audit read routes (Phase B5)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_service
from api.schemas import AuditEventSchema
from api.services import PlatformService

router = APIRouter(tags=["audit"])


@router.get("/audit/logs", response_model=list[AuditEventSchema])
def get_audit_logs(
    limit: int = Query(100, ge=1, le=500),
    resource_type: str | None = Query(default=None),
    actor: str | None = Query(default=None),
    service: PlatformService = Depends(get_service),
) -> list[AuditEventSchema]:
    records = service.list_audit(limit=limit, resource_type=resource_type, actor=actor)
    return [AuditEventSchema(**r) for r in records]


@router.get("/audit/events")
def recent_events(
    limit: int = Query(50, ge=1, le=500),
    service: PlatformService = Depends(get_service),
) -> dict:
    return {"events": service.recent_events(limit=limit)}
