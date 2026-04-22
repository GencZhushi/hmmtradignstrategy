"""Approval workflow routes (Phase B3)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_service, require_role
from api.schemas import ApprovalDecisionRequest, ApprovalSchema
from api.services import PlatformService

router = APIRouter(tags=["approvals"])


@router.get("/approvals/pending", response_model=list[ApprovalSchema])
def pending_approvals(service: PlatformService = Depends(get_service)) -> list[ApprovalSchema]:
    return [ApprovalSchema(**a) for a in service.list_approvals()]


@router.get("/approvals/history", response_model=list[ApprovalSchema])
def approval_history(
    limit: int = Query(50, ge=1, le=500),
    service: PlatformService = Depends(get_service),
) -> list[ApprovalSchema]:
    return [ApprovalSchema(**a) for a in service.list_approvals(include_history=True)[:limit]]


@router.post("/approvals/approve")
def approve(
    payload: ApprovalDecisionRequest,
    service: PlatformService = Depends(get_service),
    principal=Depends(require_role("operator")),
) -> dict:
    try:
        return service.approve(
            payload.approval_id,
            actor=principal.subject,
            actor_type=principal.actor_type,
            reason=payload.reason,
            notes=payload.notes,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/approvals/reject")
def reject(
    payload: ApprovalDecisionRequest,
    service: PlatformService = Depends(get_service),
    principal=Depends(require_role("operator")),
) -> dict:
    try:
        return service.reject(
            payload.approval_id,
            actor=principal.subject,
            actor_type=principal.actor_type,
            reason=payload.reason,
            notes=payload.notes,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
