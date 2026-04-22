"""FastAPI router that exposes the OpenClaw tool surface (Spec C)."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from api.dependencies import get_service, require_role
from api.services import PlatformService
from core.types import AgentActionResult
from integrations.openclaw.command_parser import parse_agent_request
from integrations.openclaw.policy import AgentPolicy, PermissionTier
from integrations.openclaw.tool_adapter import OpenClawAdapter, TOOL_SPECS

router = APIRouter(tags=["agent"], prefix="/agent")


class ToolInvocation(BaseModel):
    tool: str
    params: dict[str, Any] = Field(default_factory=dict)


class NaturalLanguageRequest(BaseModel):
    text: str
    source: str = "openclaw"


def _adapter(service: PlatformService, role: str) -> OpenClawAdapter:
    tier = _role_to_tier(role)
    policy = AgentPolicy(tier=tier, allow_paper_auto_execute=False)
    return OpenClawAdapter(service=service, policy=policy)


def _role_to_tier(role: str) -> PermissionTier:
    return {
        "viewer": PermissionTier.READONLY,
        "operator": PermissionTier.PREVIEW,
        "admin": PermissionTier.PAPER_EXECUTE,
    }.get(role, PermissionTier.PREVIEW)


def _result_to_payload(result: AgentActionResult) -> dict[str, Any]:
    return asdict(result)


@router.get("/tools")
def list_tools(service: PlatformService = Depends(get_service)) -> dict[str, Any]:
    return {"tools": [asdict(spec) for spec in TOOL_SPECS]}


@router.post("/tools/invoke")
def invoke_tool(
    payload: ToolInvocation,
    service: PlatformService = Depends(get_service),
    principal=Depends(require_role("viewer")),
) -> dict[str, Any]:
    adapter = _adapter(service, principal.role)
    result = adapter.invoke(payload.tool, payload.params)
    return _result_to_payload(result)


@router.post("/command")
def command(
    payload: NaturalLanguageRequest,
    service: PlatformService = Depends(get_service),
    principal=Depends(require_role("viewer")),
) -> dict[str, Any]:
    parsed = parse_agent_request(payload.text)
    if parsed.tool == "parse_error":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"reason": parsed.params.get("reason"), "raw_text": payload.text})
    adapter = _adapter(service, principal.role)
    result = adapter.invoke(parsed.tool, parsed.params)
    return {
        "parsed": {
            "tool": parsed.tool,
            "params": parsed.params,
            "confidence": parsed.confidence,
            "warnings": parsed.warnings,
        },
        "result": _result_to_payload(result),
    }


@router.get("/pending")
def pending(service: PlatformService = Depends(get_service)) -> dict[str, Any]:
    return {"approvals": service.list_approvals()}
