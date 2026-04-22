"""Shared DI helpers for the FastAPI app (Phase B1)."""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from api.auth import AuthPrincipal, AuthSettings, resolve_principal
from api.services import PlatformService


def get_service(request: Request) -> PlatformService:
    service: PlatformService | None = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="platform service not initialized")
    return service


def get_auth_settings(request: Request) -> AuthSettings:
    settings: AuthSettings | None = getattr(request.app.state, "auth_settings", None)
    if settings is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="auth settings missing")
    return settings


def _optional_principal(
    auth_settings: AuthSettings = Depends(get_auth_settings),
    authorization: str | None = Header(default=None),
    x_service_token: str | None = Header(default=None),
    x_agent_token: str | None = Header(default=None),
) -> AuthPrincipal | None:
    try:
        return resolve_principal(
            auth_settings,
            authorization=authorization,
            service_header=x_service_token,
            agent_header=x_agent_token,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc


def require_principal(principal: AuthPrincipal | None = Depends(_optional_principal)) -> AuthPrincipal:
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")
    return principal


def require_role(role: str):
    def _dependency(principal: AuthPrincipal = Depends(require_principal)) -> AuthPrincipal:
        if not principal.has_role(role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"role '{role}' required; have '{principal.role}'",
            )
        return principal

    return _dependency


OptionalPrincipal = Annotated[AuthPrincipal | None, Depends(_optional_principal)]
RequiredPrincipal = Annotated[AuthPrincipal, Depends(require_principal)]


def idempotency_key_header(
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> str | None:
    """Extract optional ``Idempotency-Key`` header (Phase B6)."""
    return idempotency_key
