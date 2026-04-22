"""User login route (Phase B1)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import AuthSettings, issue_access_token, verify_password
from api.dependencies import get_auth_settings, get_service
from api.schemas import LoginRequest, LoginResponse
from api.services import PlatformService

router = APIRouter(tags=["auth"], prefix="/auth")


@router.post("/login", response_model=LoginResponse)
def login(
    payload: LoginRequest,
    auth_settings: AuthSettings = Depends(get_auth_settings),
    service: PlatformService = Depends(get_service),
) -> LoginResponse:
    with service.repository.session() as session:
        from storage.models import UserRecord  # local import to avoid cycles
        from sqlalchemy import select

        record = session.scalar(select(UserRecord).where(UserRecord.username == payload.username))
    if record is None or not verify_password(payload.password, record.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid credentials")
    service.repository.update_user_login(payload.username)
    token = issue_access_token(auth_settings, payload.username, role=record.role)
    return LoginResponse(access_token=token, role=record.role, username=payload.username)
