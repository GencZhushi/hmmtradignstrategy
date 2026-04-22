"""Authentication + role-based authorization (Phase B1).

Supports two parallel modes:

- **User auth:** username/password logins that return a short-lived JWT.
- **Service auth:** static service tokens for OpenClaw and system integrations.

Roles: ``viewer`` < ``operator`` < ``admin``. Admin-only actions (close-all,
reload config, arm live mode, rotate tokens, change approval policy) reuse the
``require_role`` dependency.
"""
from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

try:
    from jose import JWTError, jwt  # type: ignore
except ImportError:  # pragma: no cover - dependency missing
    jwt = None  # type: ignore[assignment]
    JWTError = Exception  # type: ignore[misc,assignment]

try:
    import bcrypt  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    bcrypt = None  # type: ignore[assignment]

LOG = logging.getLogger(__name__)

ROLES_ORDER = ("viewer", "operator", "admin")


@dataclass
class AuthPrincipal:
    """Resolved caller identity."""

    subject: str
    role: str = "viewer"
    actor_type: str = "user"  # user | service | agent
    token_type: str = "jwt"   # jwt | service_token

    def has_role(self, required: str) -> bool:
        try:
            return ROLES_ORDER.index(self.role) >= ROLES_ORDER.index(required)
        except ValueError:
            return False


@dataclass
class AuthSettings:
    jwt_secret: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    jwt_algorithm: str = "HS256"
    jwt_ttl_minutes: int = 60
    service_tokens: dict[str, str] = field(default_factory=dict)  # token -> role
    agent_tokens: dict[str, str] = field(default_factory=dict)    # token -> role


def _truncate(password: str) -> bytes:
    # bcrypt limits the input to 72 bytes; longer secrets must be truncated.
    return password.encode("utf-8")[:72]


def hash_password(password: str) -> str:
    if bcrypt is None:  # pragma: no cover - dev fallback
        return f"plain:{password}"
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(_truncate(password), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    if bcrypt is None:  # pragma: no cover
        return password_hash == f"plain:{password}"
    if password_hash.startswith("plain:"):
        return password_hash == f"plain:{password}"
    try:
        return bcrypt.checkpw(_truncate(password), password_hash.encode("utf-8"))
    except ValueError:
        return False


def issue_access_token(settings: AuthSettings, subject: str, *, role: str) -> str:
    if jwt is None:
        raise RuntimeError("python-jose is required for JWT support")
    payload = {
        "sub": subject,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_ttl_minutes),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(settings: AuthSettings, token: str) -> AuthPrincipal:
    if jwt is None:
        raise RuntimeError("python-jose is required for JWT support")
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:  # pragma: no cover - auth failures
        raise PermissionError(f"Invalid token: {exc}") from exc
    return AuthPrincipal(
        subject=str(payload.get("sub", "")),
        role=str(payload.get("role", "viewer")),
        actor_type="user",
        token_type="jwt",
    )


def resolve_principal(settings: AuthSettings, *, authorization: str | None, service_header: str | None, agent_header: str | None) -> AuthPrincipal | None:
    """Resolve the principal for the current request, if any."""
    if service_header and service_header in settings.service_tokens:
        role = settings.service_tokens[service_header]
        return AuthPrincipal(subject="service", role=role, actor_type="service", token_type="service_token")
    if agent_header and agent_header in settings.agent_tokens:
        role = settings.agent_tokens[agent_header]
        return AuthPrincipal(subject="openclaw", role=role, actor_type="agent", token_type="service_token")
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1]
        return decode_access_token(settings, token)
    return None


def build_auth_settings(secrets_map: Mapping[str, Any]) -> AuthSettings:
    """Produce an ``AuthSettings`` from the loaded ``Secrets`` dataclass."""
    jwt_secret = str(secrets_map.get("jwt_secret") or secrets.token_urlsafe(32))
    settings = AuthSettings(jwt_secret=jwt_secret)
    if service_token := secrets_map.get("service_token"):
        settings.service_tokens[str(service_token)] = "admin"
    if agent_token := secrets_map.get("openclaw_service_token"):
        settings.agent_tokens[str(agent_token)] = "operator"
    return settings
