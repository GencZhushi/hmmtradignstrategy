"""FastAPI application factory (Phase B1 + B5).

Wires together:

- config/secrets -> auth settings
- TradingApplication (Spec A engine) -> PlatformService (Spec B facade)
- SQLite-backed ``Repository`` -> storage for intents, orders, approvals, audit
- All API routes (health/portfolio/regime/market/signals/orders/approvals/audit/config/auth/streaming)
- OpenClaw agent router (Spec C) if ``integrations.openclaw`` is present
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from api.auth import AuthPrincipal, AuthSettings, build_auth_settings, hash_password
from api.routes import (
    approvals as approvals_routes,
    audit as audit_routes,
    auth as auth_routes,
    config as config_routes,
    health as health_routes,
    market as market_routes,
    orders as orders_routes,
    portfolio as portfolio_routes,
    regime as regime_routes,
    signals as signals_routes,
    streaming as streaming_routes,
)
from api.services import ApprovalPolicy, PlatformService
from config.loader import AppConfig
from monitoring.application import TradingApplication
from storage.repository import Repository, build_session_factory

LOG = logging.getLogger("regime_trader.api")


def _default_admin_user(service: PlatformService, cfg: AppConfig) -> None:
    """Bootstrap/sync the admin user so the .env password always works.

    On every startup we re-hash ``REGIME_TRADER_ADMIN_BOOTSTRAP_PASSWORD``
    and upsert it, so rotating the password in .env is enough (no DB
    surgery required). If the user already exists with a matching hash,
    bcrypt's salt makes the new hash differ but ``verify_password``
    still succeeds -- either way the stored secret ends up valid for
    the current .env value.
    """
    from sqlalchemy import select

    from storage.models import UserRecord

    password = cfg.secrets.admin_bootstrap_password or "regime-admin"
    password_hash = hash_password(password)
    with service.repository.session() as session:
        record = session.scalar(select(UserRecord).where(UserRecord.username == "admin"))
        if record is None:
            session.add(UserRecord(username="admin", password_hash=password_hash, role="admin"))
        else:
            record.password_hash = password_hash
            record.role = "admin"


def create_app(
    cfg: AppConfig,
    *,
    application: TradingApplication | None = None,
    dry_run: bool = False,
) -> FastAPI:
    """Build the FastAPI app.

    ``application`` overrides for tests (they pre-build their own
    ``TradingApplication`` with the desired ``dry_run`` setting). When not
    injected, we construct one respecting ``dry_run`` (default False so
    real Alpaca paper keys are honored when present; set True to force the
    SimulatedBroker regardless of credentials).
    """
    platform = cfg.section("platform")
    db_url = str(platform.get("sqlite_path", "state/regime_trader.db"))
    if not db_url.startswith("sqlite:") and not db_url.startswith("postgresql:"):
        path = Path(db_url)
        if not path.is_absolute() and cfg.source_path:
            path = Path(cfg.source_path).parent.parent / path
        db_url = f"sqlite:///{path}"
    session_factory = build_session_factory(db_url)
    repository = Repository(session_factory)

    application = application or TradingApplication(cfg, dry_run=dry_run)

    approval_policy = ApprovalPolicy(
        mode=str(platform.get("approval_mode", "manual")),
        require_approval_in_paper=bool(platform.get("require_approval_in_paper", True)),
    )
    service = PlatformService(
        application=application,
        repository=repository,
        approval_policy=approval_policy,
    )
    _default_admin_user(service, cfg)

    auth_settings = build_auth_settings({
        "jwt_secret": cfg.secrets.jwt_secret,
        "service_token": cfg.secrets.service_token,
        "openclaw_service_token": cfg.secrets.openclaw_service_token,
    })

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        try:
            application.shutdown()
        except Exception as exc:  # pragma: no cover - shutdown logging
            LOG.warning("Engine shutdown error: %s", exc)

    app = FastAPI(title="Regime Trader API", version="1.0.0", lifespan=_lifespan)
    app.state.service = service
    app.state.auth_settings = auth_settings
    app.state.application = application

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(auth_routes.router)
    app.include_router(health_routes.router)
    app.include_router(portfolio_routes.router)
    app.include_router(regime_routes.router)
    app.include_router(market_routes.router)
    app.include_router(signals_routes.router)
    app.include_router(orders_routes.router)
    app.include_router(approvals_routes.router)
    app.include_router(audit_routes.router)
    app.include_router(config_routes.router)
    app.include_router(streaming_routes.router)

    try:
        from integrations.openclaw.routes import router as openclaw_router

        app.include_router(openclaw_router)
    except ImportError as exc:  # pragma: no cover - integration optional
        LOG.warning("OpenClaw router not available: %s", exc)

    _register_dashboard(app, cfg)

    return app


def _register_dashboard(app: FastAPI, cfg: AppConfig) -> None:
    web_dir = Path(__file__).resolve().parent.parent / "web"
    index = web_dir / "index.html"

    if index.exists():
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def dashboard() -> HTMLResponse:
            return HTMLResponse(content=index.read_text(encoding="utf-8"))

        @app.get("/web/index.html", response_class=HTMLResponse, include_in_schema=False)
        def dashboard_web() -> HTMLResponse:
            return HTMLResponse(content=index.read_text(encoding="utf-8"))

    # Mount web static assets if folder exists.
    static_dir = web_dir / "static"
    if static_dir.exists():
        from starlette.staticfiles import StaticFiles

        app.mount("/web/static", StaticFiles(directory=str(static_dir)), name="web_static")

    @app.get("/info")
    def info() -> dict[str, Any]:
        return {
            "service": "regime-trader",
            "trading_mode": cfg.get("broker.trading_mode"),
            "execution_enabled": cfg.get("broker.execution_enabled"),
            "approval_mode": cfg.get("platform.approval_mode"),
        }


def run_server(
    cfg: AppConfig,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    dry_run: bool = False,
) -> int:  # pragma: no cover - CLI path
    import uvicorn

    app = create_app(cfg, dry_run=dry_run)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0
