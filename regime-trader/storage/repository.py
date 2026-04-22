"""Repository layer: hides SQLAlchemy session plumbing from the API routes.

Exposes typed helpers for intents, orders, approvals, audit events, and the
portfolio snapshot history. SQLite is the default backend; callers may pass any
SQLAlchemy URL.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from core.types import (
    AuditEvent,
    IntentStatus,
    OrderPlan,
    TradeIntent,
)
from storage.models import (
    ApprovalRecord,
    AuditEventRecord,
    Base,
    BreakerEventRecord,
    ConfigRecord,
    IntentRecord,
    LiveArmingRecord,
    OrderRecord,
    PortfolioSnapshotRecord,
    UserRecord,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def build_session_factory(db_url: str, *, echo: bool = False) -> sessionmaker[Session]:
    """Create all tables and return a session factory for the given URL."""
    if db_url.startswith("sqlite:"):
        path = db_url.split("///", 1)[1] if "///" in db_url else ""
        if path and path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(db_url, echo=echo, future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Repository:
    """Typed wrapper around a ``sessionmaker``."""

    def __init__(self, session_factory: sessionmaker[Session]):
        self._session_factory = session_factory

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ---------------------------------------------------------- intents
    def upsert_intent(self, intent: TradeIntent, *, plan: OrderPlan | None = None, status: str | None = None) -> IntentRecord:
        try:
            return self._upsert_intent(intent, plan=plan, status=status)
        except IntegrityError:
            # Another thread or process inserted the same idempotency_key between
            # our SELECT and INSERT. Retry once; the second attempt will find the
            # existing row and perform an UPDATE instead.
            return self._upsert_intent(intent, plan=plan, status=status)

    def _upsert_intent(self, intent: TradeIntent, *, plan: OrderPlan | None = None, status: str | None = None) -> IntentRecord:
        with self.session() as session:
            existing = session.scalar(select(IntentRecord).where(IntentRecord.intent_id == intent.intent_id))
            if existing is None and intent.idempotency_key:
                # On a retry the coordinator reuses the idempotency key but the
                # service layer minted a fresh intent_id. Look up by the key to
                # update the original row instead of inserting a duplicate.
                existing = session.scalar(
                    select(IntentRecord).where(IntentRecord.idempotency_key == intent.idempotency_key)
                )
            now = _utcnow()
            if existing is None:
                record = IntentRecord(
                    intent_id=intent.intent_id,
                    idempotency_key=intent.idempotency_key or intent.intent_id,
                    actor=intent.actor or intent.source,
                    actor_type=intent.source,
                    symbol=intent.symbol,
                    direction=intent.direction.value,
                    allocation_pct=float(intent.allocation_pct),
                    requested_leverage=float(intent.requested_leverage),
                    intent_type=intent.intent_type,
                    thesis=intent.thesis,
                    requires_confirmation=intent.requires_confirmation,
                    status=(status or intent.status.value) if isinstance(intent.status, IntentStatus) else (status or str(intent.status)),
                    plan_id=plan.plan_id if plan else None,
                    payload=_safe_payload(intent, plan),
                    created_at=intent.created_at,
                    updated_at=now,
                )
                session.add(record)
                return record
            existing.status = status or existing.status
            existing.plan_id = plan.plan_id if plan else existing.plan_id
            existing.payload = _safe_payload(intent, plan)
            existing.updated_at = now
            return existing

    def get_intent(self, intent_id: str) -> IntentRecord | None:
        with self.session() as session:
            return session.scalar(select(IntentRecord).where(IntentRecord.intent_id == intent_id))

    def intent_by_idempotency(self, key: str) -> IntentRecord | None:
        with self.session() as session:
            return session.scalar(select(IntentRecord).where(IntentRecord.idempotency_key == key))

    def list_intents(self, *, limit: int = 50, status: str | None = None) -> list[IntentRecord]:
        with self.session() as session:
            stmt = select(IntentRecord)
            if status:
                stmt = stmt.where(IntentRecord.status == status)
            stmt = stmt.order_by(IntentRecord.created_at.desc()).limit(limit)
            return list(session.scalars(stmt))

    # ---------------------------------------------------------- orders
    def upsert_order(self, order_summary: Mapping[str, Any]) -> OrderRecord:
        with self.session() as session:
            order_id = str(order_summary["order_id"])
            record = session.scalar(select(OrderRecord).where(OrderRecord.order_id == order_id))
            now = _utcnow()
            if record is None:
                record = OrderRecord(
                    order_id=order_id,
                    trade_id=str(order_summary.get("trade_id", order_id)),
                    intent_id=str(order_summary.get("intent_id", "")),
                    symbol=str(order_summary.get("symbol", "")),
                    side=str(order_summary.get("side", "BUY")),
                    quantity=float(order_summary.get("quantity", 0.0)),
                    limit_price=_nullable_float(order_summary.get("limit_price")),
                    stop_price=_nullable_float(order_summary.get("stop_price")),
                    take_profit=_nullable_float(order_summary.get("take_profit")),
                    status=str(order_summary.get("status", "new")),
                    attempts=list(order_summary.get("attempts", [])),
                )
                session.add(record)
            record.status = str(order_summary.get("status", record.status))
            record.filled_qty = float(order_summary.get("filled_qty", record.filled_qty))
            record.avg_fill_price = _nullable_float(order_summary.get("avg_fill_price"))
            record.protective_stop_status = str(order_summary.get("protective_stop_status", record.protective_stop_status))
            record.rejection_reason = order_summary.get("rejection_reason")
            record.attempts = list(order_summary.get("attempts", record.attempts or []))
            record.updated_at = now
            return record

    def get_order(self, order_id: str) -> OrderRecord | None:
        with self.session() as session:
            return session.scalar(select(OrderRecord).where(OrderRecord.order_id == order_id))

    def list_orders(self, *, limit: int = 100, status: str | None = None) -> list[OrderRecord]:
        with self.session() as session:
            stmt = select(OrderRecord)
            if status:
                stmt = stmt.where(OrderRecord.status == status)
            stmt = stmt.order_by(OrderRecord.created_at.desc()).limit(limit)
            return list(session.scalars(stmt))

    # ---------------------------------------------------------- approvals
    def create_approval(
        self,
        *,
        intent_id: str,
        plan_id: str | None,
        requested_by: str,
        requested_by_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> ApprovalRecord:
        with self.session() as session:
            approval = ApprovalRecord(
                approval_id=f"approval-{uuid.uuid4().hex[:12]}",
                intent_id=intent_id,
                plan_id=plan_id,
                requested_by=requested_by,
                requested_by_type=requested_by_type,
                payload=dict(payload or {}),
            )
            session.add(approval)
            return approval

    def resolve_approval(
        self,
        approval_id: str,
        *,
        status: str,
        decided_by: str,
        decided_by_type: str,
        reason: str = "",
        notes: str = "",
    ) -> ApprovalRecord:
        with self.session() as session:
            approval = session.scalar(select(ApprovalRecord).where(ApprovalRecord.approval_id == approval_id))
            if approval is None:
                raise KeyError(approval_id)
            if approval.status != "pending":
                raise ValueError(f"Approval {approval_id} already resolved as {approval.status}")
            approval.status = status
            approval.decided_by = decided_by
            approval.decided_by_type = decided_by_type
            approval.decided_at = _utcnow()
            approval.reason = reason
            approval.notes = notes
            return approval

    def pending_approvals(self) -> list[ApprovalRecord]:
        with self.session() as session:
            stmt = select(ApprovalRecord).where(ApprovalRecord.status == "pending").order_by(ApprovalRecord.created_at.asc())
            return list(session.scalars(stmt))

    def list_approvals(self, *, limit: int = 50) -> list[ApprovalRecord]:
        with self.session() as session:
            stmt = select(ApprovalRecord).order_by(ApprovalRecord.created_at.desc()).limit(limit)
            return list(session.scalars(stmt))

    # ---------------------------------------------------------- audit
    def record_audit(self, event: AuditEvent) -> AuditEventRecord:
        with self.session() as session:
            record = AuditEventRecord(
                event_id=event.event_id,
                action=event.action,
                resource_type=event.resource_type,
                resource_id=event.resource_id,
                actor=event.actor,
                actor_type=event.actor_type,
                reason=event.reason,
                before=event.before,
                after=event.after,
                timestamp=event.timestamp,
            )
            session.add(record)
            return record

    def list_audit(self, *, limit: int = 100, resource_type: str | None = None, actor: str | None = None) -> list[AuditEventRecord]:
        with self.session() as session:
            stmt = select(AuditEventRecord)
            if resource_type:
                stmt = stmt.where(AuditEventRecord.resource_type == resource_type)
            if actor:
                stmt = stmt.where(AuditEventRecord.actor == actor)
            stmt = stmt.order_by(AuditEventRecord.timestamp.desc()).limit(limit)
            return list(session.scalars(stmt))

    # ---------------------------------------------------------- portfolio / breaker
    def save_portfolio_snapshot(self, payload: Mapping[str, Any]) -> PortfolioSnapshotRecord:
        with self.session() as session:
            record = PortfolioSnapshotRecord(
                equity=float(payload.get("equity", 0.0)),
                cash=float(payload.get("cash", 0.0)),
                buying_power=float(payload.get("buying_power", 0.0)),
                daily_pnl=float(payload.get("daily_pnl", 0.0)),
                weekly_pnl=float(payload.get("weekly_pnl", 0.0)),
                peak_equity=float(payload.get("peak_equity", 0.0)),
                drawdown=float(payload.get("drawdown", 0.0)),
                breaker_state=str(payload.get("breaker_state", "clear")),
                positions=dict(payload.get("positions", {})),
            )
            session.add(record)
            return record

    def latest_portfolio_snapshot(self) -> PortfolioSnapshotRecord | None:
        with self.session() as session:
            stmt = select(PortfolioSnapshotRecord).order_by(PortfolioSnapshotRecord.captured_at.desc()).limit(1)
            return session.scalar(stmt)

    def record_breaker_event(self, *, state: str, equity: float, daily_pnl: float, weekly_pnl: float, drawdown: float) -> BreakerEventRecord:
        with self.session() as session:
            record = BreakerEventRecord(
                state=state,
                equity=equity,
                daily_pnl=daily_pnl,
                weekly_pnl=weekly_pnl,
                drawdown=drawdown,
            )
            session.add(record)
            return record

    # ---------------------------------------------------------- users + arming
    def get_or_create_user(self, username: str, *, password_hash: str, role: str = "viewer") -> UserRecord:
        with self.session() as session:
            record = session.scalar(select(UserRecord).where(UserRecord.username == username))
            if record is None:
                record = UserRecord(username=username, password_hash=password_hash, role=role)
                session.add(record)
            return record

    def update_user_login(self, username: str) -> None:
        with self.session() as session:
            record = session.scalar(select(UserRecord).where(UserRecord.username == username))
            if record is not None:
                record.last_login = _utcnow()

    def arm_live(self, *, armed_by: str, ttl: timedelta, reason: str = "") -> LiveArmingRecord:
        with self.session() as session:
            record = LiveArmingRecord(
                armed_by=armed_by,
                expires_at=_utcnow() + ttl,
                reason=reason,
            )
            session.add(record)
            return record

    def active_arming(self) -> LiveArmingRecord | None:
        with self.session() as session:
            now = _utcnow()
            stmt = (
                select(LiveArmingRecord)
                .where(LiveArmingRecord.revoked_at.is_(None))
                .where(LiveArmingRecord.expires_at > now)
                .order_by(LiveArmingRecord.armed_at.desc())
                .limit(1)
            )
            return session.scalar(stmt)

    def revoke_arming(self, record_id: int) -> None:
        with self.session() as session:
            record = session.get(LiveArmingRecord, record_id)
            if record is not None:
                record.revoked_at = _utcnow()

    # ---------------------------------------------------------- config
    def set_config(self, key: str, value: str, *, updated_by: str = "system") -> ConfigRecord:
        with self.session() as session:
            record = session.scalar(select(ConfigRecord).where(ConfigRecord.key == key))
            if record is None:
                record = ConfigRecord(key=key, value=value, updated_by=updated_by)
                session.add(record)
            else:
                record.value = value
                record.updated_by = updated_by
                record.updated_at = _utcnow()
            return record

    def get_config(self, key: str) -> ConfigRecord | None:
        with self.session() as session:
            return session.scalar(select(ConfigRecord).where(ConfigRecord.key == key))


def _safe_payload(intent: TradeIntent, plan: OrderPlan | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "intent_type": intent.intent_type,
        "thesis": intent.thesis,
        "timeframe": intent.timeframe,
        "metadata": dict(intent.metadata),
    }
    if plan is not None:
        payload["plan"] = {
            "plan_id": plan.plan_id,
            "approved_signal": plan.approved_signal,
            "status": plan.status,
            "rejection_reason": plan.rejection_reason,
            "reason_codes": list(plan.reason_codes),
            "risk_adjusted_size": plan.risk_adjusted_size,
            "risk_adjusted_leverage": plan.risk_adjusted_leverage,
        }
    return payload


def _nullable_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
