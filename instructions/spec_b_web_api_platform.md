# Spec B: Web / API Platform

## Purpose

This specification defines the application platform around the trading engine:

- FastAPI service layer
- persistence and audit storage
- web dashboard
- authentication and authorization
- approval workflows
- event streaming

It explicitly excludes:

- HMM logic
- strategy logic
- backtest model selection logic
- broker-side sizing rules
- OpenClaw chat/tool contract details

Those belong to Spec A and Spec C.

---

## Platform Boundary

Spec B must treat the trading engine as an internal service.

The web/API platform may:

- query engine state
- request previews
- request execution through defined service methods
- display logs, approvals, and positions

The web/API platform may **not**:

- re-implement regime logic
- re-implement risk logic
- bypass engine-side validation

---

## Repository Scope

```text
regime-trader/
├── api/
│   ├── __init__.py
│   ├── app.py
│   ├── auth.py
│   ├── schemas.py
│   ├── dependencies.py
│   └── routes/
│       ├── health.py
│       ├── market.py
│       ├── portfolio.py
│       ├── regime.py
│       ├── signals.py
│       ├── orders.py
│       ├── approvals.py
│       ├── config.py
│       └── audit.py
├── storage/
│   ├── __init__.py
│   ├── models.py
│   ├── repository.py
│   └── migrations/
├── web/
│   └── frontend app
└── state/
    ├── approvals/
    ├── snapshots/
    └── audit/
```

---

## Core Responsibilities

## B1. API Layer

### Read endpoints

- `GET /health`
- `GET /regime/current`
- `GET /portfolio`
- `GET /positions`
- `GET /signals/latest`
- `GET /orders/history`
- `GET /risk/status`
- `GET /approvals/pending`
- `GET /audit/logs`


### Data freshness and session contract

The API/UI should expose, where relevant:

- `exchange_timezone`
- `exchange_session_state`
- `last_completed_daily_bar_time`
- `last_completed_intraday_bar_time`
- `data_freshness_status`
- `regime_effective_session_date`
- `stale_data_blocked` flag when execution is intentionally blocked

This keeps operators and OpenClaw from acting on incomplete or stale bars.

### Sector/correlation risk contract

Where relevant, the API/UI should also expose:

- sector bucket used for each instrument
- ETF handling category used for risk logic
- 60-day return-correlation metrics or summarized correlation-breach flags
- projected post-trade sector exposure
- projected post-trade correlated exposure
- explicit rejection/scaling reasons when a trade is reduced or blocked by concentration rules

---
### Action endpoints

- `POST /signals/preview`
- `POST /orders/preview`
- `POST /orders/execute`
- `POST /orders/cancel`
- `POST /positions/close`
- `POST /positions/close-all`
- `POST /approvals/approve`
- `POST /approvals/reject`
- `POST /config/reload`

### Rule

Every action endpoint must call Spec A service methods. No route may implement trading decisions directly.

---

## B2. Persistence

Persist at minimum:

- portfolio snapshots
- orders and fills
- pending approvals
- audit events
- breaker history
- last known regime state
- UI/user settings where needed

SQLite is acceptable initially. Postgres is preferred once the platform becomes multi-session or remotely hosted.

---

## B3. Authentication and Roles

### Auth modes

- web user login
- API token / service token for system integrations

### Roles

- `viewer`
- `operator`
- `admin`

### Admin-only actions

- close all positions
- reload config
- arm live mode
- change approval policy
- rotate tokens

---

## B4. Approval Workflow

### Manual mode

`Preview → pending approval → approve/reject → execute`

### Controlled automation mode

`Preview → policy check → execute automatically`

Spec B owns:

- approval storage
- approval UI
- approval audit trail
- approval status transitions

Spec A still owns the final risk validation before execution.

---

## B5. Web Dashboard

### Required pages

1. **Overview**
   - current regime
   - portfolio summary
   - risk status
   - P&L chart
   - active alerts

2. **Positions**
   - open positions
   - stop level
   - regime at entry
   - unrealized P&L
   - close/reduce actions

3. **Signals**
   - latest signals
   - confidence
   - rationale
   - preview/approve/reject actions

4. **Approvals**
   - pending approvals
   - approval history
   - actor, timestamp, reason

5. **Audit & Logs**
   - engine actions
   - user actions
   - API actions
   - execution outcomes

6. **Settings**
   - paper/live mode display
   - approval requirements
   - integration status
   - refresh preferences

---

## B6. Event Streaming

Expose live updates for the UI:

- `regime_changed`
- `signal_generated`
- `signal_rejected`
- `order_submitted`
- `order_filled`
- `position_updated`
- `circuit_breaker_triggered`
- `approval_pending`
- `approval_resolved`
- `system_alert`

WebSocket or Server-Sent Events are both acceptable.

---

## B7. Audit Model

Minimum audit record fields:

- event id
- actor
- actor type
- action
- resource type
- resource id
- before
- after
- reason
- timestamp

Every state-changing action from UI or API must be auditable.

---

## Integration Rule

Spec B may call Spec A, but Spec A must not depend on Spec B.

The correct dependency direction is:

`Web UI / API Platform → Core Trading Engine`

---

## Explicit Exclusions

The following do **not** belong in Spec B:

- HMM implementation details
- walk-forward backtest internals
- strategy selection logic
- broker-side sizing and stop formulas
- OpenClaw prompt/tool schema design

---

## Done Definition

Spec B is complete when:

- the API can expose engine state cleanly
- the web dashboard can monitor and control the system through API calls
- approval workflows work end to end
- all state-changing actions are audited
- no web/API route bypasses Spec A logic
