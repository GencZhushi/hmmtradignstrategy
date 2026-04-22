# Spec C: OpenClaw Agent Contract and Permissions

## Purpose

This specification defines how OpenClaw interacts with the trading system safely and predictably.

It covers:

- tool/action contract
- request/response schemas
- permissions and guardrails
- approval behavior
- agent-specific audit requirements

It explicitly excludes:

- engine logic
- web dashboard product requirements
- direct broker implementation details

Those belong to Spec A and Spec B.

---

## Agent Boundary

OpenClaw is a client of the platform. It is not the trading engine.

OpenClaw may:

- query state
- request previews
- submit structured trade intents
- request approval actions if policy allows
- explain why actions were rejected using system responses

OpenClaw may **not**:

- invent broker state
- bypass engine risk logic
- place raw orders outside the platform contract
- mutate settings outside its permission scope

---

## Command Philosophy

All agent actions must be structured.

Never allow:

- raw free-form order text to be sent to the broker
- code execution as a substitute for trading permissions
- direct Alpaca calls from the agent layer

All OpenClaw trade requests must follow:

`User/Agent Request → TradeIntent → Preview/Risk Check → Approval Policy → Execution`

---

## Required Tool Surface

Minimum tools/actions:

- `get_regime`
- `get_portfolio`
- `get_positions`
- `get_risk_status`
- `preview_trade`
- `submit_trade_intent`
- `approve_trade`
- `reject_trade`
- `close_position`
- `close_all_positions`
- `explain_rejection`
- `get_pending_approvals`

Each tool must return structured JSON.

---

## Canonical Payloads

## C1. `TradeIntent`

```json
{
  "intent_id": "uuid",
  "idempotency_key": "uuid-or-stable-hash",
  "source": "openclaw",
  "intent_type": "open_position",
  "symbol": "SPY",
  "direction": "LONG",
  "allocation_pct": 0.10,
  "requested_leverage": 1.0,
  "thesis": "Low-vol regime with confirmed trend",
  "timeframe": "5m",
  "requires_confirmation": true,
  "created_at": "2026-04-18T10:30:00Z",
  "status": "pending"
}
```

## C2. `OrderPlan`

```json
{
  "plan_id": "uuid",
  "intent_id": "uuid",
  "approved_signal": true,
  "risk_adjusted_size": 0.08,
  "risk_adjusted_leverage": 1.0,
  "entry_type": "limit",
  "limit_price": 520.10,
  "stop_loss": 508.00,
  "take_profit": null,
  "status": "previewed"
}
```

## C3. `AgentActionResult`

```json
{
  "ok": true,
  "action": "preview_trade",
  "resource_id": "plan_id",
  "status": "previewed",
  "message": "Trade preview created successfully",
  "requires_human_approval": true
}
```

---

## Permissions Model

### Default mode

- preview allowed
- live execution disabled
- paper execution optional, based on policy

### Permission tiers

- `agent_readonly`
- `agent_preview`
- `agent_paper_execute`
- `agent_live_execute`

### Live trading policy

Live execution requires all of the following:

- `live_enabled = true`
- session armed by authorized actor
- request passes Spec A risk validation
- request is below configured size/leverage limits
- request is fully auditable

Any circuit breaker or halt state disables new agent-driven execution.

---

## Approval Modes

### Mode A: Human approval required

`intent submitted → preview created → pending approval → approve/reject → execute or cancel`

### Mode B: Policy automation

`intent submitted → preview created → auto-execute if allowed by policy`

OpenClaw must know which mode is active and report it clearly.

---

## Required Rejection Behavior

If a request is rejected, the agent must return the structured reason from the platform.

Examples:

- breaker active
- leverage exceeded
- duplicate request
- stop missing
- symbol blocked
- live mode not armed
- approval required
- stale market data

The agent should explain the rejection, but only using the returned system reason.

---

## Idempotency and Safety

Every trade intent must have:

- unique `intent_id`
- audit trace
- replay-safe behavior

Repeated OpenClaw submissions of the same intent must not create duplicate live orders.

---

## Audit Requirements

Every agent action must record:

- actor = `openclaw`
- actor subtype / agent name if available
- original request summary
- structured payload
- policy result
- approval state
- final broker/system outcome
- timestamp

---

## Prompt / Adapter Rule

The OpenClaw system prompt and adapter should enforce:

- JSON-first responses for tool calls
- no assumptions about positions/orders unless fetched
- no direct broker talk outside allowed tools
- no silent execution in forbidden modes
- explain before execute when approval is required

---

## Integration Rule

Correct dependency direction:

`OpenClaw Agent → Web/API Platform (Spec B) → Core Trading Engine (Spec A)`

OpenClaw must never call the engine internals or broker directly.

---

## Explicit Exclusions

The following do **not** belong in Spec C:

- HMM feature definitions
- walk-forward backtest mechanics
- dashboard layout/product design
- broker SDK implementation

---

## Done Definition

Spec C is complete when:

- OpenClaw can query state reliably
- OpenClaw can preview trades safely
- permissions are enforced correctly
- approval rules are honored
- duplicate/unsafe execution paths are blocked
- every agent action is auditable
