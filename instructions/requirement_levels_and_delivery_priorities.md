# Requirement Levels and Delivery Priorities

This document adds practical delivery labels to the split specification so builders can separate what is essential from what is nice to have.

## Priority labels

- **MUST** = required for correctness, safety, or core operability
- **SHOULD** = strongly recommended for a production-ready build, but the system can still run without it in an earlier iteration
- **OPTIONAL** = useful enhancement, but not required for initial usable delivery
- **FUTURE** = deliberately deferred; include in roadmap, not in the first practical implementation

---

## 1. Core trading engine priorities

### MUST

- HMM training with explicit no-look-ahead handling
- no-look-ahead test for regime inference
- model metadata persistence
- single-writer order execution path
- idempotency key support for execution-capable actions
- explicit order lifecycle state machine
- partial fill handling
- startup reconciliation and restart safety
- projected post-trade risk checks before execution
- daily-HMM / intraday-execution timing rule
- split/dividend-adjusted historical data policy
- exchange calendar and timezone handling
- model versioning and rollback capability
- fallback to active/approved model if retraining fails

### SHOULD

- volatility/spread-based slippage model
- sector/correlation portfolio constraints
- bracket desync repair logic
- protective-order failure escalation policy
- candidate-vs-active model comparison before promotion
- dataset hash for training runs

### OPTIONAL

- advanced execution tactics beyond basic target-allocation movement
- advanced look-through ETF sector decomposition
- extended strategy comparison dashboards

### FUTURE

- multi-broker abstraction layer
- cross-venue smart routing
- highly advanced market microstructure simulation

---

## 2. Web / API platform priorities

### MUST

- workflow-oriented UI for core actions
- monitor regime workflow
- preview trade workflow
- approve trade workflow
- inspect rejection workflow
- close position workflow
- arm live mode workflow
- audit trail workflow
- recovery-after-restart workflow
- API exposure of intent/order/position status
- API exposure of model governance state
- API exposure of data freshness/session status

### SHOULD

- webhooks for operational state changes
- realtime dashboard refresh
- clear rejection/scaling explanations in UI
- operator notes on approvals/rejections

### OPTIONAL

- email alerts
- SMS or chat notifications
- advanced custom dashboard widgets
- downloadable analytics reports

### FUTURE

- full Postgres migration if early versions start simpler
- multi-tenant admin console
- mobile-native companion app

---

## 3. OpenClaw agent priorities

### MUST

- strict use of API contract
- no direct broker access outside approved execution path
- respect for approval gates
- respect for idempotency and retry rules
- respect for stale-data/session blocks
- respect for model governance state
- respect for concentration/risk rejections

### SHOULD

- human-readable explanations for blocked actions
- ability to summarize audit trail events
- safe retry behavior with escalation logic

### OPTIONAL

- conversational convenience features
- proactive trade idea summaries
- natural-language workflow shortcuts

### FUTURE

- multi-agent specialization layers for research, execution, and oversight
- autonomous scheduling of retraining reviews and promotion suggestions

---

## 4. Persistence, audit, and governance priorities

### MUST

- audit log for intents, approvals, order events, fills, reconciliation, and model changes
- persistent storage for model metadata
- persistent storage for active configuration and execution state needed for recovery

### SHOULD

- structured event store with replay-friendly records
- dataset fingerprint storage
- promotion-decision comparison reports

### OPTIONAL

- rich historical operator activity dashboards
- exportable governance reports

### FUTURE

- full compliance-style reporting suite
- archival cold-storage governance tooling

---

## 5. Practical examples

- **HMM no-look-ahead test = MUST**
- **webhooks = SHOULD**
- **email alerts = OPTIONAL**
- **Postgres migration = FUTURE**

---

## 6. Delivery rule for builders

If time or implementation bandwidth is limited:

1. build all **MUST** items first
2. add **SHOULD** items only after the MUST baseline is stable
3. treat **OPTIONAL** items as polish
4. keep **FUTURE** items out of the first delivery unless explicitly requested

This priority document is intended to keep Windsurf, Claude Code, OpenClaw, and human builders focused on a practical first implementation rather than an overbuilt first draft.
