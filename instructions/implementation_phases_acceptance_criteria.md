# Implementation Phases and Acceptance Criteria

This document is the execution checklist that sits on top of the split specs:

- **Spec A** — Core Trading Engine
- **Spec B** — Web / API Platform
- **Spec C** — OpenClaw Agent Contract and Permissions

Use this document to keep Claude Code, Windsurf, or OpenClaw focused on what “done” means for each stage.

Recommended build order:

1. **A1 → A8**
2. **B1 → B5**
3. **C1 → C4**

---

# Spec A — Core Trading Engine

## Phase A1 — Repository and Config Skeleton

### Required files

- `main.py`
- `requirements.txt`
- `.env.example`
- `config/settings.yaml`
- `config/credentials.yaml.example`
- `core/__init__.py`
- `broker/__init__.py`
- `data/__init__.py`
- `backtest/__init__.py`
- `monitoring/__init__.py`

### Required classes/functions

- `load_settings()`
- `load_secrets()`
- `validate_config()`
- `bootstrap_project()`

### Test cases that must pass

- config file loads successfully
- missing required secret raises clear error
- runtime override takes precedence over `settings.yaml`
- `.env` is used only for secrets

### Expected outputs

- valid project tree
- validated config object printed/logged at startup
- startup fails fast on invalid config

### Definition of done

- project starts without import errors
- config precedence works exactly as specified
- no trading logic has been implemented yet beyond skeleton/bootstrap

---

## Phase A2 — Market Data and Feature Pipeline

### Required files

- `data/market_data.py`
- `data/feature_engineering.py`
- `tests/test_features.py`
- `tests/test_look_ahead.py`

### Required classes/functions

- `MarketDataManager`
- `FeatureEngine`
- `fetch_historical_daily_bars()`
- `fetch_intraday_bars()`
- `build_daily_features()`
- `build_execution_indicators()`

### Test cases that must pass

- minimum 504 completed daily bars can be fetched/loaded
- 5-minute bars can be fetched/loaded for execution simulation
- feature matrix contains expected columns
- no feature uses future data
- NaN warmup rows are handled deterministically

### Expected outputs

- daily feature DataFrame
- intraday execution indicator DataFrame
- clean train-ready feature set

### Definition of done

- daily HMM input features are reproducible
- intraday indicators are separated from HMM features
- look-ahead test passes

---

## Phase A3 — HMM Regime Engine

### Required files

- `core/hmm_engine.py`
- `tests/test_hmm.py`
- `models/` or equivalent persistence directory

### Required classes/functions

- `VolatilityRegimeHMM`
- `fit()`
- `select_model_bic()`
- `filtered_probabilities()`
- `assign_regime_labels()`
- `save_model()`
- `load_model()`

### Test cases that must pass

- BIC model selection works across candidate state counts
- best model is persisted
- model reload returns same metadata/state count
- filtered probabilities sum to 1 for each observation
- no direct use of future-dependent decoding in live/backtest path

### Expected outputs

- trained HMM object
- saved model artifact
- model metadata JSON/YAML
- filtered probability series
- labeled regime summary

### Definition of done

- HMM can train on 504 daily bars
- filtered probabilities are returned correctly
- model metadata is persisted
- regime labels are generated for readability

---

## Phase A4 — Strategy and Allocation Layer

### Required files

- `core/regime_strategies.py`
- `core/signal_generator.py`
- `tests/test_strategies.py`

### Required classes/functions

- `LowVolBullStrategy`
- `MidVolCautiousStrategy`
- `HighVolDefensiveStrategy`
- `StrategyOrchestrator`
- `generate_target_allocation()`
- `apply_uncertainty_mode()`

### Test cases that must pass

- low-vol regime maps to highest allocation
- high-vol regime maps to reduced allocation
- uncertainty mode reduces target size as configured
- rebalance threshold blocks tiny allocation changes
- strategy selection is deterministic for same regime input

### Expected outputs

- target allocation per symbol
- leverage decision
- signal rationale payload

### Definition of done

- regime-to-allocation mapping is deterministic
- uncertainty logic is applied consistently
- strategy outputs are usable by risk manager without extra transformation

---

## Phase A5 — Walk-Forward Backtester

### Required files

- `backtest/backtester.py`
- `backtest/performance.py`
- `backtest/stress_test.py`
- `tests/test_backtest.py`
- `tests/test_look_ahead.py`

### Required classes/functions

- `WalkForwardBacktester`
- `run_walk_forward()`
- `simulate_intraday_execution()`
- `compute_performance_metrics()`
- `compare_to_benchmarks()`

### Test cases that must pass

- train window = 504, test window = 126, step size = 126
- regime is updated only from completed daily bars
- intraday execution is simulated inside out-of-sample days
- equity curve is reproducible with same config
- no-look-ahead test passes end to end

### Expected outputs

- `equity_curve.csv`
- `trade_log.csv`
- `regime_history.csv`
- performance summary
- benchmark comparison output

### Definition of done

- full walk-forward run completes without leakage
- benchmark comparison is generated
- outputs can be consumed by later reporting/dashboard layers

---

## Phase A6 — Risk Manager

### Required files

- `core/risk_manager.py`
- `tests/test_risk.py`

### Required classes/functions

- `RiskManager`
- `validate_signal()`
- `enforce_drawdown_rules()`
- `check_position_limits()`
- `check_correlation_limits()`
- `compute_stop_levels()`

### Test cases that must pass

- orders exceeding max exposure are rejected
- orders exceeding single-position cap are rejected
- daily/weekly breaker transitions work correctly
- duplicate order requests are blocked
- missing stop logic causes rejection
- uncertainty mode sizing reaches risk manager correctly

### Expected outputs

- risk decision object with `approved / modified / rejected`
- stop-loss values
- rejection/modification reason codes

### Definition of done

- no order can reach broker layer without risk validation
- rejection reasons are explicit and structured
- breaker state is test-covered and auditable

---

## Phase A7 — Broker and Execution Layer

### Required files

- `broker/alpaca_client.py`
- `broker/order_executor.py`
- `broker/position_tracker.py`
- `tests/test_orders.py`
- `tests/test_broker_sync.py`

### Required classes/functions

- `AlpacaClient`
- `OrderExecutor`
- `PositionTracker`
- `submit_order()`
- `cancel_order()`
- `modify_stop()`
- `sync_positions()`
- `handle_fill_event()`

### Test cases that must pass

- broker health check succeeds/fails clearly
- paper credentials route correctly
- startup reconciliation restores positions
- partial fills update local state correctly
- retries do not create duplicate orders
- cancel/modify flows behave predictably

### Expected outputs

- submitted order records
- updated position state
- synchronized local vs broker portfolio view

### Definition of done

- engine can place paper orders safely
- restart reconciliation works
- broker failures are retried/logged without corrupting state

---

## Phase A8 — Main Loop, Monitoring, and Recovery

### Required files

- `main.py`
- `monitoring/logger.py`
- `monitoring/dashboard.py`
- `monitoring/alerts.py`
- `tests/test_recovery.py`

### Required classes/functions

- `TradingApplication` or equivalent orchestrator
- `run()`
- `shutdown()`
- `save_state_snapshot()`
- `load_state_snapshot()`
- `emit_alert()`

### Test cases that must pass

- application starts and enters loop cleanly
- daily regime is reused across 5-minute execution bars
- weekly retrain updates model without crashing loop
- state snapshot is saved on shutdown
- restart does not double-enter positions
- alerts/logs are written when breaker or feed issues occur

### Expected outputs

- live terminal dashboard
- log files
- state snapshot file
- session summary on shutdown

### Definition of done

- end-to-end paper trading loop works
- recovery is reliable
- monitoring shows enough state to debug live behavior

---

# Spec B — Web / API Platform

## Phase B1 — API Skeleton, Storage, and Auth

### Required files

- `api/app.py`
- `api/auth.py`
- `api/schemas.py`
- `api/dependencies.py`
- `storage/models.py`
- `storage/repository.py`
- `tests/test_api_boot.py`

### Required classes/functions

- `create_app()`
- `get_current_user()`
- `get_db_session()`
- `AuditEventSchema`
- `ApprovalSchema`

### Test cases that must pass

- API app boots successfully
- auth middleware protects private routes
- storage layer initializes correctly
- health route works

### Expected outputs

- running FastAPI app
- initialized database/storage
- basic auth flow

### Definition of done

- API skeleton is operational
- protected routes can be enforced
- persistence is available for later phases

---

## Phase B2 — Read-Only State API

### Required files

- `api/routes/health.py`
- `api/routes/portfolio.py`
- `api/routes/regime.py`
- `api/routes/market.py`
- `api/routes/signals.py`
- `tests/test_api_read_routes.py`

### Required classes/functions

- `get_health()`
- `get_portfolio()`
- `get_positions()`
- `get_current_regime()`
- `get_latest_signals()`

### Test cases that must pass

- all read routes return valid JSON schema
- API state matches engine state
- empty-state responses are handled cleanly
- route errors do not expose internal traces

### Expected outputs

- consumable JSON for dashboard and agent readers

### Definition of done

- web UI and OpenClaw can observe state without writing anything
- all routes are engine-backed, not mock-state driven

---

## Phase B3 — Preview, Execute, and Approval API

### Required files

- `api/routes/orders.py`
- `api/routes/approvals.py`
- `tests/test_api_actions.py`
- `tests/test_approval_flow.py`

### Required classes/functions

- `preview_order()`
- `execute_order()`
- `approve_request()`
- `reject_request()`
- `close_position()`
- `close_all_positions()`

### Test cases that must pass

- preview route returns structured order plan
- execution route calls Spec A validation path
- approval status transitions are valid
- rejected requests cannot execute
- duplicate approval/execution is blocked

### Expected outputs

- preview payloads
- approval records
- execution audit entries

### Definition of done

- all write actions flow through a controlled API
- approval workflow works end to end
- no route bypasses Spec A risk logic

---

## Phase B4 — Web Dashboard

### Required files

- frontend pages/components for Overview, Positions, Signals, Approvals, Audit, Settings
- `tests/test_frontend_smoke.*` or equivalent

### Required classes/functions

- `OverviewPage`
- `PositionsPage`
- `SignalsPage`
- `ApprovalsPage`
- `AuditPage`
- `SettingsPage`

### Test cases that must pass

- dashboard renders with live API data
- empty/loading/error states work
- approval actions update UI correctly
- position actions reflect API outcomes

### Expected outputs

- working browser dashboard
- operator-visible state and controls

### Definition of done

- a human operator can monitor and control the system from the browser
- UI actions are auditable and routed through the API

---

## Phase B5 — Event Streaming, Audit, and Settings Control

### Required files

- streaming layer
- `api/routes/audit.py`
- `api/routes/config.py`
- `tests/test_streaming.py`
- `tests/test_audit.py`

### Required classes/functions

- `stream_events()`
- `get_audit_logs()`
- `reload_config()`
- `arm_live_mode()`

### Test cases that must pass

- streaming pushes regime/order/approval events
- audit log captures all state-changing actions
- config reload works without invalidating engine state
- admin-only actions are permission-protected

### Expected outputs

- live UI updates
- searchable audit trail
- controlled settings actions

### Definition of done

- the platform is usable for real operations in paper mode
- operators can inspect history and manage runtime safely

---

# Spec C — OpenClaw Agent Contract and Permissions

## Phase C1 — Agent Adapter Skeleton and Schemas

### Required files

- `integrations/openclaw/tool_adapter.py`
- `integrations/openclaw/command_parser.py`
- `integrations/openclaw/policy.py`
- `integrations/openclaw/prompts.md`
- `tests/test_agent_schemas.py`

### Required classes/functions

- `TradeIntent`
- `OrderPlan`
- `AgentActionResult`
- `parse_agent_request()`
- `enforce_agent_policy()`

### Test cases that must pass

- agent payloads validate against schema
- unsupported actions are rejected clearly
- parser converts known commands into structured intents

### Expected outputs

- validated agent request/response models
- policy-ready intent objects

### Definition of done

- OpenClaw has a strict structured contract
- no free-form broker command path exists

---

## Phase C2 — Read-Only Agent Tools

### Required files

- read-only tool definitions or adapter handlers
- `tests/test_agent_read_tools.py`

### Required classes/functions

- `get_regime()`
- `get_portfolio()`
- `get_positions()`
- `get_risk_status()`
- `get_pending_approvals()`

### Test cases that must pass

- all read tools return structured JSON
- responses match API state
- agent does not hallucinate missing data when API is empty/unavailable

### Expected outputs

- safe read access for agent conversations

### Definition of done

- OpenClaw can observe the system safely and reliably

---

## Phase C3 — Preview and Intent Submission Tools

### Required files

- preview/intent tool handlers
- `tests/test_agent_preview.py`

### Required classes/functions

- `preview_trade()`
- `submit_trade_intent()`
- `explain_rejection()`

### Test cases that must pass

- preview requests create valid order plans
- rejected previews return system reason codes
- duplicate intent submission is idempotent

### Expected outputs

- preview payloads
- intent records
- structured rejection messages

### Definition of done

- OpenClaw can request trades without bypassing approval/risk flow

---

## Phase C4 — Approvals, Execution Permissions, and Audit

### Required files

- execution-capable tool handlers
- `tests/test_agent_permissions.py`
- `tests/test_agent_audit.py`

### Required classes/functions

- `approve_trade()`
- `reject_trade()`
- `close_position()`
- `close_all_positions()`
- `record_agent_audit_event()`

### Test cases that must pass

- agent cannot execute live trades when session is not armed
- agent cannot bypass approval mode
- every agent action creates audit data
- breaker state blocks new agent execution

### Expected outputs

- permission-enforced agent behavior
- agent audit trail
- safe execution path in paper mode

### Definition of done

- OpenClaw actions are safe, permissioned, auditable, and routed only through the platform

---

## Phase A9 — Concurrency, Locking, and Reconciliation

### Required files

- execution coordinator module or equivalent
- lock manager module or equivalent
- idempotency store/repository
- `tests/test_idempotency.py`
- `tests/test_reconciliation.py`
- `tests/test_concurrency.py`

### Required classes/functions

- `ExecutionCoordinator`
- `LockManager`
- `acquire_order_lock()`
- `release_order_lock()`
- `register_intent()`
- `check_idempotency()`
- `reconcile_after_fill()`
- `reconcile_after_reconnect()`

### Test cases that must pass

- duplicate UI/OpenClaw requests with same idempotency key do not create duplicate execution
- simultaneous conflicting requests are serialized or rejected deterministically
- partial fills reconcile local state correctly
- reconnect forces reconciliation before new writes
- retraining model swap does not corrupt live trading state

### Expected outputs

- lock-protected execution path
- deduplicated intent records
- reconciled portfolio/order state

### Definition of done

- exactly one writer path exists for broker mutations
- duplicate execution paths are prevented
- reconciliation is automatic after fills/reconnects

---

## Phase B6 — API Idempotency and Concurrency Safety

### Required files

- API middleware/dependency for idempotency handling
- `tests/test_api_idempotency.py`
- `tests/test_api_concurrency.py`

### Required classes/functions

- `require_idempotency_key()`
- `resolve_existing_intent()`
- `reject_conflicting_write()`

### Test cases that must pass

- repeated client retry with same idempotency key returns same logical result
- UI and OpenClaw concurrent writes deduplicate correctly
- read endpoints never mutate execution state

### Expected outputs

- API-safe retry behavior
- deterministic response for duplicate/conflicting writes

### Definition of done

- the platform is safe against duplicate client submissions and race-prone retries

---

## Phase C5 — Agent Idempotency and Safe Retry Behavior

### Required files

- agent retry/idempotency handling code
- `tests/test_agent_idempotency.py`

### Required classes/functions

- `build_idempotency_key()`
- `resume_pending_intent()`
- `handle_locked_or_pending_state()`

### Test cases that must pass

- agent retry reuses existing idempotency key
- agent does not create duplicate submissions during timeout/retry scenarios
- agent correctly reports pending/locked/reconciling states

### Expected outputs

- safe agent retry behavior
- correct attachment to pending intents

### Definition of done

- OpenClaw behaves safely under retries, timeouts, and concurrent UI activity

---

## Phase A10 — Partial Fills, Retries, and Order Lifecycle

### Required files

- order state machine module or equivalent
- protective-order manager module or equivalent
- `tests/test_partial_fills.py`
- `tests/test_order_lifecycle.py`
- `tests/test_stop_failures.py`
- `tests/test_bracket_desync.py`

### Required classes/functions

- `OrderStateMachine`
- `advance_order_state()`
- `handle_partial_fill()`
- `handle_stop_failure()`
- `handle_bracket_desync()`
- `mark_order_dead()`
- `update_trailing_stop_after_partial_exit()`

### Test cases that must pass

- partial fill updates live position state correctly
- retries reuse the same trade_id and intent_id
- stop-order failure triggers alert and policy handling
- bracket child desync triggers reconciliation and repair/cancel logic
- dead-order detection works predictably
- trailing stop updates correctly after partial exits

### Expected outputs

- explicit order status transitions
- auditable retry chain by trade_id/order_attempt_id
- protective-order repair/failure events

### Definition of done

- order behavior is fully specified for partial fills, retries, and protection failures
- lifecycle transitions are explicit and test-covered

---

## Phase B7 — API Order Status and Lifecycle Exposure

### Required files

- order/trade status schema updates
- `tests/test_api_order_status.py`

### Required classes/functions

- `serialize_order_status()`
- `serialize_trade_status()`
- `serialize_attempt_history()`

### Test cases that must pass

- API exposes lifecycle states consistently
- partial-fill status is distinguishable from filled/closed
- retry history is visible via trade_id and attempt data

### Expected outputs

- consistent UI/API status payloads
- traceable order history for operators and agents

### Definition of done

- the platform exposes a clear, auditable order lifecycle to all clients

---

## Phase C6 — Agent Interpretation of Partial Fills and Retries

### Required files

- agent status interpretation rules
- `tests/test_agent_order_status.py`

### Required classes/functions

- `interpret_order_state()`
- `decide_retry_or_wait()`
- `escalate_protection_failure()`

### Test cases that must pass

- agent does not treat partial fill as terminal
- agent does not create new trades when retrying same execution path
- agent escalates protection failures correctly

### Expected outputs

- correct agent behavior across lifecycle states
- safer retry/escalation behavior

### Definition of done

- OpenClaw behaves consistently with the engine's order lifecycle model

---

## Phase A11 — Data Realism and Backtest Realism

### Required files

- market data normalization/adjustment module
- exchange calendar/session module
- slippage model module
- `tests/test_adjustments.py`
- `tests/test_calendar_timezone.py`
- `tests/test_missing_bars.py`
- `tests/test_slippage_model.py`
- `tests/test_overnight_gap_logic.py`

### Required classes/functions

- `normalize_price_series()`
- `apply_adjustment_policy()`
- `get_exchange_session_state()`
- `is_bar_complete()`
- `is_data_stale()`
- `estimate_slippage()`
- `simulate_gap_fill_behavior()`
- `get_effective_intraday_regime()`

### Test cases that must pass

- adjusted daily data is handled consistently
- survivorship-bias policy is explicit and test-checked where applicable
- exchange calendar/session logic handles holidays and timezone correctly
- incomplete/stale intraday bars are blocked from decision use
- slippage scales with volatility/spread assumptions
- overnight stop/exit behavior is simulated consistently with live policy
- daily regime stays fixed intraday and updates only from completed daily bars

### Expected outputs

- realistic, normalized market data pipeline
- session-aware backtest/live timing behavior
- more defensible execution-cost simulation

### Definition of done

- data realism assumptions are explicit, coded, and test-covered
- daily HMM vs 5-minute execution timing is unambiguous

---

## Phase B8 — API Exposure of Freshness, Session, and Regime Timing

### Required files

- API schema updates for freshness/session metadata
- `tests/test_api_freshness.py`

### Required classes/functions

- `serialize_data_freshness()`
- `serialize_session_state()`
- `serialize_regime_effective_time()`

### Test cases that must pass

- API exposes last completed bar times correctly
- API shows session state/timezone consistently
- stale/incomplete data conditions are visible to UI and agents

### Expected outputs

- UI/API visibility into session timing and data freshness

### Definition of done

- operators and agents can see whether data is current, stale, or blocked for execution

---

## Phase C7 — Agent Behavior Under Stale Data and Session Constraints

### Required files

- agent freshness/session handling rules
- `tests/test_agent_freshness_behavior.py`

### Required classes/functions

- `interpret_freshness_status()`
- `decide_wait_vs_act()`
- `respect_regime_effective_session()`

### Test cases that must pass

- agent does not act on incomplete/stale intraday bars when policy blocks execution
- agent respects fixed intraday regime timing
- agent interprets session-edge warnings correctly

### Expected outputs

- safer agent behavior around stale data, session edges, and daily/intraday timing mismatch

### Definition of done

- OpenClaw behaves consistently with the platform's freshness and regime-timing rules

---

## Phase A12 — Sector Classification and Correlation-Constrained Exposure

### Required files

- instrument metadata / sector mapping module
- correlation risk module
- `tests/test_sector_mapping.py`
- `tests/test_etf_treatment.py`
- `tests/test_correlation_constraints.py`
- `tests/test_joint_trade_breach_resolution.py`

### Required classes/functions

- `get_sector_bucket()`
- `get_etf_risk_bucket()`
- `compute_rolling_return_correlation()`
- `project_post_trade_exposure()`
- `check_sector_limit()`
- `check_correlation_limit()`
- `resolve_joint_breach()`

### Test cases that must pass

- sector classification source is deterministic and auditable
- ETF handling policy is explicit and test-covered
- 60-day rolling correlation is calculated from returns
- sector/correlation rules are evaluated on projected post-rebalance exposure
- multiple approved trades that jointly breach limits are scaled or rejected deterministically

### Expected outputs

- explicit sector buckets and ETF treatment
- portfolio-level concentration checks
- deterministic scaling/rejection decisions under joint breaches

### Definition of done

- sector and correlation logic is explicit, reproducible, and enforced pre-trade at portfolio level

---

## Phase B9 — API Exposure of Concentration and Correlation Checks

### Required files

- API schema updates for concentration risk payloads
- `tests/test_api_concentration_payloads.py`

### Required classes/functions

- `serialize_sector_exposure()`
- `serialize_correlation_risk()`
- `serialize_concentration_rejection_reason()`

### Test cases that must pass

- API exposes sector bucket and ETF handling metadata correctly
- API exposes projected post-trade concentration checks consistently
- rejection/scaling reasons are visible to operators and agents

### Expected outputs

- UI/API visibility into sector and correlation risk decisions

### Definition of done

- concentration risk decisions are transparent to users, operators, and OpenClaw

---

## Phase C8 — Agent Behavior Under Portfolio Concentration Constraints

### Required files

- agent concentration-risk interpretation rules
- `tests/test_agent_concentration_behavior.py`

### Required classes/functions

- `interpret_concentration_rejection()`
- `handle_scaled_trade_decision()`
- `respect_joint_breach_resolution()`

### Test cases that must pass

- agent does not resubmit blocked trades without changed exposure context
- agent understands scaled-versus-rejected outcomes
- agent respects portfolio-level concentration limits even when a single signal looks valid in isolation

### Expected outputs

- safer agent behavior under sector/correlation constraints

### Definition of done

- OpenClaw behaves consistently with portfolio-level concentration and diversification rules

---

## Phase A13 — Model Governance, Promotion, Rollback, and Fallback

### Required files

- model registry / governance module
- promotion policy module
- `tests/test_model_versioning.py`
- `tests/test_model_promotion.py`
- `tests/test_model_rollback.py`
- `tests/test_retraining_fallback.py`

### Required classes/functions

- `register_model_version()`
- `store_training_metadata()`
- `compute_training_dataset_hash()`
- `compare_candidate_vs_active_model()`
- `promote_model()`
- `rollback_model()`
- `get_fallback_model()`

### Test cases that must pass

- each training run creates a unique, traceable model version record
- training metadata is stored and queryable
- dataset hash is reproducible for the same training basis
- candidate model is compared against active model before promotion
- rollback restores the prior approved model version
- failed retraining leaves the system on active or fallback model without breaking execution

### Expected outputs

- traceable model registry
- explicit promotion/rollback decisions
- safe fallback behavior under retraining failure

### Definition of done

- model lifecycle is governed, auditable, and safe for operational use

---

## Phase B10 — API Exposure of Model Governance State

### Required files

- API schema updates for model governance payloads
- `tests/test_api_model_governance.py`

### Required classes/functions

- `serialize_active_model_status()`
- `serialize_training_metadata()`
- `serialize_promotion_decision()`
- `serialize_fallback_model_status()`

### Test cases that must pass

- API exposes active model version correctly
- API exposes latest training/promotion metadata consistently
- API shows rollback/fallback state where applicable

### Expected outputs

- UI/API visibility into which model is live and why

### Definition of done

- operators and OpenClaw can inspect model governance state without ambiguity

---

## Phase C9 — Agent Behavior Under Model Promotion, Rollback, and Fallback

### Required files

- agent model-status interpretation rules
- `tests/test_agent_model_governance_behavior.py`

### Required classes/functions

- `interpret_active_model_status()`
- `handle_model_rollback_event()`
- `respect_unpromoted_candidate_model()`

### Test cases that must pass

- agent does not assume the newest trained model is active
- agent correctly interprets rollback/fallback states
- agent uses platform-reported active model identity for operational reasoning

### Expected outputs

- safer agent behavior around retraining, promotion, and rollback events

### Definition of done

- OpenClaw behaves consistently with the platform's model-governance lifecycle

---

## Phase B12 — Final Architecture and Sequence Diagrams

### Required files

- `architecture_and_sequence_diagrams.md`
- embedded Mermaid diagrams or equivalent renderable diagrams

### Required artifacts

- system architecture diagram
- trade request sequence diagram
- startup/recovery sequence diagram

### Test cases that must pass

- diagrams match the single-writer execution model
- diagrams match approval-before-execution workflow
- diagrams match startup reconciliation and recovery rules
- diagrams are consistent with model governance, audit logging, and dashboard read-model behavior

### Expected outputs

- one canonical architecture diagram
- one canonical trade request sequence
- one canonical startup/recovery sequence

### Definition of done

- the spec includes visual diagrams that align all builders on the same component map and operational flows

---

## Phase X1 — Requirement-Level Labeling and Delivery Prioritization

### Required files

- `requirement_levels_and_delivery_priorities.md`

### Required artifacts

- priority legend for MUST / SHOULD / OPTIONAL / FUTURE
- labeled delivery priorities across core engine, web/API, agent, and governance areas

### Test cases that must pass

- essential safety/correctness features are labeled MUST
- production-hardening items are distinguished from optional enhancements
- future roadmap items are separated from first-build scope

### Expected outputs

- clearer build prioritization
- reduced scope drift
- more practical first implementation planning

### Definition of done

- the spec clearly distinguishes mandatory build items from optional or future enhancements

---

# Final Rule for AI Builders

When giving this to Claude Code, Windsurf, or OpenClaw:

- give the **spec** plus the **relevant phase section**
- tell it to complete only the current phase
- require all listed tests to pass before moving on
- do not allow it to jump ahead to later layers

That will keep the build much more controlled.
