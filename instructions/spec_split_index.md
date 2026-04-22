# Trading System Spec Split Index

The original master markdown was mixing four concerns in one place:

- research/backtest logic
- live execution logic
- web/API platform requirements
- OpenClaw agent requirements

This split separates the system into three implementation specs:

## Spec A

**Core Trading Engine**

Covers:

- HMM regime detection
- strategy logic
- backtesting
- risk manager
- broker integration
- orchestration loop
- engine-side monitoring

File: `spec_a_core_trading_engine.md`

## Spec B

**Web / API Platform**

Covers:

- FastAPI layer
- persistence
- auth
- approval workflows
- web dashboard
- event streaming
- audit trails

File: `spec_b_web_api_platform.md`

## Spec C

**OpenClaw Agent Contract and Permissions**

Covers:

- tool/action contract
- JSON schemas
- permissions
- approval behavior
- audit requirements
- safe execution boundaries

File: `spec_c_openclaw_agent_contract.md`

## Dependency Order

Implementation order should be:

1. **Spec A first**
2. **Spec B second**
3. **Spec C third**

Correct dependency flow:

`Spec C → Spec B → Spec A`

or operationally:

`OpenClaw → API Platform → Trading Engine`


## Additional cross-cutting spec

- `architecture_and_sequence_diagrams.md` — final system architecture diagram, trade request sequence, and startup/recovery sequence

- `requirement_levels_and_delivery_priorities.md` — MUST / SHOULD / OPTIONAL / FUTURE labels across the split spec
