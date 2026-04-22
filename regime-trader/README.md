# Regime Trader

HMM-driven, vol-regime allocation trading system with a FastAPI web/API platform
and an OpenClaw agent contract layer. Built to the three-part spec:

- `instructions/spec_a_core_trading_engine.md`
- `instructions/spec_b_web_api_platform.md`
- `instructions/spec_c_openclaw_agent_contract.md`

The implementation follows `requirement_levels_and_delivery_priorities.md`
(MUST > SHOULD > OPTIONAL > FUTURE) and the phase map in
`implementation_phases_acceptance_criteria.md`.

## Architecture at a glance

```
User / Operator ──► Web Dashboard ──► API Platform ──► Core Trading Engine
                                    ▲                │
                  OpenClaw Agent ───┘                ▼
                                                 Alpaca Broker
```

- **Core engine (Spec A):** HMM regime detection (daily bars), allocation layer,
  walk-forward backtester, risk manager, single-writer Alpaca executor,
  orchestration loop, and reconciliation.
- **Web/API platform (Spec B):** FastAPI service, SQLite-backed storage, auth
  (JWT + service tokens), approval workflows, audit trail, event streaming.
- **OpenClaw agent (Spec C):** structured tool adapter, policy enforcement,
  idempotent intent submission, read/preview/execute tools routed through the
  API platform only.

## Quick start

```bash
cd regime-trader
python -m venv .venv && .venv\Scripts\activate         # Windows
pip install -r requirements.txt
copy .env.example .env                                 # add Alpaca paper keys
pytest -q                                              # run MUST tests
python main.py --train-only                            # train HMM on bundled data
python main.py --backtest                              # walk-forward evaluation
python main.py --serve-api                             # launch the web/API platform
```

## Dependency order

Per the spec split index, operational dependency flows
`OpenClaw -> API Platform -> Trading Engine`. Build order is the mirror image:
Spec A first, then Spec B, then Spec C.

## CLI flags

| Flag | Purpose |
| --- | --- |
| `--settings PATH` | Override `config/settings.yaml` |
| `--env PATH` | Override `.env` file |
| `--trading-mode paper\|live` | Override broker mode |
| `--dry-run` | Full pipeline with no broker mutations |
| `--backtest` | Walk-forward backtester |
| `--train-only` | Train HMM and exit |
| `--serve-api` | Launch FastAPI platform |
| `--host`, `--port` | Bind address for API server |

## Tests

```bash
pytest -q                              # all tests
pytest -q tests/test_look_ahead.py     # critical no-look-ahead guard
pytest -q tests/test_hmm.py            # HMM training + persistence
pytest -q tests/test_risk.py           # risk breakers + veto logic
```

## Disclaimer

For research and paper trading. No guaranteed profits. Live trading requires
explicit arming, proper credentials, and ownership of the resulting risk.
