# Spec A: Core Trading Engine

## Purpose

This specification defines the standalone trading engine. It covers market data ingestion, HMM-based regime detection, allocation logic, backtesting, risk management, broker execution, and the orchestration loop.

It explicitly excludes:

- web UI concerns
- public/internal API platform concerns
- OpenClaw agent chat/tool behavior
- approval workflows outside the engine's internal execution policy

---

## System Boundary

The core engine is the single source of truth for:

- regime detection
- portfolio sizing
- risk enforcement
- order generation
- portfolio state
- broker synchronization
- audit-grade execution state

Everything else must call into the engine rather than re-implementing trading logic.

---

## Architecture Standard

### Timeframe standard

Use **Daily regime + intraday execution**.

- HMM training and filtered inference use **completed daily bars only**
- Execution, rebalancing, stop updates, and live orchestration use **5-minute bars**
- The daily regime remains fixed intraday unless a new completed daily bar is processed

### Training standard

- `min_train_bars = 504`
- walk-forward `train_window = 504`
- walk-forward `test_window = 126`
- walk-forward `step_size = 126`

### Configuration precedence

1. `.env` = secrets only
2. `settings.yaml` = runtime behavior/configuration
3. CLI flags override `settings.yaml` for the current process

---

## Repository Scope

```text
regime-trader/
├── config/
│   ├── settings.yaml
│   └── credentials.yaml.example
├── core/
│   ├── __init__.py
│   ├── hmm_engine.py
│   ├── regime_strategies.py
│   ├── risk_manager.py
│   └── signal_generator.py
├── broker/
│   ├── __init__.py
│   ├── alpaca_client.py
│   ├── order_executor.py
│   └── position_tracker.py
├── data/
│   ├── __init__.py
│   ├── market_data.py
│   └── feature_engineering.py
├── monitoring/
│   ├── __init__.py
│   ├── logger.py
│   ├── dashboard.py
│   └── alerts.py
├── backtest/
│   ├── __init__.py
│   ├── backtester.py
│   ├── performance.py
│   └── stress_test.py
├── tests/
│   ├── test_hmm.py
│   ├── test_look_ahead.py
│   ├── test_strategies.py
│   ├── test_risk.py
│   └── test_orders.py
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Config Model

### `settings.yaml`

```yaml
broker:
  trading_mode: paper
  execution_enabled: true
  symbols: [SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, NVDA, META, TSLA, AMD]
  regime_timeframe: 1Day
  execution_timeframe: 5Min

hmm:
  n_candidates: [3, 4, 5, 6, 7]
  n_init: 10
  covariance_type: full
  min_train_bars: 504
  stability_bars: 3
  flicker_window: 20
  flicker_threshold: 4
  min_confidence: 0.55

strategy:
  low_vol_allocation: 0.95
  mid_vol_allocation_trend: 0.95
  mid_vol_allocation_no_trend: 0.60
  high_vol_allocation: 0.60
  low_vol_leverage: 1.25
  rebalance_threshold: 0.10
  uncertainty_size_mult: 0.50

risk:
  max_risk_per_trade: 0.01
  max_exposure: 0.80
  max_leverage: 1.25
  max_single_position: 0.15
  max_concurrent: 5
  max_daily_trades: 20
  daily_dd_reduce: 0.02
  daily_dd_halt: 0.03
  weekly_dd_reduce: 0.05
  weekly_dd_halt: 0.07
  max_dd_from_peak: 0.10

backtest:
  slippage_pct: 0.0005
  initial_capital: 100000
  train_window: 504
  test_window: 126
  step_size: 126
  risk_free_rate: 0.045

monitoring:
  dashboard_refresh_seconds: 5
  alert_rate_limit_minutes: 15
```

### `.env`

```env
ALPACA_PAPER_API_KEY=your_paper_key_here
ALPACA_PAPER_SECRET_KEY=your_paper_secret_here
ALPACA_LIVE_API_KEY=your_live_key_here
ALPACA_LIVE_SECRET_KEY=your_live_secret_here
```

---

## Functional Modules

## A1. HMM Regime Engine

### Objective

Classify market volatility regime from completed daily OHLCV features.

### Rules

- use Gaussian HMM with BIC-based model selection across `n_components ∈ [3..7]`
- use multiple random initializations
- use filtered inference only
- do **not** use `model.predict()` in backtest/live logic
- label regimes for readability only; strategy uses volatility rank, not labels

### Outputs

- current regime id
- regime label
- regime probability distribution
- expected return / volatility metadata
- regime stability / flicker diagnostics

### Acceptance criteria

- BIC selection runs and is logged
- model metadata is saved
- no-look-ahead test passes
- filtered inference returns stable probabilities bar by bar

---

## A2. Feature Engineering

### Objective

Produce HMM input features from completed daily OHLCV bars.

### Includes

- returns
- realized volatility
- volatility ratio
- normalized volume and volume trend
- ADX / trend slope
- RSI / distance from moving average
- ROC
- normalized ATR
- rolling z-score normalization

### Acceptance criteria

- features use only prior/completed data
- NaN warmup is handled explicitly
- unit tests verify no future leakage

---

## A3. Allocation Strategy Layer

### Objective

Map volatility rank into long-only target allocation and leverage.

### Strategy classes

- `LowVolBullStrategy`
- `MidVolCautiousStrategy`
- `HighVolDefensiveStrategy`
- `StrategyOrchestrator`

### Rules

- always long or flat; never short
- low vol = highest allocation
- high vol = reduced allocation, not reversal
- uncertainty mode reduces position size
- rebalance only when delta exceeds threshold

### Acceptance criteria

- same regime metadata always maps to same strategy class
- uncertainty mode halves size as configured
- rebalance threshold prevents churn in tests

---

## A4. Walk-Forward Backtesting

### Objective

Evaluate the engine with realistic rolling HMM retraining.

### Standard

- train HMM on 504 completed daily bars
- evaluate for 126 trading days out of sample
- during each out-of-sample day, allow 5-minute execution simulation
- regime is derived only from completed daily data
- mark-to-market can occur intraday

### Outputs

- `equity_curve.csv`
- `trade_log.csv`
- `regime_history.csv`
- `benchmark_comparison.csv`

### Acceptance criteria

- no-look-ahead test passes
- equity path is reproducible with same seed/config
- benchmark report is generated automatically

---

## A5. Risk Management Layer

### Objective

Apply portfolio-level and position-level veto rules independent of the HMM.

### Includes

- max exposure and leverage
- max single position
- max concurrent positions
- drawdown circuit breakers
- gap-aware overnight risk sizing
- duplicate order blocking
- correlation checks
- order validation before submission

### Acceptance criteria

- orders without stop logic are rejected
- breaker states halt trading correctly
- drawdown actions are logged with reason

---

## A6. Broker Integration

### Objective

Translate approved order plans into Alpaca-compatible order actions.

### Includes

- paper and live credential routing
- health checks
- reconnect logic
- order submission / modification / cancellation
- position reconciliation
- fill handling via websocket

### Acceptance criteria

- startup reconciliation matches broker state
- partial fills update local state correctly
- retry logic does not duplicate fills/orders
- order submission/modification uses the single-writer path only
- idempotency keys prevent duplicate execution
- partial-fill lifecycle is handled explicitly
- protective stop failures and bracket desyncs trigger reconciliation/policy handling

---

## A7. Main Orchestration Loop

### Startup

1. load config
2. load secrets
3. verify broker connectivity
4. load or train HMM
5. restore state snapshot
6. subscribe to market data
7. initialize risk + position state

### Execution loop

1. receive new 5-minute execution bar
2. update execution indicators
3. if a new completed daily bar exists, update daily HMM features and recompute filtered regime
4. apply regime stability/flicker logic
5. compute target allocation
6. validate with risk manager
7. submit/modify/reject via broker layer
8. update stops and local state
9. refresh monitoring
10. retrain weekly on completed daily data

### Acceptance criteria

- loop can restart from saved state
- loop does not double-enter after restart
- loop preserves regime state between intraday execution bars

---

## A8. Monitoring and Alerts (Engine Scope Only)

This spec includes only engine-side monitoring:

- structured logs
- terminal dashboard
- alert generation

It does **not** include web UI/dashboard product requirements. Those belong in Spec B.

---

## Explicit Exclusions

The following do **not** belong in Spec A:

- browser UI
- REST API product design
- user authentication flows
- approval inbox UI
- OpenClaw tool schemas and permissions
- chat UX

---

## Done Definition

Spec A is complete when:

- the engine can run in paper mode without any web UI
- the HMM/backtest/risk/broker loop works end to end
- state recovery works
- logs and terminal dashboard work
- all critical tests pass
