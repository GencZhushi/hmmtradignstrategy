# Claude Code Prompt: HMM Regime-Based Trading Bot — Final

**How to use:** Copy each phase into Claude Code one at a time. Complete and test each phase before moving to the next.

---

## PHASE 1: Project Scaffolding & Environment Setup

Create a Python project called `regime-trader` with the following structure:

```text
regime-trader/
├── config/
│   ├── settings.yaml              # All configurable parameters
│   └── credentials.yaml.example
├── core/
│   ├── __init__.py
│   ├── hmm_engine.py              # HMM regime detection engine
│   ├── regime_strategies.py       # Vol-based allocation strategies
│   ├── risk_manager.py            # Position sizing, leverage, drawdown limits
│   └── signal_generator.py        # Combines HMM + strategy into signals
├── broker/
│   ├── __init__.py
│   ├── alpaca_client.py           # Alpaca API wrapper
│   ├── order_executor.py          # Order placement, modification, cancellation
│   └── position_tracker.py        # Track open positions, P&L
├── data/
│   ├── __init__.py
│   ├── market_data.py             # Real-time and historical data fetching
│   └── feature_engineering.py     # Technical indicators, feature computation
├── monitoring/
│   ├── __init__.py
│   ├── logger.py                  # Structured logging
│   ├── dashboard.py               # Terminal-based live dashboard
│   └── alerts.py                  # Email/webhook alerts for critical events
├── backtest/
│   ├── __init__.py
│   ├── backtester.py              # Walk-forward allocation backtester
│   ├── performance.py             # Sharpe, drawdown, regime breakdown, benchmarks
│   └── stress_test.py             # Crash injection, gap simulation
├── tests/
│   ├── test_hmm.py
│   ├── test_look_ahead.py         # Verify no look-ahead bias
│   ├── test_strategies.py
│   ├── test_risk.py
│   └── test_orders.py
├── main.py                        # Entry point
├── requirements.txt
├── .env.example
└── README.md
```

Set up `requirements.txt` with:

- `hmmlearn`
- `alpaca-trade-api`
- `alpaca-py`
- `pandas`, `numpy`, `scipy`
- `ta` (technical analysis library)
- `scikit-learn`
- `pyyaml`
- `python-dotenv`
- `websocket-client`
- `schedule`
- `rich` (for terminal dashboard)

Create `settings.yaml` with **ALL** parameters, grouped by section, with defaults and comments:

- `broker (paper_trading: true, symbols: [SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, NVDA, META, TSLA, AMD], timeframe: 1Day)`
- `hmm (n_candidates: [3, 4, 5, 6, 7], n_init: 10, covariance_type: full, min_train_bars: 252, stability_bars: 3, flicker_window: 20, flicker_threshold: 4, min_confidence: 0.55)`
- `strategy (low_vol_allocation: 0.95, mid_vol_allocation_trend: 0.95, mid_vol_allocation_no_trend: 0.60, high_vol_allocation: 0.60, low_vol_leverage: 1.25, rebalance_threshold: 0.10, uncertainty_size_mult: 0.50)`
- `risk (max_risk_per_trade: 0.01, max_exposure: 0.80, max_leverage: 1.25, max_single_position: 0.15, max_concurrent: 5, max_daily_trades: 20, daily_dd_reduce: 0.02, daily_dd_halt: 0.03, weekly_dd_reduce: 0.05, weekly_dd_halt: 0.07, max_dd_from_peak: 0.10)`
- `backtest (slippage_pct: 0.0005, initial_capital: 100000, train_window: 252, test_window: 126, step_size: 126, risk_free_rate: 0.045)`
- `monitoring (dashboard_refresh_seconds: 5, alert_rate_limit_minutes: 15)`

Create `.env.example` with:

```env
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true
```

Do **NOT** implement any logic yet — just the skeleton with imports, class stubs, type hints, and docstrings.

Add `.env` and `credentials.yaml` to `.gitignore`.

---

## PHASE 2: HMM Regime Detection Engine

Implement `core/hmm_engine.py` and `data/feature_engineering.py`.

### DESIGN PHILOSOPHY

The HMM is a **VOLATILITY CLASSIFIER**. It detects whether the market is in a calm, moderate, or turbulent volatility environment. It does **NOT** predict price direction. The strategy layer uses the volatility classification to set portfolio allocation — be fully invested when conditions are calm, reduce when turbulent.

### REQUIREMENTS

#### 1. Gaussian HMM with automatic model selection

- Test `n_components = [3, 4, 5, 6, 7]` during training
- For each candidate, train and compute BIC (Bayesian Information Criterion)
- `BIC = -2 * log_likelihood + n_params * log(n_samples)`
- Select lowest BIC score (simplest model that explains the data)
- Run multiple random initializations per candidate (`n_init=10`)
- Log ALL candidate BIC scores and which was selected

After training, sort regimes by mean return (ascending) for **LABELING**:

- Lowest return → `CRASH / BEAR`
- Highest return → `BULL / EUPHORIA`
- Assign labels based on selected count:
  - 3 regimes: `BEAR, NEUTRAL, BULL`
  - 4 regimes: `CRASH, BEAR, BULL, EUPHORIA`
  - 5 regimes: `CRASH, BEAR, NEUTRAL, BULL, EUPHORIA`
  - 6 regimes: `CRASH, STRONG_BEAR, WEAK_BEAR, WEAK_BULL, STRONG_BULL, EUPHORIA`
  - 7 regimes: `CRASH, STRONG_BEAR, WEAK_BEAR, NEUTRAL, WEAK_BULL, STRONG_BULL, EUPHORIA`

**IMPORTANT:** Labels are sorted by return for human readability. But the **STRATEGY** layer sorts by **VOLATILITY** independently. The labels don't drive strategy decisions.

#### 2. Observable features (inputs to HMM)

Implement in `data/feature_engineering.py` as pure functions.

Compute from OHLCV:

- Returns: log returns over 1, 5, 20 periods
- Volatility: realized vol (20-period rolling std), vol ratio (5-period / 20-period)
- Volume: normalized volume (z-score vs 50-period mean), volume trend (slope of 10-period SMA)
- Trend: ADX (14-period), slope of 50-period SMA
- Mean reversion: RSI(14) z-score, distance from 200 SMA as % of price
- Momentum: ROC 10 and 20 period
- Range: normalized ATR (14-period ATR / close)

Standardize **ALL** features with rolling z-scores (252-period lookback).

#### 3. Model training

- `hmmlearn.GaussianHMM`, covariance_type=`"full"`
- Minimum 2 years daily data (504 trading days)
- Expanding window retraining: retrain at configurable intervals
- Store model with pickle + metadata (`n_regimes, bic, training_date, labels`)
- Log: likelihood, BIC, convergence, iterations

#### 4. Regime detection — no look-ahead bias

***THIS IS THE MOST IMPORTANT TECHNICAL DETAIL.***

Do **NOT** use `model.predict()`. `predict()` runs the Viterbi algorithm which processes the **ENTIRE** sequence and revises past states using future data. This is look-ahead bias that makes backtests unrealistically good.

Instead implement **FORWARD ALGORITHM ONLY** (filtered inference):

```python
def predict_regime_filtered(self, features_up_to_now):
    """
    Compute P(state_t | observations_1:t) using forward algorithm.
    Uses ONLY past and present data. No future data.
    """

    # Use model's startprob_, transmat_, means_, covars_
    # Implement forward pass manually:
    # 1. alpha_0 = startprob * emission_prob(obs_0)
    # 2. alpha_t = (alpha_{t-1} @ transmat) * emission_prob(obs_t)
    # 3. Normalize at each step (work in log space)
    # 4. alpha_T = filtered distribution at current time
    # Cache previous alpha for efficiency in live/backtest loop
```

Mandatory test — `tests/test_look_ahead.py`:

```python
def test_no_look_ahead_bias():
    """Regime at T must be identical with data[0:T] vs data[0:T+100]."""
    model = train_hmm(full_data)
    regime_short = predict_regime_filtered(data[0:400])[-1]
    regime_long = predict_regime_filtered(data[0:500])[400]
    assert regime_short == regime_long, "LOOK-AHEAD BIAS DETECTED"
```

#### 5. Regime stability filter

- Regime change only "confirmed" after persisting N bars (default 3)
- During transition: keep previous regime, reduce sizes by 25%
- Track flicker rate (changes per 20 bars)
- If flicker rate > threshold (default 4): force uncertainty mode

#### 6. Additional methods

- `predict_regime_proba()` -> probability distribution
- `get_regime_stability()` -> consecutive bars in current regime
- `get_transition_matrix()` -> learned transition probabilities
- `detect_regime_change()` -> True only if confirmed
- `get_regime_flicker_rate()` -> changes per window
- `is_flickering()` -> True if flicker rate exceeds threshold

#### 7. Regime metadata

`RegimeInfo` dataclass:

- `regime_id, regime_name, expected_return, expected_volatility`
- `recommended_strategy_type, max_leverage_allowed`
- `max_position_size_pct, min_confidence_to_act`

`RegimeState` dataclass:

- `label, state_id, probability, state_probabilities`
- `timestamp, is_confirmed, consecutive_bars`

Log regime changes as **WARNING**. Log confirmations as **INFO**.

---

## PHASE 3: Volatility-Based Allocation Strategy

Implement `core/regime_strategies.py` — the allocation layer that sizes positions based on the HMM's volatility regime detection.

### DESIGN INSIGHT

The HMM excels at detecting **VOLATILITY ENVIRONMENTS**, not market direction. Stocks trend upward roughly 70% of the time in low-volatility periods. The worst drawdowns cluster in high-volatility spikes. So the strategy is simple:

- Low vol → be fully invested (calm markets trend up)
- Mid vol → stay invested if trend intact, reduce if not
- High vol → reduce but stay partially invested (catch V-shaped rebounds)

The edge comes from **AVOIDING BIG DRAWDOWNS** through vol-based sizing. When you cut your worst drawdown in half, compounding works in your favor over time.

### ALWAYS LONG. NEVER SHORT.

Shorting was tested extensively in walk-forward backtesting and consistently destroyed returns because:

1. Markets have long-term upward drift
2. V-shaped recoveries happen fast and the HMM is 2–3 days late detecting them
3. Short positions during rebounds wipe out crash gains

The correct response to high volatility is **REDUCING** allocation, not reversing direction.

### THREE STRATEGY CLASSES (based on volatility rank)

#### 1. `LowVolBullStrategy` (lowest third of regimes by `expected_volatility`)

- Direction: `LONG`
- Allocation: 95% of portfolio
- Leverage: 1.25x (modest leverage in calm conditions)
- Stop: `max(price - 3 ATR, 50 EMA - 0.5 ATR)`
- This is where most returns are generated. Calm markets + leverage = compounding.

#### 2. `MidVolCautiousStrategy` (middle third by `expected_volatility`)

- Direction: `LONG`
- If price > 50 EMA: allocation 95%, leverage 1.0x (trend intact, stay invested)
- If price < 50 EMA: allocation 60%, leverage 1.0x (trend broken, reduce)
- Stop: `50 EMA - 0.5 ATR`

#### 3. `HighVolDefensiveStrategy` (top third by `expected_volatility`)

- Direction: `LONG` (NOT short)
- Allocation: 60% of portfolio
- Leverage: 1.0x
- Stop: `50 EMA - 1.0 ATR` (wider for volatile conditions)
- Staying 60% invested catches the sharp rebounds after selloffs.

### VOLATILITY RANK MAPPING

For any regime count (3–7), map each regime's vol rank to a strategy:

- `position = rank / (n_regimes - 1)`  # 0.0 = lowest vol, 1.0 = highest
- `position <= 0.33` → `LowVolBullStrategy`
- `position >= 0.67` → `HighVolDefensiveStrategy`
- else → `MidVolCautiousStrategy`

### STRATEGY ORCHESTRATOR

- Takes `regime_infos` from HMM
- Sorts by `expected_volatility` (ascending) to compute `vol_rank` per regime
- Maps `regime_id → vol_rank → strategy class`
- This sort is **INDEPENDENT** of the label sort (which is by return)
- `"BULL"` label does **NOT** mean low vol. The orchestrator ignores labels.

### CONFIDENCE AND UNCERTAINTY

- Minimum confidence threshold: `0.55` (configurable)
- Uncertainty mode triggers when: `prob < threshold`, or `is_flickering=True`
- In uncertainty mode: halve all position sizes, force leverage to 1.0x
- Append `"UNCERTAINTY — size halved"` to reasoning

### REBALANCING

- Only rebalance when target allocation differs from current by >10%
- This prevents churn from minor probability fluctuations
- Fewer trades = less slippage = better real-world performance

### IMPLEMENTATION

- `BaseStrategy` ABC: `generate_signal(symbol, bars, regime_state) -> Optional[Signal]`
- `LowVolBullStrategy`, `MidVolCautiousStrategy`, `HighVolDefensiveStrategy`
- `StrategyOrchestrator`:
  - `__init__(config, regime_infos)`: sorts by vol, maps strategies
  - `generate_signals(symbols, bars, regime_state, is_flickering) -> list[Signal]`
  - `update_regime_infos(regime_infos)`: rebuilds mapping after HMM retrain

`Signal` dataclass:

- `symbol, direction (LONG or FLAT), confidence, entry_price, stop_loss`
- `take_profit (Optional), position_size_pct (0.60 to 0.95)`
- `leverage (1.0 or 1.25), regime_id, regime_name, regime_probability`
- `timestamp, reasoning, strategy_name, metadata`

Keep backward-compatible aliases:

- `CrashDefensiveStrategy = HighVolDefensiveStrategy`
- `BearTrendStrategy = HighVolDefensiveStrategy`
- `MeanReversionStrategy = MidVolCautiousStrategy`
- `BullTrendStrategy = LowVolBullStrategy`
- `EuphoriaCautiousStrategy = LowVolBullStrategy`
- etc.

Create `LABEL_TO_STRATEGY` dict for all possible labels → strategy class.

---

## PHASE 4: Walk-Forward Backtesting & Validation

Implement `backtest/backtester.py`, `backtest/performance.py`, and `backtest/stress_test.py`.

This is an **ALLOCATION-BASED** walk-forward backtester. It does **NOT** track individual trade entries and exits. It sets a target portfolio allocation each bar based on the detected volatility regime and rebalances when the allocation changes meaningfully. This is how real systematic strategies work.

### 1. Walk-forward optimization engine (`backtester.py`)

Rolling windows:

- In-Sample (IS): 252 trading days (1 year) for HMM training + model selection
- Out-of-Sample (OOS): 126 trading days (6 months) for evaluation
- Step size: 126 trading days (6 months)

For each window:

a. Train HMM on IS data (BIC model selection)  
b. Compute vol rankings from trained model's `regime_infos`  
c. Walk through OOS bar by bar:

- Compute features using **ONLY** data up to current bar
- Run filtered HMM (forward algorithm)
- Get strategy signal: target allocation based on vol rank
- If allocation changed >10% from current → rebalance
- Mark to market: `equity = cash + shares * price`

d. Record regime predictions and equity at each bar  
e. Record a "trade" whenever allocation changes for metrics

### ALLOCATION MATH (must be exactly correct)

```text
equity = cash + shares * current_price
target_shares = int(equity * target_allocation / current_price)
delta = target_shares - current_shares
cash -= delta * price
shares = target_shares
```

When leverage > 1.0 (e.g., 1.25x in low vol), `target_allocation > 1.0`, so `target_shares * price > equity`, making cash negative. This is margin.

`equity = cash + shares * price` is still correct because share value exceeds the margin debt. Alpaca supports this with 2x overnight leverage.

### REALISTIC SIMULATION

- Slippage: 0.05% on each rebalance (configurable)
- Rebalancing threshold: 10% (prevents churn)
- Fill delay: 1 bar (signal bar N → rebalance at bar N+1 open)
- No individual trade stops in backtester (stops are for live trading only)
- Commission: $0 default (Alpaca commission-free)

### 2. Performance metrics (`performance.py`)

Core:

- Total return (%), CAGR
- Sharpe ratio (annualized), Sortino ratio
- Calmar ratio (CAGR / max drawdown)
- Max drawdown: percentage **AND** duration in trading days
- Win rate, avg win/loss, profit factor
- Total trades, avg holding period

Regime-specific (table format):

- `Regime | % Time In | Return Contribution | Avg Trade P&L | Win Rate | Sharpe`
- Show for each detected regime. This proves each vol environment's strategy is performing as expected.

Confidence-bucketed:

- `Confidence | Trades | Sharpe | Win Rate | Avg P&L`
- `< 50%, 50-60%, 60-70%, 70%+`
- If high-confidence trades outperform low-confidence → HMM adds value.

Benchmark comparisons (run automatically with `--compare` flag):

a. Buy-and-hold: hold the asset entire period  
b. 200 SMA trend: long above 200 SMA, cash below  
c. Random entry + same risk management: random allocation changes at same frequency, same position sizing rules. 100 random seeds, report mean/std.

Worst-case:

- worst day, worst week, worst month, max consecutive losses, longest time underwater.

Output:

- Rich formatted tables to terminal
- `equity_curve.csv`, `trade_log.csv`, `regime_history.csv`, `benchmark_comparison.csv`

### 3. Stress testing (`stress_test.py`)

a. Crash injection: insert -5% to -15% single-day gaps at 10 random points.  
Run 100 Monte Carlo simulations. Report: mean max loss, worst case, % where circuit breaker fired.

b. Gap risk: insert overnight gaps of 2-5x ATR at random points.  
Report: expected loss vs actual.

c. Regime misclassification: deliberately shuffle regime labels.  
Verify risk management contains damage even with wrong regimes.  
If system blows up → risk management isn't independent enough.

---

## PHASE 5: Risk Management Layer

Implement `core/risk_manager.py`.

The risk manager operates **INDEPENDENTLY** of the HMM. Even if the HMM fails completely, circuit breakers catch drawdowns based on actual P&L. Defense in depth. The risk manager has **ABSOLUTE VETO POWER** over any signal.

### 1. Portfolio-level limits

- Max total exposure: 80% of portfolio (20% cash minimum — note: when using 1.25x leverage, the notional exposure exceeds equity but the margin requirement stays within Alpaca's limits)
- Max single position: 15%
- Max correlated exposure: 30% in one sector
- Max concurrent positions: 5
- Max daily trades: 20
- Max portfolio leverage: 1.25x

### 2. Circuit breakers (fire on actual P&L, independent of regime)

- Daily DD > 2%: reduce all sizes 50% rest of day
- Daily DD > 3%: close ALL positions, halt rest of day
- Weekly DD > 5%: reduce all sizes 50% rest of week
- Weekly DD > 7%: close ALL, halt rest of week
- Peak DD > 10%: halt ALL trading, write `trading_halted.lock` file requiring manual deletion to resume

Log every trigger with: breaker type, actual DD, equity, positions closed, HMM regime at time (track if HMM was wrong).

### 3. Position-level risk

- Every position MUST have a stop loss — system refuses orders without one
- Max risk per trade: 1% of portfolio
- Position size = `(portfolio * 0.01) / abs(entry - stop_loss)`
- Cap at regime max, then portfolio max (15%)
- Minimum position: $100
- GAP RISK: overnight positions assume 3x stop gap-through.  
  Overnight size = `min(normal, size where 3x gap = 2% of portfolio)`

### 4. Leverage rules

- Default: 1.0x
- Only low-vol regimes may use up to 1.25x
- Force 1.0x if: regime uncertain, any circuit breaker active, 3+ positions open, high flicker rate
- Alpaca supports 2x overnight (Reg T, $2k+ equity) and 4x intraday (PDT, $25k+ equity). Our 1.25x max is deliberately conservative.

### 5. Order validation

- Check buying power, tradable status, bid-ask spread < 0.5%
- Block duplicates (same symbol + direction within 60 seconds)
- Log every rejection with structured reason

### 6. Correlation check

- 60-day rolling correlation with existing positions
- Correlation > 0.7: reduce size 50%
- Correlation > 0.85: reject trade

### IMPLEMENTATION

- `RiskManager: validate_signal(signal, portfolio_state) -> RiskDecision`
- `RiskDecision: approved, modified_signal, rejection_reason, modifications list`
- `PortfolioState: equity, cash, buying_power, positions, daily/weekly pnl, peak equity, drawdown, circuit_breaker_status, flicker_rate`
- `CircuitBreaker: check(), update(pnl), reset_daily(), reset_weekly(), get_history()`
- All thresholds from `settings.yaml`

---

## PHASE 6: Alpaca Broker Integration

Implement the `broker/` package.

### 1. `broker/alpaca_client.py`

- alpaca-py SDK wrapper
- Credentials from `.env` (NEVER hardcoded, `.env` in `.gitignore`)
- Paper: `https://paper-api.alpaca.markets` (DEFAULT)
- Live: `https://api.alpaca.markets`
- If `paper_trading: false`, require confirmation:
  - **"4 LIVE TRADING MODE. Type 'YES I UNDERSTAND THE RISKS' to confirm:"**
- Methods: `get_account()`, `get_positions()`, `get_order_history()`, `is_market_open()`, `get_clock()`, `get_available_margin()`
- Health check on startup, auto-reconnect with exponential backoff

### 2. `broker/order_executor.py`

- `submit_order(signal)`: LIMIT orders by default (+/- 0.1% of current price), cancel after 30s if unfilled, optionally retry at market
- `submit_bracket_order(signal)`: entry + stop + take_profit via Alpaca OCO
- `modify_stop(symbol, new_stop)`: only tighten, never widen
- `cancel_order()`, `close_position()`, `close_all_positions()`
- Unique `trade_id` linking `signal → risk_decision → order → fill`

### 3. `broker/position_tracker.py`

- WebSocket subscription for instant fill notifications
- Update `PortfolioState` and `CircuitBreaker` on every fill
- Per-position tracking: entry time/price, current price, unrealized P&L, stop level, holding period, regime at entry vs current
- Sync with Alpaca on startup (reconcile tracked vs actual positions)

### 4. `data/market_data.py`

- `get_historical_bars(symbol, timeframe, start, end)`
- `subscribe_bars(symbols, timeframe, callback)` via WebSocket
- `subscribe_quotes(symbols, callback)` for spread checks
- `get_latest_bar()`, `get_latest_quote()`, `get_snapshot()`
- Handle gaps (weekends, holidays, halts) gracefully

---

## PHASE 7: Main Loop & Orchestration

Implement `main.py`.

### STARTUP

1. Load config, connect to Alpaca, verify account
2. Check market hours (wait or exit if closed)
3. Load or train HMM (if model >7 days old or missing, retrain)
4. Initialize risk manager with current portfolio from Alpaca
5. Initialize position tracker, sync positions
6. Check for `state_snapshot.json` (recovery from previous session)
7. Start WebSocket data feeds
8. Print system state, log "System online"

### MAIN LOOP (each bar close, default 5-min bars)

1. New bar from WebSocket
2. Compute features (rolling window, no future data)
3. Filtered HMM prediction (forward algorithm only)
4. Regime stability check (3-bar persistence)
5. Flicker rate check → uncertainty mode if high
6. StrategyOrchestrator: target allocation per symbol
7. For each signal: `risk_manager.validate_signal()`
   - approved: `order_executor.submit_order()`
   - modified: log, submit modified
   - rejected: log reason
8. Update trailing stops per regime
9. Circuit breaker check
10. Dashboard refresh
11. Weekly: retrain HMM

### SHUTDOWN (SIGINT/SIGTERM)

- Close WebSocket connections
- Do NOT close positions (stops in place)
- Save `state_snapshot.json`
- Print session summary

### ERROR HANDLING

- Alpaca API: 3 retries, exponential backoff
- HMM error: hold current regime
- Data feed drop: pause signals, keep stops active
- Unhandled: log traceback, save state, alert

### CLI

- `--dry-run` — Full pipeline, no orders
- `--backtest` — Walk-forward backtester
- `--train-only` — Train HMM and exit
- `--stress-test` — Run stress tests
- `--compare` — Benchmark comparisons
- `--dashboard` — Show dashboard for running instance

---

## PHASE 8: Monitoring, Alerts & Dashboard

Implement `monitoring/` package.

### 1. `monitoring/logger.py`

- Structured JSON logging
- Rotating files (10MB, 30 days): `main.log`, `trades.log`, `alerts.log`, `regime.log`
- Every entry includes: `timestamp, regime, probability, equity, positions, daily_pnl`

### 2. `monitoring/dashboard.py` (rich library)

```text
┌ REGIME ───────────────────────────────────────────────┐
│ BULL (72%) | Stability: 14 bars | Flicker: 1/20 |    │
├ PORTFOLIO ────────────────────────────────────────────┤
│ Equity: $105,230 | Daily: +$340 (+0.32%) |           │
│ Allocation: 95% | Leverage: 1.25x                    │
├ POSITIONS ────────────────────────────────────────────┤
│ SPY | LONG | $520.30 | +1.2% | Stop: $508 | 3h |     │
├ RECENT SIGNALS ───────────────────────────────────────┤
│ 14:30 | SPY | Rebalance 60%→95% | Low vol |          │
├ RISK STATUS ──────────────────────────────────────────┤
│ Daily DD: 0.3%/3% ✅ | From Peak: 1.2%/10% ✅ |       │
├ SYSTEM ───────────────────────────────────────────────┤
│ Data: ✅ | API: ✅ 23ms | HMM: 2d ago | PAPER |       │
└───────────────────────────────────────────────────────┘
```

- Refresh every 5 seconds
- Color-coded risk bars

### 3. `monitoring/alerts.py`

- Triggers: regime change, circuit breaker, large P&L, data feed down, API lost, HMM retrained, flicker exceeded
- Delivery: console, log file, email (optional), webhook (optional)
- Rate limit: 1 per event type per 15 minutes

---

## PHASE 9: Integration Testing & Documentation

### 1. TESTS

- End-to-end dry run: `data → HMM → strategy → risk → simulated orders`
- Look-ahead bias: `test_look_ahead.py` passes, backtest identical with different end dates
- Risk stress: extreme signals capped, rapid-fire blocked, no-stop rejected
- Alpaca paper: place bracket order, modify stop, cancel, verify clean state
- Recovery: kill process, restart, verify state recovery and no double-entry

### 2. `README.md`

- Philosophy: `risk management > signal generation`
- Architecture diagram: `data → features → HMM → vol rank → allocation → risk → broker`
- Quick start (6 steps)
- CLI reference
- Configuration guide
- FAQ: forward algorithm, BIC selection, trade rejections, live trading switch
- Disclaimer: educational, no guaranteed profits, paper trade first

---

## OPENCLAW + WEB APP ADAPTATION PATCH

This section is **not** from the screenshots. It is an implementation patch to make the system usable from a web page and controllable through an OpenClaw agent.

### GOAL

Keep the trading engine as the single source of truth, but expose it through:

1. a web dashboard for monitoring and manual control
2. an internal API for structured commands
3. an OpenClaw-compatible agent action layer so the agent can request, preview, approve, and execute trades through defined endpoints instead of raw free-form code execution

### CORE PRINCIPLE

The OpenClaw agent must **never** bypass:

`Agent/User Request → Intent Parser → Risk Manager → Order Preview → Execution Policy → Broker`

The agent may suggest or initiate trade workflows, but all actual orders must go through the exact same risk, position, leverage, and circuit-breaker checks already defined in this document.

### REQUIRED PROJECT ADDITIONS

Add these packages/folders to the project:

```text
regime-trader/
├── api/
│   ├── __init__.py
│   ├── app.py                     # FastAPI entrypoint
│   ├── auth.py                    # API key/JWT/service token auth
│   ├── schemas.py                 # Pydantic request/response models
│   ├── dependencies.py            # Shared DI helpers
│   └── routes/
│       ├── health.py
│       ├── market.py
│       ├── portfolio.py
│       ├── regime.py
│       ├── signals.py
│       ├── orders.py
│       ├── agent.py
│       └── config.py
├── integrations/
│   └── openclaw/
│       ├── __init__.py
│       ├── tool_adapter.py        # Maps OpenClaw requests to internal API calls
│       ├── command_parser.py      # Converts chat intent to structured TradeIntent
│       ├── policy.py              # Agent permissions and live-trading guardrails
│       └── prompts.md             # System prompt / tool contract for OpenClaw
├── storage/
│   ├── __init__.py
│   ├── models.py                  # SQLAlchemy models or persistence layer
│   ├── repository.py
│   └── migrations/
├── web/
│   ├── README.md
│   └── (frontend app: Next.js/React or similar)
└── state/
    ├── approvals/
    ├── snapshots/
    └── audit/
```

### ADDITIONAL DEPENDENCIES

Add these to `requirements.txt` for web/API/OpenClaw mode:

- `fastapi`
- `uvicorn[standard]`
- `pydantic`
- `httpx`
- `sqlalchemy`
- `alembic`
- `orjson`
- `python-jose` or `PyJWT`
- `passlib`
- `aiofiles`
- `websockets`

### FIXES / NORMALIZATIONS TO EARLIER SECTIONS

Normalize the earlier configuration so it is production-ready:

- In `risk` config, replace the duplicated weekly field with:
  - `weekly_dd_reduce: 0.05`
  - `weekly_dd_halt: 0.07`
- Keep paper trading as the default everywhere.
- Use one canonical internal state model for positions, pending approvals, risk flags, and agent actions.
- Treat the web UI and OpenClaw as clients of the same API — do **not** let them implement separate trading logic.

### API DESIGN

Implement `api/app.py` with FastAPI and expose the following endpoints.

#### Read endpoints

- `GET /health` — service status, broker connectivity, HMM age, data feed status
- `GET /regime/current` — current regime, confidence, flicker, stability, probabilities
- `GET /portfolio` — equity, cash, buying power, exposure, leverage, drawdown, daily/weekly P&L
- `GET /positions` — open positions with stop, unrealized P&L, holding period, regime at entry
- `GET /signals/latest` — latest signals and whether they were approved/modified/rejected
- `GET /orders/history` — submitted, filled, canceled, rejected orders
- `GET /risk/status` — breaker state, active constraints, blocked symbols, uncertainty mode
- `GET /agent/pending` — trade intents waiting for human approval or policy release
- `GET /audit/logs` — summarized audit trail for agent and user actions

#### Action endpoints

- `POST /signals/preview` — generate signal preview without sending orders
- `POST /orders/preview` — convert a proposed trade into an `OrderPlan` after risk checks
- `POST /orders/execute` — execute an already-approved order plan
- `POST /orders/cancel` — cancel pending order
- `POST /positions/close` — close one position
- `POST /positions/close-all` — close all positions
- `POST /agent/trade-intent` — submit structured trade intent from OpenClaw
- `POST /agent/approve` — approve pending intent/order plan
- `POST /agent/reject` — reject pending intent/order plan
- `POST /agent/arm-live` — temporary live-trading unlock with short TTL and audit log
- `POST /config/reload` — reload settings without restarting

### STRUCTURED AGENT CONTRACT

OpenClaw should not send raw broker instructions. It should send a normalized payload like:

```json
{
  "intent_type": "open_position",
  "symbol": "SPY",
  "direction": "LONG",
  "allocation_pct": 0.10,
  "thesis": "Low-vol regime with confirmed trend",
  "timeframe": "5m",
  "urgency": "normal",
  "source": "openclaw",
  "requires_confirmation": true
}
```

Convert that into the following pipeline:

`TradeIntent → RiskDecision → OrderPlan → BrokerExecution → AuditEvent`

### OPENCLAW TOOL ADAPTER BEHAVIOR

The adapter should support commands such as:

- "Show current regime"
- "What positions are open?"
- "Preview a 10% SPY allocation"
- "Reduce exposure by 20%"
- "Close all positions"
- "Why was my last trade rejected?"
- "Switch to paper mode"

The adapter must map each request to one or more internal API calls and return structured results, not free-form guessed state.

### EXECUTION POLICY FOR AGENT-DRIVEN TRADING

Implement `integrations/openclaw/policy.py` with these rules:

- Default mode: `paper_only`
- Live mode must be explicitly armed for a short session window, e.g. 15 minutes
- Agent may preview trades without restriction
- Agent may execute paper trades automatically if `auto_execute_paper=true`
- Agent may execute live trades only if all of the following are true:
  - `live_enabled=true`
  - session is armed
  - request passes risk manager
  - request is below configured notional and leverage caps
  - request is logged with full audit metadata
- Any breaker/halt state immediately disables new agent executions
- Any uncertainty mode may force agent requests into preview-only mode

### APPROVAL WORKFLOW

Support both of these modes:

#### Mode A: Manual approval

`OpenClaw/User message → preview → pending approval → approve in web UI → execute`

#### Mode B: Controlled automation

`OpenClaw/User message → preview → policy check → execute automatically (paper only by default)`

### WEB APP REQUIREMENTS

Build a responsive web dashboard that consumes the API.

#### Main pages

1. **Overview**
   - regime card
   - portfolio summary
   - risk status
   - P&L chart
   - active alerts

2. **Positions**
   - open positions table
   - stop level, holding time, regime at entry, unrealized P&L
   - close / reduce / tighten stop actions

3. **Signals**
   - latest generated signals
   - confidence score
   - regime rationale
   - preview / approve / reject buttons

4. **Agent Console**
   - chat panel for OpenClaw-style commands
   - structured response panel
   - pending approvals list
   - execution history

5. **Audit & Logs**
   - who requested what
   - what policy/risk rule changed the request
   - final broker outcome

6. **Settings**
   - paper/live mode
   - auto-execute paper toggle
   - approval requirements
   - API/service tokens
   - OpenClaw integration status

### WEBSOCKET / STREAMING EVENTS

Expose a live event stream for both the web UI and OpenClaw integration:

- `regime_changed`
- `signal_generated`
- `signal_rejected`
- `order_submitted`
- `order_filled`
- `position_updated`
- `circuit_breaker_triggered`
- `agent_intent_pending`
- `agent_intent_approved`
- `agent_intent_rejected`
- `system_alert`

### AUTHENTICATION

Use separate auth modes:

- **Web user auth**: username/password + JWT session
- **OpenClaw service auth**: API key or signed service token
- **Admin actions** (live-mode arming, close-all, config changes): elevated role required

### PERSISTENCE REQUIREMENTS

Persist at minimum:

- portfolio snapshots
- orders and fills
- pending approvals
- agent intents
- audit events
- breaker history
- HMM model metadata
- last known regime state

SQLite is fine for a single-user local deployment; Postgres is preferred for reliability.

### NEW DATA MODELS

Add these dataclasses / schemas:

#### `TradeIntent`

- `intent_id`
- `source` (`user`, `web`, `openclaw`, `scheduler`)
- `symbol`
- `direction`
- `allocation_pct`
- `requested_leverage`
- `thesis`
- `requires_confirmation`
- `created_at`
- `status`

#### `OrderPlan`

- `plan_id`
- `intent_id`
- `approved_signal`
- `risk_adjusted_size`
- `risk_adjusted_leverage`
- `entry_type`
- `limit_price`
- `stop_loss`
- `take_profit`
- `expiration`
- `status`

#### `AuditEvent`

- `event_id`
- `actor`
- `actor_type`
- `action`
- `resource_type`
- `resource_id`
- `before`
- `after`
- `reason`
- `timestamp`

### PHASE 8 DASHBOARD ENHANCEMENT

Extend the earlier dashboard with agent-aware sections:

```text
┌ AGENT STATUS ─────────────────────────────────────────┐
│ OpenClaw: CONNECTED | Mode: PAPER | Auto: OFF |      │
├ PENDING APPROVALS ────────────────────────────────────┤
│ 1) BUY SPY 10% | Reason: Low-vol trend | Waiting     │
├ LAST AGENT ACTION ────────────────────────────────────┤
│ 14:32 | Previewed QQQ rebalance | Not executed       │
└───────────────────────────────────────────────────────┘
```

### PHASE 9 TESTING EXTENSION

Add integration tests for the web/API/agent layer:

- OpenClaw preview request returns structured `OrderPlan`
- OpenClaw execution request is blocked when breaker is active
- Live-mode execution is rejected when session is not armed
- Duplicate agent intent is idempotent
- Web approve → execute path works end to end
- API restart preserves pending approvals and audit logs

### NEW PHASE 10: DEPLOYMENT MODES

Support three run modes:

#### Mode 1. Local research

- backtest + dashboard only
- no broker execution
- OpenClaw connected in preview-only mode

#### Mode 2. Paper trading

- Alpaca paper account
- web UI enabled
- OpenClaw allowed to preview and optionally auto-execute paper trades

#### Mode 3. Live trading

- explicit live enable flag
- short-lived arming token required
- stricter limits than paper mode
- all actions fully audited

### CLI ADDITIONS

Add these CLI flags:

- `--serve-api` — run FastAPI server
- `--webhook-mode` — enable webhook endpoints for OpenClaw or external triggers
- `--agent-mode` — enable OpenClaw integration
- `--paper-auto` — allow auto-execution in paper mode
- `--arm-live` — manually arm live execution for current session
- `--require-approval` — force manual approval even in paper mode

### RECOMMENDED STACK

For best OpenClaw compatibility:

- **Backend:** FastAPI + Pydantic + SQLAlchemy
- **Frontend:** Next.js or React dashboard
- **Broker:** Alpaca
- **Streaming:** WebSocket / Server-Sent Events
- **Agent bridge:** OpenClaw tool adapter calling internal API
- **Persistence:** SQLite first, Postgres later

### FINAL IMPLEMENTATION RULE

The web page, the OpenClaw agent, and any future automation must all use the same internal trading engine and policy enforcement path. One engine, many interfaces — never many engines.

---

## Note

This Markdown file is a cleaned transcription of the screenshots you shared, plus an added OpenClaw/web-app adaptation patch and a few practical corrections for implementation.
