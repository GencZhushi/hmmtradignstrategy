"""Walk-forward backtester with daily HMM retraining + intraday execution (Spec A5).

Operates on **adjusted daily bars**. Intraday execution is simulated using the
slippage model; the backtester never looks ahead because the HMM is re-fit
inside each in-sample window and inference is filtered-only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from core.hmm_engine import VolatilityRegimeHMM
from core.regime_strategies import StrategyConfig, StrategyOrchestrator
from core.types import Direction
from data.feature_engineering import FeatureEngine
from data.slippage import SlippageModel

LOG = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a single walk-forward run."""

    train_window: int = 504
    test_window: int = 126
    step_size: int = 126
    initial_capital: float = 100_000.0
    slippage: SlippageModel = field(default_factory=SlippageModel)
    risk_free_rate: float = 0.045
    rebalance_threshold: float = 0.10
    hmm_n_candidates: tuple[int, ...] = (3, 4, 5)
    hmm_n_init: int = 2
    hmm_covariance: str = "full"
    random_state: int | None = 42

    @classmethod
    def from_settings(cls, backtest_cfg: Mapping[str, object], strategy_cfg: Mapping[str, object], hmm_cfg: Mapping[str, object]) -> "BacktestConfig":
        return cls(
            train_window=int(backtest_cfg.get("train_window", 504)),
            test_window=int(backtest_cfg.get("test_window", 126)),
            step_size=int(backtest_cfg.get("step_size", 126)),
            initial_capital=float(backtest_cfg.get("initial_capital", 100_000.0)),
            slippage=SlippageModel(base_pct=float(backtest_cfg.get("slippage_pct", 0.0005))),
            risk_free_rate=float(backtest_cfg.get("risk_free_rate", 0.045)),
            rebalance_threshold=float(strategy_cfg.get("rebalance_threshold", 0.10)),
            hmm_n_candidates=tuple(hmm_cfg.get("n_candidates", (3, 4, 5))),
            hmm_n_init=int(hmm_cfg.get("n_init", 2)),
            hmm_covariance=str(hmm_cfg.get("covariance_type", "full")),
        )


@dataclass
class BacktestArtifacts:
    """Paths and in-memory frames returned from a walk-forward run."""

    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    regime_history: pd.DataFrame
    benchmark_comparison: pd.DataFrame
    output_dir: Path | None = None


@dataclass
class WalkForwardBacktester:
    """Walk-forward backtester for a single symbol (usually SPY or QQQ)."""

    config: BacktestConfig
    feature_engine: FeatureEngine = field(default_factory=FeatureEngine)
    strategy_config: StrategyConfig = field(default_factory=StrategyConfig)

    def run_walk_forward(
        self,
        *,
        symbol: str,
        daily_bars: pd.DataFrame,
    ) -> BacktestArtifacts:
        if len(daily_bars) < self.config.train_window + self.config.test_window:
            raise ValueError(
                "Need at least train_window + test_window bars, "
                f"got {len(daily_bars)} for {symbol}"
            )
        features = self.feature_engine.build_daily_features(daily_bars)
        if len(features) < self.config.train_window + self.config.test_window:
            raise ValueError(
                f"Feature frame too short after warmup: {len(features)} rows"
            )

        equity_rows: list[dict] = []
        trade_rows: list[dict] = []
        regime_rows: list[dict] = []

        cash = self.config.initial_capital
        shares = 0.0

        start = 0
        while start + self.config.train_window + self.config.test_window <= len(features):
            train_slice = features.iloc[start : start + self.config.train_window]
            test_slice = features.iloc[start + self.config.train_window : start + self.config.train_window + self.config.test_window]
            price_slice = daily_bars.loc[test_slice.index]

            hmm = VolatilityRegimeHMM(
                n_candidates=self.config.hmm_n_candidates,
                n_init=self.config.hmm_n_init,
                covariance_type=self.config.hmm_covariance,
                random_state=self.config.random_state,
            )
            hmm.fit(train_slice)
            orchestrator = StrategyOrchestrator(config=self.strategy_config, regime_infos=hmm.regimes)

            # Run filtered inference over the entire history (train+test) but only
            # act on test-slice rows. This matches the live-trading contract.
            all_features = features.iloc[: start + self.config.train_window + self.config.test_window]
            proba = hmm.filtered_probabilities(all_features).iloc[-len(test_slice):]

            for date, proba_row in proba.iterrows():
                regime_id = int(proba_row.to_numpy().argmax())
                regime_prob = float(proba_row.max())
                price = float(price_slice.loc[date, "close"])
                regime_info = next(r for r in hmm.regimes if r.regime_id == regime_id)
                strategy = orchestrator.strategy_for(regime_id)
                signal = strategy.generate_signal(
                    symbol=symbol,
                    bars=daily_bars.loc[:date].tail(260),
                    regime_info=regime_info,
                    regime_probability=regime_prob,
                )
                target_alloc = signal.target_allocation_pct * signal.leverage if signal else 0.0
                equity = cash + shares * price
                current_alloc = shares * price / equity if equity > 0 else 0.0

                rebalance = abs(target_alloc - current_alloc) >= self.config.rebalance_threshold
                if rebalance:
                    target_shares = int(equity * target_alloc / price) if price > 0 else 0
                    delta = target_shares - shares
                    if delta != 0:
                        side = "BUY" if delta > 0 else "SELL"
                        exec_price = self.config.slippage.apply(
                            reference_price=price,
                            side=side,
                            realized_vol=float(train_slice["realized_vol_20"].tail(20).abs().mean()),
                        )
                        cash -= delta * exec_price
                        shares = float(target_shares)
                        trade_rows.append(
                            {
                                "date": date,
                                "symbol": symbol,
                                "side": side,
                                "qty": abs(delta),
                                "exec_price": exec_price,
                                "regime_id": regime_id,
                                "regime_name": regime_info.regime_name,
                                "regime_probability": regime_prob,
                                "target_alloc": target_alloc,
                            }
                        )

                equity = cash + shares * price
                equity_rows.append({"date": date, "equity": equity, "cash": cash, "shares": shares, "price": price})
                regime_rows.append(
                    {
                        "date": date,
                        "regime_id": regime_id,
                        "regime_name": regime_info.regime_name,
                        "regime_probability": regime_prob,
                        "model_version": hmm.metadata.model_version,
                    }
                )

            start += self.config.step_size

        equity_df = pd.DataFrame(equity_rows).set_index("date") if equity_rows else pd.DataFrame(columns=["equity", "cash", "shares", "price"])
        trade_df = pd.DataFrame(trade_rows)
        regime_df = pd.DataFrame(regime_rows).set_index("date") if regime_rows else pd.DataFrame()

        if equity_df.empty:
            raise RuntimeError("Backtest produced no rows")

        benchmark = _buy_and_hold_benchmark(symbol, daily_bars.loc[equity_df.index], self.config.initial_capital)
        comparison = pd.concat(
            {
                "strategy": equity_df["equity"],
                "buy_and_hold": benchmark,
            },
            axis=1,
        )
        return BacktestArtifacts(
            equity_curve=equity_df,
            trade_log=trade_df,
            regime_history=regime_df,
            benchmark_comparison=comparison,
        )

    def simulate_intraday_execution(
        self,
        *,
        symbol: str,
        intraday_bars: pd.DataFrame,
        target_allocation: float,
        capital: float,
    ) -> float:
        if intraday_bars.empty:
            return capital
        price = float(intraday_bars["close"].iloc[-1])
        exec_price = self.config.slippage.apply(reference_price=price, side="BUY")
        notional = capital * target_allocation
        shares = notional / exec_price
        return capital - shares * exec_price + shares * price


def _buy_and_hold_benchmark(symbol: str, bars: pd.DataFrame, initial_capital: float) -> pd.Series:
    if bars.empty:
        return pd.Series(dtype=float)
    price = bars["close"]
    shares = initial_capital / float(price.iloc[0])
    return shares * price


def run_backtest_cli(cfg) -> int:  # pragma: no cover - convenience only
    """Entrypoint used by ``main.py --backtest``."""
    symbols = cfg.get("broker.symbols", ["SPY"])
    if not symbols:
        LOG.error("No symbols configured for backtest")
        return 1
    symbol = symbols[0]
    from data.market_data import build_provider, MarketDataManager
    from monitoring.application import build_data_config

    state_dir = Path(cfg.get("platform.state_dir", "state"))
    if not state_dir.is_absolute() and cfg.source_path is not None:
        state_dir = Path(cfg.source_path).parent.parent / state_dir
    provider = build_provider(build_data_config(cfg, state_dir))
    manager = MarketDataManager(provider=provider)
    bars = manager.fetch_historical_daily_bars(symbol, lookback_bars=2000)
    backtest_cfg = BacktestConfig.from_settings(
        backtest_cfg=cfg.section("backtest"),
        strategy_cfg=cfg.section("strategy"),
        hmm_cfg=cfg.section("hmm"),
    )
    strategy_cfg = StrategyConfig.from_config(cfg.section("strategy"))
    backtester = WalkForwardBacktester(config=backtest_cfg, strategy_config=strategy_cfg)
    artifacts = backtester.run_walk_forward(symbol=symbol, daily_bars=bars)
    out_dir = Path("state/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts.equity_curve.to_csv(out_dir / "equity_curve.csv")
    artifacts.trade_log.to_csv(out_dir / "trade_log.csv", index=False)
    artifacts.regime_history.to_csv(out_dir / "regime_history.csv")
    artifacts.benchmark_comparison.to_csv(out_dir / "benchmark_comparison.csv")
    LOG.info("Backtest artifacts written to %s", out_dir)
    return 0
