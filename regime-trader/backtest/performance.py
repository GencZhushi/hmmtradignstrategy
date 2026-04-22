"""Performance metrics and benchmark comparisons (Spec A5 / Phase A5)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass
class PerformanceSummary:
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float


def compute_performance_metrics(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    *,
    risk_free_rate: float = 0.045,
) -> PerformanceSummary:
    if equity.empty:
        return PerformanceSummary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    returns = equity.pct_change().dropna()
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)
    mean = float(returns.mean() * TRADING_DAYS)
    std = float(returns.std(ddof=0) * np.sqrt(TRADING_DAYS))
    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=0) * np.sqrt(TRADING_DAYS)) if not downside.empty else 0.0
    sharpe = float((mean - risk_free_rate) / std) if std > 0 else 0.0
    sortino = float((mean - risk_free_rate) / downside_std) if downside_std > 0 else 0.0
    max_dd, dd_duration = _max_drawdown(equity)
    calmar = float(cagr / abs(max_dd)) if max_dd != 0 else 0.0

    trade_metrics = _trade_metrics(trades)
    return PerformanceSummary(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        max_drawdown_duration=dd_duration,
        win_rate=trade_metrics["win_rate"],
        profit_factor=trade_metrics["profit_factor"],
        total_trades=trade_metrics["total_trades"],
        avg_trade_pnl=trade_metrics["avg_trade_pnl"],
    )


def compare_to_benchmarks(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
) -> pd.DataFrame:
    strat = compute_performance_metrics(strategy_equity)
    bench = compute_performance_metrics(benchmark_equity)
    return pd.DataFrame(
        {
            "strategy": strat.__dict__,
            "buy_and_hold": bench.__dict__,
        }
    )


def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = float(dd.min()) if not dd.empty else 0.0
    idx = int(np.argmin(dd.to_numpy())) if not dd.empty else 0
    dd_duration = int((equity.index[-1] - equity.index[idx]).days) if len(equity) > 1 else 0
    return max_dd, dd_duration


def _trade_metrics(trades: pd.DataFrame | None) -> dict[str, float]:
    if trades is None or trades.empty:
        return {"win_rate": 0.0, "profit_factor": 0.0, "total_trades": 0, "avg_trade_pnl": 0.0}
    qty = trades.get("qty")
    exec_price = trades.get("exec_price")
    if qty is None or exec_price is None:
        return {"win_rate": 0.0, "profit_factor": 0.0, "total_trades": int(len(trades)), "avg_trade_pnl": 0.0}
    # Proxy P&L from alternating BUY/SELL rows; backtester emits allocation changes.
    trades = trades.reset_index(drop=True)
    pnls: list[float] = []
    position = 0.0
    avg = 0.0
    for _, row in trades.iterrows():
        side = str(row.get("side", "")).upper()
        q = float(row.get("qty", 0.0))
        price = float(row.get("exec_price", 0.0))
        if side == "BUY":
            new_position = position + q
            avg = (avg * position + price * q) / new_position if new_position > 0 else price
            position = new_position
        else:  # SELL
            qty_closed = min(q, position)
            pnls.append((price - avg) * qty_closed)
            position -= qty_closed
    if not pnls:
        return {"win_rate": 0.0, "profit_factor": 0.0, "total_trades": int(len(trades)), "avg_trade_pnl": 0.0}
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf") if wins else 0.0
    return {
        "win_rate": len(wins) / len(pnls),
        "profit_factor": float(profit_factor),
        "total_trades": int(len(trades)),
        "avg_trade_pnl": float(np.mean(pnls)),
    }


def regime_breakdown(
    equity: pd.Series,
    regime_history: pd.DataFrame,
) -> pd.DataFrame:
    if equity.empty or regime_history.empty:
        return pd.DataFrame()
    merged = equity.to_frame("equity").join(regime_history, how="inner")
    merged["returns"] = merged["equity"].pct_change()
    stats = (
        merged.dropna(subset=["returns"])
        .groupby("regime_name")["returns"]
        .agg(["count", "mean", "std"])
        .rename(columns={"count": "bars", "mean": "avg_return", "std": "volatility"})
    )
    stats["sharpe"] = stats["avg_return"] / stats["volatility"].replace(0, np.nan) * np.sqrt(TRADING_DAYS)
    stats["pct_time"] = stats["bars"] / len(merged)
    return stats.fillna(0.0)
