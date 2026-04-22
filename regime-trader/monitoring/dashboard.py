"""Terminal dashboard rendering (Phase A8).

Uses ``rich`` when available; falls back to plain text so the engine still runs
in environments where the dependency is not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


@dataclass
class DashboardSnapshot:
    """View-model passed to the dashboard on every refresh."""

    regime_name: str | None
    regime_probability: float
    stability_bars: int
    flicker_rate: float
    equity: float
    daily_pnl: float
    allocation: float
    leverage: float
    positions: list[Mapping[str, Any]]
    recent_signals: list[Mapping[str, Any]]
    breaker_state: str
    last_update: datetime


def build_snapshot(
    *,
    regime_state: Mapping[str, Any] | None,
    portfolio: Mapping[str, Any],
    positions: Iterable[Mapping[str, Any]],
    recent_signals: Iterable[Mapping[str, Any]],
    breaker_state: str = "clear",
) -> DashboardSnapshot:
    regime_name = None
    probability = 0.0
    stability = 0
    flicker = 0.0
    if regime_state:
        regime_name = regime_state.get("regime_name")
        probability = float(regime_state.get("probability", 0.0))
        stability = int(regime_state.get("consecutive_bars", 0))
        flicker = float(regime_state.get("flicker_rate", 0.0))
    return DashboardSnapshot(
        regime_name=regime_name,
        regime_probability=probability,
        stability_bars=stability,
        flicker_rate=flicker,
        equity=float(portfolio.get("equity", 0.0)),
        daily_pnl=float(portfolio.get("daily_pnl", 0.0)),
        allocation=float(portfolio.get("allocation", 0.0)),
        leverage=float(portfolio.get("leverage", 1.0)),
        positions=list(positions),
        recent_signals=list(recent_signals),
        breaker_state=breaker_state,
        last_update=datetime.now(timezone.utc),
    )


def render_plain(snapshot: DashboardSnapshot) -> str:
    lines = [
        f"=== Regime Trader ({snapshot.last_update.isoformat()}) ===",
        f"Regime: {snapshot.regime_name or 'n/a'} (p={snapshot.regime_probability:.2f}) "
        f"stability={snapshot.stability_bars} flicker={snapshot.flicker_rate:.2f}",
        f"Equity: {snapshot.equity:,.2f} | Daily PnL: {snapshot.daily_pnl:,.2f}",
        f"Allocation: {snapshot.allocation:.0%} | Leverage: {snapshot.leverage:.2f}x",
        f"Breaker: {snapshot.breaker_state}",
        "Positions:",
    ]
    for pos in snapshot.positions[:10]:
        lines.append(
            f"  {pos.get('symbol')} qty={pos.get('quantity')} entry={pos.get('avg_entry_price')} "
            f"stop={pos.get('stop_price')}"
        )
    lines.append("Recent signals:")
    for sig in snapshot.recent_signals[:10]:
        lines.append(
            f"  {sig.get('timestamp', '')} {sig.get('symbol', '')} "
            f"{sig.get('direction', '')} {sig.get('target_allocation_pct', 0):.0%} "
            f"({sig.get('strategy_name', '')})"
        )
    return "\n".join(lines)


def render(snapshot: DashboardSnapshot) -> str:
    try:
        from rich.console import Console  # type: ignore
        from rich.panel import Panel  # type: ignore
        from rich.table import Table  # type: ignore
    except ImportError:
        return render_plain(snapshot)

    console = Console(record=True)
    header = (
        f"[bold]{snapshot.regime_name or 'UNKNOWN'}[/bold] "
        f"(p={snapshot.regime_probability:.2f}) | stability={snapshot.stability_bars} "
        f"flicker={snapshot.flicker_rate:.2f}"
    )
    portfolio = (
        f"Equity: [green]{snapshot.equity:,.2f}[/green] | "
        f"Daily: {snapshot.daily_pnl:+,.2f} | "
        f"Alloc: {snapshot.allocation:.0%} | Leverage: {snapshot.leverage:.2f}x"
    )
    console.print(Panel(f"{header}\n{portfolio}\nBreaker: {snapshot.breaker_state}", title="Regime Trader"))
    if snapshot.positions:
        table = Table(title="Positions")
        for col in ("Symbol", "Qty", "Entry", "Stop", "Regime"):
            table.add_column(col)
        for pos in snapshot.positions[:10]:
            table.add_row(
                str(pos.get("symbol", "")),
                f"{pos.get('quantity', 0):.2f}",
                f"{pos.get('avg_entry_price', 0):.2f}",
                str(pos.get("stop_price", "-")),
                str(pos.get("regime_at_entry", "-")),
            )
        console.print(table)
    if snapshot.recent_signals:
        table = Table(title="Recent Signals")
        for col in ("When", "Symbol", "Dir", "Alloc", "Strategy"):
            table.add_column(col)
        for sig in snapshot.recent_signals[:10]:
            table.add_row(
                str(sig.get("timestamp", "")),
                str(sig.get("symbol", "")),
                str(sig.get("direction", "")),
                f"{float(sig.get('target_allocation_pct', 0)):.0%}",
                str(sig.get("strategy_name", "")),
            )
        console.print(table)
    return console.export_text()
