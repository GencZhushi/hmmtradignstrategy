"""Market data / freshness routes (Phase B2 + B8)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import get_service
from api.services import PlatformService

router = APIRouter(tags=["market"], prefix="/market")


@router.get("/symbols")
def list_symbols(service: PlatformService = Depends(get_service)) -> dict:
    return {"symbols": service.config.get("broker.symbols", [])}


@router.get("/bars/daily")
def daily_bars(symbol: str = Query(...), lookback: int = Query(120, ge=20, le=1500), service: PlatformService = Depends(get_service)) -> dict:
    try:
        bars = service.application.market_data.fetch_historical_daily_bars(symbol, lookback_bars=lookback)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"daily bars unavailable: {exc}") from exc
    if bars.empty:
        return {"symbol": symbol, "bars": []}
    return {
        "symbol": symbol,
        "bars": [
            {"timestamp": ts.isoformat(), **row._asdict()} for ts, row in zip(bars.index, bars.itertuples(index=False))
        ],
    }
