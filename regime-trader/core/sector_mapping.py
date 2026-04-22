"""Deterministic sector classification and ETF risk handling (Phase A12).

The mapping is intentionally small, explicit, and auditable. It is the single
source of truth used by both the risk manager and the API/agent surface so
concentration decisions are consistent end to end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


DEFAULT_SECTORS: dict[str, str] = {
    # Mega-cap tech
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "CommunicationServices",
    "META": "CommunicationServices",
    "NVDA": "Technology",
    "AMD": "Technology",
    "TSLA": "ConsumerCyclical",
    "AMZN": "ConsumerCyclical",
    # Broad ETFs
    "SPY": "BroadMarketETF",
    "QQQ": "BroadMarketETF",
    "VTI": "BroadMarketETF",
    "IWM": "BroadMarketETF",
    # Sector ETFs
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "ConsumerCyclical",
    "XLP": "ConsumerDefensive",
    "XLRE": "RealEstate",
    "XLB": "Materials",
}


DEFAULT_ETF_BUCKETS: dict[str, str] = {
    "SPY": "broad_beta",
    "QQQ": "broad_beta",
    "VTI": "broad_beta",
    "IWM": "small_cap_beta",
    "XLK": "sector_etf",
    "XLF": "sector_etf",
    "XLE": "sector_etf",
    "XLV": "sector_etf",
    "XLI": "sector_etf",
    "XLY": "sector_etf",
    "XLP": "sector_etf",
    "XLRE": "sector_etf",
    "XLB": "sector_etf",
}


UNKNOWN_SECTOR = "Unclassified"
UNKNOWN_ETF_BUCKET = "single_name"


@dataclass
class SectorClassifier:
    """Deterministic classifier with explicit overrides from config."""

    sectors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SECTORS))
    etf_buckets: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ETF_BUCKETS))

    def get_sector_bucket(self, symbol: str) -> str:
        return self.sectors.get(symbol.upper(), UNKNOWN_SECTOR)

    def get_etf_risk_bucket(self, symbol: str) -> str:
        return self.etf_buckets.get(symbol.upper(), UNKNOWN_ETF_BUCKET)

    def is_broad_etf(self, symbol: str) -> bool:
        return self.get_etf_risk_bucket(symbol) == "broad_beta"

    def project_post_trade_exposure(
        self,
        *,
        current_exposure: Mapping[str, float],
        delta: Mapping[str, float],
    ) -> dict[str, float]:
        """Sector weights after applying a prospective set of allocation changes."""
        projected = {k: float(v) for k, v in current_exposure.items()}
        for symbol, change in delta.items():
            bucket = self.get_sector_bucket(symbol)
            projected[bucket] = projected.get(bucket, 0.0) + float(change)
        return projected

    def apply_overrides(self, overrides: Mapping[str, str]) -> None:
        for sym, bucket in overrides.items():
            self.sectors[sym.upper()] = bucket


def symbols_in_bucket(classifier: SectorClassifier, symbols: Iterable[str], bucket: str) -> list[str]:
    return [s for s in symbols if classifier.get_sector_bucket(s) == bucket]
