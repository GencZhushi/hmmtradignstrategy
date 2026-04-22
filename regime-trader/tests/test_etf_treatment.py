"""Phase A12 - ETF sector/risk treatment must be explicit and auditable.

The classifier is the single source of truth for sector and ETF-risk buckets.
Tests verify:

- broad-market ETFs and sector ETFs return the right buckets by default
- single-name stocks outside the seed map fall back to the documented
  ``Unclassified`` / ``single_name`` sentinels (no silent misclassification)
- runtime overrides (from config) update the sector mapping deterministically
"""
from __future__ import annotations

import pytest

from core.sector_mapping import (
    DEFAULT_ETF_BUCKETS,
    DEFAULT_SECTORS,
    UNKNOWN_ETF_BUCKET,
    UNKNOWN_SECTOR,
    SectorClassifier,
    symbols_in_bucket,
)


def test_default_broad_etfs_map_to_broad_market_sector() -> None:
    cls = SectorClassifier()
    for sym in ("SPY", "QQQ", "VTI"):
        assert cls.get_sector_bucket(sym) == "BroadMarketETF"
        assert cls.get_etf_risk_bucket(sym) == "broad_beta"
        assert cls.is_broad_etf(sym) is True


def test_sector_etfs_map_to_named_sectors() -> None:
    cls = SectorClassifier()
    assert cls.get_sector_bucket("XLK") == "Technology"
    assert cls.get_sector_bucket("XLF") == "Financials"
    assert cls.get_sector_bucket("XLE") == "Energy"
    # Sector ETFs carry the ``sector_etf`` ETF-risk bucket.
    assert cls.get_etf_risk_bucket("XLK") == "sector_etf"
    assert cls.is_broad_etf("XLK") is False


def test_mega_cap_tech_mapped_to_technology() -> None:
    cls = SectorClassifier()
    for sym in ("AAPL", "MSFT", "NVDA", "AMD"):
        assert cls.get_sector_bucket(sym) == "Technology"
    # Single-name stocks default to ``single_name`` ETF bucket (no ETF overlap).
    assert cls.get_etf_risk_bucket("AAPL") == UNKNOWN_ETF_BUCKET


def test_communication_services_vs_consumer_cyclical() -> None:
    cls = SectorClassifier()
    assert cls.get_sector_bucket("GOOGL") == "CommunicationServices"
    assert cls.get_sector_bucket("META") == "CommunicationServices"
    assert cls.get_sector_bucket("TSLA") == "ConsumerCyclical"
    assert cls.get_sector_bucket("AMZN") == "ConsumerCyclical"


def test_unknown_symbol_falls_back_to_sentinel_buckets() -> None:
    cls = SectorClassifier()
    assert cls.get_sector_bucket("ZZZZ") == UNKNOWN_SECTOR
    assert cls.get_etf_risk_bucket("ZZZZ") == UNKNOWN_ETF_BUCKET


def test_lowercase_symbol_is_normalised() -> None:
    cls = SectorClassifier()
    assert cls.get_sector_bucket("aapl") == "Technology"
    assert cls.get_etf_risk_bucket("xlk") == "sector_etf"


def test_apply_overrides_updates_sector_mapping() -> None:
    cls = SectorClassifier()
    cls.apply_overrides({"NEW": "Energy"})
    assert cls.get_sector_bucket("NEW") == "Energy"


def test_project_post_trade_exposure_combines_current_and_delta() -> None:
    cls = SectorClassifier()
    current = {"Technology": 0.10}
    delta = {"MSFT": 0.05, "AAPL": 0.03, "XLE": 0.02}
    projected = cls.project_post_trade_exposure(current_exposure=current, delta=delta)
    assert projected["Technology"] == pytest.approx(0.10 + 0.05 + 0.03)
    assert projected["Energy"] == pytest.approx(0.02)


def test_symbols_in_bucket_returns_only_matching_symbols() -> None:
    cls = SectorClassifier()
    assert set(symbols_in_bucket(cls, ["AAPL", "XLE", "MSFT"], "Technology")) == {"AAPL", "MSFT"}


def test_default_map_has_matching_etf_keys() -> None:
    # Every ETF-risk bucket entry must have a corresponding sector bucket so
    # governance audits never encounter half-classified symbols.
    for symbol in DEFAULT_ETF_BUCKETS:
        assert symbol in DEFAULT_SECTORS
