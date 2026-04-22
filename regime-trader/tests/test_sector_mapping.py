"""Phase A12 - sector classification + ETF bucket logic."""
from __future__ import annotations

from core.sector_mapping import (
    DEFAULT_ETF_BUCKETS,
    DEFAULT_SECTORS,
    SectorClassifier,
    UNKNOWN_ETF_BUCKET,
    UNKNOWN_SECTOR,
    symbols_in_bucket,
)


def test_default_sectors_cover_configured_symbols() -> None:
    for sym in ("SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "TSLA", "AMD"):
        assert DEFAULT_SECTORS.get(sym) is not None


def test_unknown_symbol_falls_back_cleanly() -> None:
    classifier = SectorClassifier()
    assert classifier.get_sector_bucket("ZZZZ") == UNKNOWN_SECTOR
    assert classifier.get_etf_risk_bucket("ZZZZ") == UNKNOWN_ETF_BUCKET


def test_etf_buckets_distinguish_broad_from_sector() -> None:
    classifier = SectorClassifier()
    assert classifier.is_broad_etf("SPY")
    assert classifier.get_etf_risk_bucket("XLK") == "sector_etf"


def test_overrides_apply_consistently() -> None:
    classifier = SectorClassifier()
    classifier.apply_overrides({"NVDA": "Semiconductors"})
    assert classifier.get_sector_bucket("NVDA") == "Semiconductors"


def test_project_post_trade_exposure_adds_delta() -> None:
    classifier = SectorClassifier()
    projected = classifier.project_post_trade_exposure(
        current_exposure={"Technology": 0.1},
        delta={"AAPL": 0.05, "NVDA": 0.03},
    )
    assert projected["Technology"] == 0.1 + 0.05 + 0.03


def test_symbols_in_bucket_filters_deterministically() -> None:
    classifier = SectorClassifier()
    tech = symbols_in_bucket(classifier, ["AAPL", "SPY", "MSFT", "XLE"], "Technology")
    assert tech == ["AAPL", "MSFT"]
