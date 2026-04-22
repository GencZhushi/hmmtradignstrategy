"""Phase A12 - joint multi-trade sector breach resolution.

When several candidate trades *individually* satisfy the single-name cap but
*jointly* breach a sector limit, the engine must scale contributing symbols
down proportionally. The resolution is deterministic so callers (risk manager,
agent, UI) observe identical allocations for identical inputs.
"""
from __future__ import annotations

import pytest

from core.correlation_risk import project_joint_breach, resolve_joint_breach
from core.sector_mapping import SectorClassifier


def _classifier() -> SectorClassifier:
    return SectorClassifier(
        sectors={
            "A": "Tech",
            "B": "Tech",
            "C": "Tech",
            "D": "Energy",
            "E": "Energy",
        }
    )


def test_project_joint_breach_flags_sectors_over_limit() -> None:
    cls = _classifier()
    breaches = project_joint_breach(
        candidate_allocations={"A": 0.10, "B": 0.15, "D": 0.05},
        current_sector_exposure={"Tech": 0.0, "Energy": 0.0},
        sector_limit=0.20,
        sector_of=cls.get_sector_bucket,
    )
    # Tech projected = 0.25 -> breach. Energy = 0.05 -> no breach.
    assert set(breaches) == {"Tech"}
    assert breaches["Tech"] == pytest.approx(0.25)


def test_project_joint_breach_includes_existing_exposure() -> None:
    cls = _classifier()
    breaches = project_joint_breach(
        candidate_allocations={"A": 0.05},
        current_sector_exposure={"Tech": 0.18},
        sector_limit=0.20,
        sector_of=cls.get_sector_bucket,
    )
    assert breaches["Tech"] == pytest.approx(0.23)


def test_resolve_joint_breach_scales_contributors_proportionally() -> None:
    cls = _classifier()
    candidate = {"A": 0.10, "B": 0.10, "D": 0.05}
    scaled = resolve_joint_breach(
        candidate_allocations=candidate,
        breaches={"Tech": 0.30},
        sector_limit=0.20,
        sector_of=cls.get_sector_bucket,
    )
    # Tech contributors (A+B) must sum to the sector limit post scaling.
    assert scaled["A"] + scaled["B"] == pytest.approx(0.20)
    # Contributors scale by the same factor -> equal post-scale allocations.
    assert scaled["A"] == pytest.approx(scaled["B"])
    # Non-breaching sector (Energy) is left untouched.
    assert scaled["D"] == pytest.approx(0.05)


def test_resolve_joint_breach_is_deterministic() -> None:
    cls = _classifier()
    candidate = {"A": 0.12, "B": 0.08, "C": 0.05, "D": 0.04}
    breaches = {"Tech": 0.25}
    first = resolve_joint_breach(
        candidate_allocations=candidate,
        breaches=breaches,
        sector_limit=0.15,
        sector_of=cls.get_sector_bucket,
    )
    second = resolve_joint_breach(
        candidate_allocations=candidate,
        breaches=breaches,
        sector_limit=0.15,
        sector_of=cls.get_sector_bucket,
    )
    assert first == second


def test_resolve_joint_breach_no_breaches_returns_unchanged() -> None:
    cls = _classifier()
    candidate = {"A": 0.05, "D": 0.05}
    result = resolve_joint_breach(
        candidate_allocations=candidate,
        breaches={},
        sector_limit=0.20,
        sector_of=cls.get_sector_bucket,
    )
    assert result == {"A": 0.05, "D": 0.05}


def test_resolve_joint_breach_handles_multiple_sectors() -> None:
    cls = _classifier()
    candidate = {"A": 0.12, "B": 0.08, "D": 0.15, "E": 0.10}
    scaled = resolve_joint_breach(
        candidate_allocations=candidate,
        breaches={"Tech": 0.20, "Energy": 0.25},
        sector_limit=0.15,
        sector_of=cls.get_sector_bucket,
    )
    assert scaled["A"] + scaled["B"] == pytest.approx(0.15)
    assert scaled["D"] + scaled["E"] == pytest.approx(0.15)


def test_resolve_joint_breach_zero_contribution_is_left_alone() -> None:
    cls = _classifier()
    # All candidate Tech allocations are zero so no reduction possible.
    candidate = {"A": 0.0, "B": 0.0, "D": 0.05}
    scaled = resolve_joint_breach(
        candidate_allocations=candidate,
        breaches={"Tech": 0.30},
        sector_limit=0.20,
        sector_of=cls.get_sector_bucket,
    )
    assert scaled == {"A": 0.0, "B": 0.0, "D": 0.05}
