"""Phase C9 - agent behaviour under model promotion, rollback, and fallback."""
from __future__ import annotations

from integrations.openclaw.interpreters import (
    handle_model_rollback_event,
    interpret_active_model_status,
    respect_unpromoted_candidate_model,
)


def test_no_active_model_is_flagged_unsafe() -> None:
    payload = {
        "active_model_version": None,
        "fallback_model_version": None,
        "candidates": [],
    }
    interp = interpret_active_model_status(payload)
    assert interp.active_version is None
    assert interp.reason == "no_active_model"
    assert respect_unpromoted_candidate_model(payload) is False


def test_active_model_is_surfaced_with_candidates() -> None:
    payload = {
        "active_model_version": "hmm-v2",
        "fallback_model_version": "hmm-v1",
        "candidates": [
            {"model_version": "hmm-v1", "status": "fallback"},
            {"model_version": "hmm-v2", "status": "active"},
            {"model_version": "hmm-v3", "status": "candidate"},
        ],
    }
    interp = interpret_active_model_status(payload)
    assert interp.active_version == "hmm-v2"
    assert interp.fallback_version == "hmm-v1"
    assert "hmm-v3" in interp.candidate_versions
    assert interp.has_unpromoted_candidate is True
    assert respect_unpromoted_candidate_model(payload) is True


def test_agent_never_treats_unpromoted_candidate_as_active() -> None:
    """An unpromoted candidate must never be used as if it were the active model."""
    payload = {
        "active_model_version": "hmm-v1",
        "fallback_model_version": None,
        "candidates": [
            {"model_version": "hmm-v1", "status": "active"},
            {"model_version": "hmm-v2", "status": "candidate"},
        ],
    }
    interp = interpret_active_model_status(payload)
    # Active is still v1 despite v2 being a newer candidate.
    assert interp.active_version == "hmm-v1"
    assert interp.has_unpromoted_candidate is True


def test_rollback_event_detects_change() -> None:
    previous = {"active_model_version": "hmm-v2"}
    current = {"active_model_version": "hmm-v1"}
    event = handle_model_rollback_event(previous, current)
    assert event["rolled_back"] is True
    assert event["previous"] == "hmm-v2"
    assert event["current"] == "hmm-v1"
    # The agent must discard prior reasoning tied to the old active model.
    assert event["reuse_prior_reasoning"] is False


def test_rollback_event_ignores_unchanged_active() -> None:
    snapshot = {"active_model_version": "hmm-v1"}
    event = handle_model_rollback_event(snapshot, snapshot)
    assert event["rolled_back"] is False


def test_on_fallback_is_reported() -> None:
    payload = {
        "active_model_version": "hmm-v1",
        "fallback_model_version": "hmm-v1",
        "candidates": [
            {"model_version": "hmm-v1", "status": "fallback"},
        ],
    }
    interp = interpret_active_model_status(payload)
    assert interp.active_is_fallback is True
    assert interp.reason == "on_fallback"
