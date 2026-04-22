"""Natural-language command parser for OpenClaw (Spec C1).

The parser maps a small set of free-form phrases to structured tool invocations.
It only exists so legacy chat-style prompts are converted into tool calls before
they hit the platform. Anything it cannot parse returns a ``parse_error`` result
so the agent never fabricates an action.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedCommand:
    """Result of parsing a free-form agent request."""

    tool: str
    params: dict[str, Any]
    confidence: float = 1.0
    raw_text: str = ""
    warnings: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.warnings is None:
            self.warnings = []


_ALLOCATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_SYMBOL_RE = re.compile(r"\b([A-Z]{1,5})\b")


def parse_agent_request(text: str) -> ParsedCommand:
    """Best-effort parse; returns a ``parse_error`` tool for unknown inputs."""
    clean = text.strip()
    lower = clean.lower()
    if not clean:
        return ParsedCommand(tool="parse_error", params={"reason": "empty_input"}, confidence=0.0, raw_text=text)

    if re.search(r"\bcurrent\s+regime\b|\bwhat('?s|\s+is)?\s+the\s+regime\b", lower):
        return ParsedCommand(tool="get_regime", params={}, raw_text=text)

    if re.search(
        r"positions?\s+open\b|open\s+positions?\b|list\s+positions?|positions?\s+(?:are|that\s+are)\s+open|show\s+positions?",
        lower,
    ):
        return ParsedCommand(tool="get_positions", params={}, raw_text=text)

    if re.search(r"portfolio|pnl|equity|cash", lower):
        return ParsedCommand(tool="get_portfolio", params={}, raw_text=text)

    if re.search(r"risk\s+status|breaker|circuit\s+breaker", lower):
        return ParsedCommand(tool="get_risk_status", params={}, raw_text=text)

    if re.search(r"pending\s+approvals?|awaiting\s+approval", lower):
        return ParsedCommand(tool="get_pending_approvals", params={}, raw_text=text)

    if re.search(r"data\s+(fresh(ness)?|stale)|market\s+session", lower):
        return ParsedCommand(tool="get_freshness", params={}, raw_text=text)

    if re.search(r"model\s+(version|governance|active)|which\s+model", lower):
        return ParsedCommand(tool="get_model_governance", params={}, raw_text=text)

    if re.search(r"close\s+all\b", lower):
        return ParsedCommand(tool="close_all_positions", params={}, raw_text=text)

    close_match = re.search(r"close\s+(?:my\s+)?(?:position\s+in\s+)?([A-Z]{1,5})\b", clean)
    if close_match:
        return ParsedCommand(tool="close_position", params={"symbol": close_match.group(1).upper()}, raw_text=text)

    if re.search(r"why\s+was\s+(my\s+)?(trade|intent|order)\s+rejected", lower):
        return ParsedCommand(tool="explain_rejection", params={}, confidence=0.5, raw_text=text, warnings=["intent_id missing"])

    # Preview / submit phrasing: "preview SPY 10%" or "buy 10% SPY"
    preview_match = re.search(r"preview\s+([A-Z]{1,5})\s+(?:a\s+)?(\d+(?:\.\d+)?)\s*%", clean, re.IGNORECASE)
    if preview_match:
        params = {
            "symbol": preview_match.group(1).upper(),
            "direction": "LONG",
            "allocation_pct": float(preview_match.group(2)) / 100,
        }
        return ParsedCommand(tool="preview_trade", params=params, raw_text=text)

    buy_match = re.search(r"(buy|allocate|go\s+long)\s+(\d+(?:\.\d+)?)\s*%\s+(?:of\s+)?([A-Z]{1,5})", clean, re.IGNORECASE)
    if buy_match:
        params = {
            "symbol": buy_match.group(3).upper(),
            "direction": "LONG",
            "allocation_pct": float(buy_match.group(2)) / 100,
        }
        return ParsedCommand(tool="submit_trade_intent", params=params, raw_text=text)

    return ParsedCommand(
        tool="parse_error",
        params={"reason": "unrecognized_command", "raw_text": text},
        confidence=0.0,
        raw_text=text,
    )
