"""Load explicit exchange calendar exceptions without guessing holidays."""

from __future__ import annotations

from datetime import date, time
import json
from pathlib import Path

from nifty_terminal.calendar.nse import (
    ClosingAuctionRule,
    ContinuousSessionRule,
    NseSessionCalendar,
)


def load_nse_calendar(path: Path | None) -> NseSessionCalendar:
    if path is None:
        return NseSessionCalendar()
    payload = json.loads(path.read_text(encoding="utf-8"))
    holidays = frozenset(date.fromisoformat(item) for item in payload.get("holidays", []))
    special_sessions = {
        date.fromisoformat(session_date): (
            time.fromisoformat(values["open"]),
            time.fromisoformat(values["close"]),
        )
        for session_date, values in payload.get("special_sessions", {}).items()
    }
    continuous_session_rules = tuple(
        ContinuousSessionRule(
            effective_from=date.fromisoformat(values["effective_from"]),
            opens=time.fromisoformat(values["open"]),
            closes=time.fromisoformat(values["close"]),
        )
        for values in payload.get("continuous_session_rules", [])
    )
    closing_auction_rules = tuple(
        ClosingAuctionRule(
            effective_from=date.fromisoformat(values["effective_from"]),
            reference_starts=time.fromisoformat(values["reference_starts"]),
            order_entry_starts=time.fromisoformat(values["order_entry_starts"]),
            matching_starts=time.fromisoformat(values["matching_starts"]),
            matching_ends=time.fromisoformat(values["matching_ends"]),
        )
        for values in payload.get("closing_auction_rules", [])
    )
    return NseSessionCalendar(
        holidays=holidays,
        special_sessions=special_sessions,
        continuous_session_rules=continuous_session_rules or None,
        closing_auction_rules=closing_auction_rules or None,
    )


def validate_calendar_coverage(
    metadata: object,
    *,
    starts_on: date,
    ends_on: date,
) -> None:
    """Require an explicitly sourced calendar covering the entire dataset."""

    if not isinstance(metadata, dict):
        raise ValueError("Calendar JSON must contain an object")
    if metadata.get("exchange") != "NSE" or metadata.get("segment") != "CAPITAL_MARKET":
        raise ValueError("Calendar must be verified for the NSE capital-market segment")
    try:
        verified_from = date.fromisoformat(str(metadata["verified_from"]))
        verified_through = date.fromisoformat(str(metadata["verified_through"]))
    except (KeyError, ValueError) as error:
        raise ValueError("Calendar must declare verified_from and verified_through") from error
    if starts_on < verified_from or ends_on > verified_through:
        raise ValueError(
            "Requested history exceeds the explicitly verified exchange calendar: "
            f"{verified_from} through {verified_through}"
        )
