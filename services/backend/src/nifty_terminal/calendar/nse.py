"""Explicit NSE cash-session rules for deterministic candle boundaries.

The MVP uses a fixed UTC+05:30 offset because NSE does not observe daylight
saving time. Holidays and special sessions are injected; the code never guesses
them or treats a closed market as missing data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import StrEnum
from typing import Mapping

from nifty_terminal.domain.candle import Timeframe


IST = timezone(timedelta(hours=5, minutes=30), name="Asia/Kolkata")
UTC = timezone.utc


class SessionKind(StrEnum):
    REGULAR = "REGULAR"
    SPECIAL = "SPECIAL"


class MarketPhase(StrEnum):
    CLOSED = "CLOSED"
    CONTINUOUS_TRADING = "CONTINUOUS_TRADING"
    CLOSING_AUCTION_REFERENCE = "CLOSING_AUCTION_REFERENCE"
    CLOSING_AUCTION_ORDER_ENTRY = "CLOSING_AUCTION_ORDER_ENTRY"
    CLOSING_AUCTION_MATCHING = "CLOSING_AUCTION_MATCHING"

    @property
    def is_closing_auction(self) -> bool:
        return self in {
            MarketPhase.CLOSING_AUCTION_REFERENCE,
            MarketPhase.CLOSING_AUCTION_ORDER_ENTRY,
            MarketPhase.CLOSING_AUCTION_MATCHING,
        }


@dataclass(frozen=True, slots=True)
class ContinuousSessionRule:
    effective_from: date
    opens: time
    closes: time


@dataclass(frozen=True, slots=True)
class ClosingAuctionRule:
    effective_from: date
    reference_starts: time
    order_entry_starts: time
    matching_starts: time
    matching_ends: time


DEFAULT_CONTINUOUS_SESSION_RULES = (
    ContinuousSessionRule(date.min, time(9, 15), time(15, 30)),
    ContinuousSessionRule(date(2026, 8, 3), time(9, 15), time(15, 15)),
)

DEFAULT_CLOSING_AUCTION_RULES = (
    ClosingAuctionRule(
        date(2026, 8, 3),
        time(15, 15),
        time(15, 20),
        time(15, 30),
        time(15, 35),
    ),
)


@dataclass(frozen=True, slots=True)
class TradingSession:
    session_date: date
    opens_at: datetime
    closes_at: datetime
    kind: SessionKind


@dataclass(frozen=True, slots=True)
class CandleBucket:
    timeframe: Timeframe
    opens_at: datetime
    closes_at: datetime
    expected_minutes: int
    is_partial: bool


class NseSessionCalendar:
    """Session-aware bucket calculation for the NSE cash market."""

    def __init__(
        self,
        *,
        holidays: frozenset[date] = frozenset(),
        special_sessions: Mapping[date, tuple[time, time]] | None = None,
        continuous_session_rules: tuple[ContinuousSessionRule, ...] | None = None,
        closing_auction_rules: tuple[ClosingAuctionRule, ...] | None = None,
    ) -> None:
        self._holidays = holidays
        self._special_sessions = dict(special_sessions or {})
        self._continuous_session_rules = _ordered_rules(
            continuous_session_rules or DEFAULT_CONTINUOUS_SESSION_RULES
        )
        self._closing_auction_rules = _ordered_rules(
            closing_auction_rules or DEFAULT_CLOSING_AUCTION_RULES
        )
        _validate_session_rules(
            self._continuous_session_rules,
            self._closing_auction_rules,
        )

    def session_for_date(self, session_date: date) -> TradingSession | None:
        special = self._special_sessions.get(session_date)
        if special is not None:
            opens, closes = special
            return TradingSession(
                session_date=session_date,
                opens_at=datetime.combine(session_date, opens, IST),
                closes_at=datetime.combine(session_date, closes, IST),
                kind=SessionKind.SPECIAL,
            )
        if session_date.weekday() >= 5 or session_date in self._holidays:
            return None
        rule = _effective_rule(self._continuous_session_rules, session_date)
        if rule is None:
            raise ValueError("No continuous-session rule covers this date")
        return TradingSession(
            session_date=session_date,
            opens_at=datetime.combine(session_date, rule.opens, IST),
            closes_at=datetime.combine(session_date, rule.closes, IST),
            kind=SessionKind.REGULAR,
        )

    def market_phase(self, instant: datetime) -> MarketPhase:
        aware = _require_aware(instant)
        local = aware.astimezone(IST)
        session = self.session_for_date(local.date())
        if session is None:
            return MarketPhase.CLOSED
        if session.opens_at <= local < session.closes_at:
            return MarketPhase.CONTINUOUS_TRADING
        rule = _effective_rule(self._closing_auction_rules, local.date())
        if rule is None or session.kind is SessionKind.SPECIAL:
            return MarketPhase.CLOSED
        reference = datetime.combine(local.date(), rule.reference_starts, IST)
        order_entry = datetime.combine(local.date(), rule.order_entry_starts, IST)
        matching = datetime.combine(local.date(), rule.matching_starts, IST)
        matching_end = datetime.combine(local.date(), rule.matching_ends, IST)
        if reference <= local < order_entry:
            return MarketPhase.CLOSING_AUCTION_REFERENCE
        if order_entry <= local < matching:
            return MarketPhase.CLOSING_AUCTION_ORDER_ENTRY
        if matching <= local < matching_end:
            return MarketPhase.CLOSING_AUCTION_MATCHING
        return MarketPhase.CLOSED

    def session_containing(self, instant: datetime) -> TradingSession | None:
        aware = _require_aware(instant)
        local = aware.astimezone(IST)
        session = self.session_for_date(local.date())
        if session is None or not (session.opens_at <= local < session.closes_at):
            return None
        return session

    def bucket_for(self, instant: datetime, timeframe: Timeframe) -> CandleBucket:
        aware = _require_aware(instant)
        session = self.session_containing(aware)
        if session is None:
            raise ValueError("Timestamp is outside an explicitly known NSE session")

        local = aware.astimezone(IST)
        elapsed_minutes = int((local - session.opens_at).total_seconds() // 60)
        bucket_index = elapsed_minutes // timeframe.minutes
        opens_local = session.opens_at + timedelta(minutes=bucket_index * timeframe.minutes)
        nominal_close = opens_local + timedelta(minutes=timeframe.minutes)
        closes_local = min(nominal_close, session.closes_at)
        expected_minutes = int((closes_local - opens_local).total_seconds() // 60)

        return CandleBucket(
            timeframe=timeframe,
            opens_at=opens_local.astimezone(UTC),
            closes_at=closes_local.astimezone(UTC),
            expected_minutes=expected_minutes,
            is_partial=expected_minutes != timeframe.minutes,
        )

    def expected_minute_opens(self, bucket: CandleBucket) -> tuple[datetime, ...]:
        return tuple(
            bucket.opens_at + timedelta(minutes=index)
            for index in range(bucket.expected_minutes)
        )


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must be timezone-aware")
    return value


def _ordered_rules(rules: tuple[object, ...]) -> tuple[object, ...]:
    if not rules:
        return ()
    ordered = tuple(sorted(rules, key=lambda item: item.effective_from))
    if len({item.effective_from for item in ordered}) != len(ordered):
        raise ValueError("Effective session-rule dates must be unique")
    return ordered


def _effective_rule(rules: tuple[object, ...], session_date: date):
    selected = None
    for rule in rules:
        if rule.effective_from > session_date:
            break
        selected = rule
    return selected


def _validate_session_rules(
    continuous_rules: tuple[object, ...],
    auction_rules: tuple[object, ...],
) -> None:
    if not continuous_rules:
        raise ValueError("At least one continuous-session rule is required")
    for rule in continuous_rules:
        if rule.opens >= rule.closes:
            raise ValueError("Continuous-session rule must open before it closes")
    for rule in auction_rules:
        if not (
            rule.reference_starts
            < rule.order_entry_starts
            < rule.matching_starts
            < rule.matching_ends
        ):
            raise ValueError("Closing-auction phase times must be strictly increasing")
        continuous = _effective_rule(continuous_rules, rule.effective_from)
        if continuous is None or continuous.closes != rule.reference_starts:
            raise ValueError(
                "Closing auction must start when continuous trading closes"
            )
