"""Calendar-authorized normalization and coverage diagnostics for research history."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta
from typing import Mapping

from nifty_terminal.calendar.nse import IST, NseSessionCalendar
from nifty_terminal.domain.candle import FinalizedMinuteBarInput, Timeframe
from nifty_terminal.history.models import HistoricalBatch
from nifty_terminal.history.sources.angelone_source import historical_rows_sha256


MINIMUM_EXPECTED_MINUTE_COVERAGE = 0.995
MAX_CONSECUTIVE_MISSING_MINUTES = 5
MAX_MISSING_MINUTES_PER_SESSION = 15


@dataclass(frozen=True, slots=True)
class SessionNormalizationResult:
    batch: HistoricalBatch
    raw_source_sha256: str
    raw_row_count: int
    excluded_row_count: int
    excluded_reasons: tuple[tuple[str, int], ...]
    excluded_timestamp_samples: tuple[str, ...]

    def to_contract(self) -> dict[str, object]:
        return {
            "raw_source_sha256": self.raw_source_sha256,
            "raw_row_count": self.raw_row_count,
            "retained_session_row_count": len(self.batch.rows),
            "excluded_row_count": self.excluded_row_count,
            "excluded_reasons": dict(self.excluded_reasons),
            "excluded_timestamp_samples": list(self.excluded_timestamp_samples),
        }


@dataclass(frozen=True, slots=True)
class CoverageDiagnostics:
    expected_minutes: int
    observed_expected_minutes: int
    missing_minutes: int
    coverage_ratio: float
    affected_sessions: int
    max_missing_minutes_in_one_session: int
    max_consecutive_missing_minutes: int
    worst_sessions: tuple[tuple[str, int], ...]
    missing_timestamp_samples: tuple[str, ...]

    @property
    def research_acceptable(self) -> bool:
        return (
            self.coverage_ratio >= MINIMUM_EXPECTED_MINUTE_COVERAGE
            and self.max_missing_minutes_in_one_session
            <= MAX_MISSING_MINUTES_PER_SESSION
            and self.max_consecutive_missing_minutes
            <= MAX_CONSECUTIVE_MISSING_MINUTES
        )

    def blockers(self) -> tuple[str, ...]:
        blockers: list[str] = []
        if self.coverage_ratio < MINIMUM_EXPECTED_MINUTE_COVERAGE:
            blockers.append("EXPECTED_MINUTE_COVERAGE_BELOW_99_5_PERCENT")
        if self.max_missing_minutes_in_one_session > MAX_MISSING_MINUTES_PER_SESSION:
            blockers.append("TOO_MANY_MISSING_MINUTES_IN_ONE_SESSION")
        if self.max_consecutive_missing_minutes > MAX_CONSECUTIVE_MISSING_MINUTES:
            blockers.append("CONSECUTIVE_MISSING_MINUTE_GAP_EXCEEDS_5")
        return tuple(blockers)

    def to_contract(self) -> dict[str, object]:
        return {
            "expected_minutes": self.expected_minutes,
            "observed_expected_minutes": self.observed_expected_minutes,
            "missing_minutes": self.missing_minutes,
            "coverage_ratio": self.coverage_ratio,
            "coverage_percent": round(self.coverage_ratio * 100, 6),
            "affected_sessions": self.affected_sessions,
            "max_missing_minutes_in_one_session": self.max_missing_minutes_in_one_session,
            "max_consecutive_missing_minutes": self.max_consecutive_missing_minutes,
            "worst_sessions": dict(self.worst_sessions),
            "missing_timestamp_samples": list(self.missing_timestamp_samples),
            "research_acceptable": self.research_acceptable,
            "blockers": list(self.blockers()),
            "policy": {
                "minimum_expected_minute_coverage": MINIMUM_EXPECTED_MINUTE_COVERAGE,
                "max_missing_minutes_per_session": MAX_MISSING_MINUTES_PER_SESSION,
                "max_consecutive_missing_minutes": MAX_CONSECUTIVE_MISSING_MINUTES,
                "missing_bars_are_filled": False,
            },
        }


def normalize_to_continuous_sessions(
    batch: HistoricalBatch,
    *,
    calendar: NseSessionCalendar,
    calendar_metadata: Mapping[str, object],
) -> SessionNormalizationResult:
    """Exclude only explicitly authorized non-trading provider observations."""

    ignored_windows = _ignored_windows(calendar_metadata)
    excluded_sessions = _research_excluded_sessions(calendar_metadata)
    retained: list[FinalizedMinuteBarInput] = []
    excluded: list[tuple[FinalizedMinuteBarInput, str]] = []
    unauthorized: list[FinalizedMinuteBarInput] = []

    for row in batch.rows:
        local_date = row.opens_at.astimezone(IST).date()
        excluded_reason = excluded_sessions.get(local_date)
        if excluded_reason is not None:
            excluded.append((row, excluded_reason))
            continue
        try:
            bucket = calendar.bucket_for(row.opens_at, Timeframe.M1)
            aligned = bucket.opens_at == row.opens_at and bucket.closes_at == row.closes_at
        except ValueError:
            aligned = False
        if aligned:
            retained.append(row)
            continue
        reason = _authorized_exclusion_reason(row, ignored_windows, calendar)
        if reason is None:
            unauthorized.append(row)
        else:
            excluded.append((row, reason))

    if unauthorized:
        samples = ", ".join(
            row.opens_at.astimezone(IST).isoformat() for row in unauthorized[:10]
        )
        raise ValueError(
            "Provider returned observations outside the verified continuous-trading "
            f"calendar ({len(unauthorized)} rows; samples: {samples})"
        )

    ordered = tuple(retained)
    normalized_batch = replace(
        batch,
        source_label=f"{batch.source_label}:continuous-session-only",
        source_sha256=historical_rows_sha256(ordered),
        rows=ordered,
    )
    reasons = Counter(reason for _, reason in excluded)
    return SessionNormalizationResult(
        batch=normalized_batch,
        raw_source_sha256=batch.source_sha256,
        raw_row_count=len(batch.rows),
        excluded_row_count=len(excluded),
        excluded_reasons=tuple(sorted(reasons.items())),
        excluded_timestamp_samples=tuple(
            row.opens_at.astimezone(IST).isoformat() for row, _ in excluded[:20]
        ),
    )


def diagnose_expected_minute_coverage(
    batch: HistoricalBatch,
    *,
    calendar: NseSessionCalendar,
    calendar_metadata: Mapping[str, object] | None = None,
) -> CoverageDiagnostics:
    excluded_sessions = _research_excluded_sessions(calendar_metadata or {})
    expected = _expected_opens(batch, calendar, excluded_sessions=frozenset(excluded_sessions))
    observed = {row.opens_at for row in batch.rows}
    missing = sorted(expected.difference(observed))
    by_session = Counter(item.astimezone(IST).date().isoformat() for item in missing)
    worst = tuple(sorted(by_session.items(), key=lambda item: (-item[1], item[0]))[:20])
    return CoverageDiagnostics(
        expected_minutes=len(expected),
        observed_expected_minutes=len(expected.intersection(observed)),
        missing_minutes=len(missing),
        coverage_ratio=(len(expected.intersection(observed)) / len(expected)) if expected else 0.0,
        affected_sessions=len(by_session),
        max_missing_minutes_in_one_session=max(by_session.values(), default=0),
        max_consecutive_missing_minutes=_max_consecutive_gap(missing),
        worst_sessions=worst,
        missing_timestamp_samples=tuple(
            item.astimezone(IST).isoformat() for item in missing[:20]
        ),
    )


def _expected_opens(
    batch: HistoricalBatch,
    calendar: NseSessionCalendar,
    *,
    excluded_sessions: frozenset[date] = frozenset(),
) -> set[datetime]:
    expected: set[datetime] = set()
    current = batch.request.starts_at.astimezone(IST).date()
    last = (batch.request.ends_at - timedelta(microseconds=1)).astimezone(IST).date()
    while current <= last:
        if current in excluded_sessions:
            current += timedelta(days=1)
            continue
        session = calendar.session_for_date(current)
        if session is not None:
            cursor = session.opens_at
            while cursor < session.closes_at:
                instant = cursor.astimezone(batch.request.starts_at.tzinfo)
                if batch.request.starts_at <= instant < batch.request.ends_at:
                    expected.add(instant)
                cursor += timedelta(minutes=1)
        current += timedelta(days=1)
    return expected


def _research_excluded_sessions(
    metadata: Mapping[str, object],
) -> dict[date, str]:
    """Parse exact sessions quarantined because their source data is incomplete.

    This is intentionally distinct from exchange holidays: the market traded, but
    the acquired feed cannot support an honest full-session research sample.
    """

    raw = metadata.get("research_excluded_sessions", {})
    if not isinstance(raw, Mapping):
        raise ValueError("research_excluded_sessions must be an object")
    result: dict[date, str] = {}
    for raw_date, raw_details in raw.items():
        if not isinstance(raw_details, Mapping):
            raise ValueError("Research-excluded session details must be an object")
        reason = str(raw_details.get("reason", "")).strip()
        if not reason:
            raise ValueError("Research-excluded session reason is required")
        result[date.fromisoformat(str(raw_date))] = reason
    return result


def _max_consecutive_gap(missing: list[datetime]) -> int:
    longest = 0
    current = 0
    previous: datetime | None = None
    for instant in missing:
        same_session = (
            previous is not None
            and previous.astimezone(IST).date() == instant.astimezone(IST).date()
        )
        if same_session and instant - previous == timedelta(minutes=1):
            current += 1
        else:
            current = 1
        longest = max(longest, current)
        previous = instant
    return longest


def _ignored_windows(
    metadata: Mapping[str, object],
) -> dict[date, tuple[tuple[time, time, str], ...]]:
    raw = metadata.get("ignored_provider_observation_windows", {})
    if not isinstance(raw, Mapping):
        raise ValueError("ignored_provider_observation_windows must be an object")
    result: dict[date, tuple[tuple[time, time, str], ...]] = {}
    for raw_date, raw_windows in raw.items():
        if not isinstance(raw_windows, list):
            raise ValueError("Ignored provider observation windows must be arrays")
        windows: list[tuple[time, time, str]] = []
        for raw_window in raw_windows:
            if not isinstance(raw_window, Mapping):
                raise ValueError("Ignored provider observation window must be an object")
            opens = time.fromisoformat(str(raw_window["open"]))
            closes = time.fromisoformat(str(raw_window["close"]))
            reason = str(raw_window["reason"]).strip()
            if opens >= closes or not reason:
                raise ValueError("Ignored provider observation window is invalid")
            windows.append((opens, closes, reason))
        result[date.fromisoformat(str(raw_date))] = tuple(windows)
    return result


def _authorized_exclusion_reason(
    row: FinalizedMinuteBarInput,
    windows: dict[date, tuple[tuple[time, time, str], ...]],
    calendar: NseSessionCalendar,
) -> str | None:
    if calendar.market_phase(row.opens_at).is_closing_auction:
        return "CLOSING_AUCTION_OBSERVATION_NOT_CONTINUOUS_CANDLE"
    local_open = row.opens_at.astimezone(IST)
    local_close = row.closes_at.astimezone(IST)
    for opens, closes, reason in windows.get(local_open.date(), ()):
        window_open = datetime.combine(local_open.date(), opens, IST)
        window_close = datetime.combine(local_open.date(), closes, IST)
        if window_open <= local_open and local_close <= window_close:
            return reason
    return None
