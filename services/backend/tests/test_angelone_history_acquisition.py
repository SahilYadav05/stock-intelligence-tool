from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase

from nifty_terminal.calendar.nse import IST, MarketPhase, NseSessionCalendar
from nifty_terminal.domain.candle import FinalizedMinuteBarInput, Timeframe
from nifty_terminal.history.calendar_loader import load_nse_calendar, validate_calendar_coverage
from nifty_terminal.history.models import HistoricalBatch, HistoricalRequest
from nifty_terminal.history.session_normalizer import (
    diagnose_expected_minute_coverage,
    normalize_to_continuous_sessions,
)
from nifty_terminal.history.sources.angelone_source import (
    AcquiredHistoricalDataSource,
    AngelOneHistoricalAcquirer,
)


START = datetime(2026, 8, 24, 3, 45, tzinfo=timezone.utc)
ACQUIRED = datetime(2026, 8, 25, 11, 0, tzinfo=timezone.utc)


class FakeMinuteProvider:
    def __init__(self, rows_by_call: tuple[tuple[FinalizedMinuteBarInput, ...], ...]) -> None:
        self._rows = list(rows_by_call)
        self.calls: list[tuple[datetime, datetime]] = []

    async def fetch_finalized_minutes(
        self,
        *,
        from_time: datetime,
        to_time: datetime,
    ) -> tuple[FinalizedMinuteBarInput, ...]:
        self.calls.append((from_time, to_time))
        return self._rows.pop(0)


class AngelOneHistoricalAcquirerTests(IsolatedAsyncioTestCase):
    async def test_chunked_acquisition_is_sorted_hashed_and_volume_free(self) -> None:
        provider = FakeMinuteProvider(((_bar(0, minute=1), _bar(0)), (_bar(1),)))
        request = _request(days=2)
        progress: list[tuple[int, int, int]] = []
        acquirer = AngelOneHistoricalAcquirer(
            provider=provider,
            chunk_days=1,
            request_delay_milliseconds=0,
            clock=lambda: ACQUIRED,
        )

        batch = await acquirer.acquire(
            request,
            progress=lambda completed, total, rows: progress.append(
                (completed, total, rows)
            ),
        )

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(
            tuple(item.opens_at for item in batch.rows),
            (START, START + timedelta(minutes=1), START + timedelta(days=1)),
        )
        self.assertTrue(all(item.volume is None for item in batch.rows))
        self.assertEqual(len(batch.source_sha256), 64)
        self.assertEqual(progress, [(1, 2, 2), (2, 2, 3)])
        self.assertEqual(AcquiredHistoricalDataSource(batch).fetch(request), batch)

    async def test_conflicting_rows_stop_instead_of_silently_selecting_one(self) -> None:
        first = _bar(0)
        conflicting = FinalizedMinuteBarInput(
            provider_bar_id="conflicting-provider-id",
            provider=first.provider,
            instrument_id=first.instrument_id,
            opens_at=first.opens_at,
            closes_at=first.closes_at,
            open=first.open,
            high=first.high + Decimal("5"),
            low=first.low,
            close=first.close,
            volume=None,
            provider_revision=1,
            finalized_at=first.finalized_at,
            source_watermark="conflicting-watermark",
        )
        provider = FakeMinuteProvider(((first, conflicting), ()))
        acquirer = AngelOneHistoricalAcquirer(
            provider=provider,
            chunk_days=1,
            request_delay_milliseconds=0,
        )

        with self.assertRaisesRegex(ValueError, "conflicting finalized rows"):
            await acquirer.acquire(_request(days=2))

    async def test_nifty_volume_is_rejected_at_acquisition_boundary(self) -> None:
        original = _bar(0)
        with_volume = FinalizedMinuteBarInput(
            provider_bar_id=original.provider_bar_id,
            provider=original.provider,
            instrument_id=original.instrument_id,
            opens_at=original.opens_at,
            closes_at=original.closes_at,
            open=original.open,
            high=original.high,
            low=original.low,
            close=original.close,
            volume=Decimal("100"),
            provider_revision=1,
            finalized_at=original.finalized_at,
            source_watermark=original.source_watermark,
        )
        provider = FakeMinuteProvider(((with_volume,),))
        acquirer = AngelOneHistoricalAcquirer(
            provider=provider,
            request_delay_milliseconds=0,
        )

        with self.assertRaisesRegex(ValueError, "volume must remain null"):
            await acquirer.acquire(_request(days=1))


class CalendarCoverageTests(TestCase):
    def test_explicit_verified_range_is_required(self) -> None:
        metadata = {
            "exchange": "NSE",
            "segment": "CAPITAL_MARKET",
            "verified_from": "2025-01-01",
            "verified_through": "2026-08-25",
        }
        validate_calendar_coverage(
            metadata,
            starts_on=date(2025, 1, 1),
            ends_on=date(2026, 8, 25),
        )
        with self.assertRaisesRegex(ValueError, "exceeds"):
            validate_calendar_coverage(
                metadata,
                starts_on=date(2024, 12, 31),
                ends_on=date(2026, 8, 25),
            )

    def test_budget_weekend_sessions_are_explicitly_open(self) -> None:
        calendar = load_nse_calendar(
            Path("config/nse-calendar-through-2026-08-25.json")
        )
        for session_date in (date(2025, 2, 1), date(2026, 2, 1)):
            session = calendar.session_for_date(session_date)
            self.assertIsNotNone(session)
            self.assertEqual(session.opens_at.timetz().replace(tzinfo=None), time(9, 15))
            self.assertEqual(session.closes_at.timetz().replace(tzinfo=None), time(15, 30))

    def test_cas_go_live_changes_continuous_close_and_exposes_phases(self) -> None:
        calendar = load_nse_calendar(
            Path("config/nse-calendar-through-2026-08-25.json")
        )
        legacy = calendar.session_for_date(date(2026, 7, 31))
        cas_day = calendar.session_for_date(date(2026, 8, 3))
        self.assertEqual(legacy.closes_at.timetz().replace(tzinfo=None), time(15, 30))
        self.assertEqual(cas_day.closes_at.timetz().replace(tzinfo=None), time(15, 15))
        self.assertEqual(
            calendar.market_phase(datetime(2026, 8, 3, 15, 17, tzinfo=IST)),
            MarketPhase.CLOSING_AUCTION_REFERENCE,
        )
        self.assertEqual(
            calendar.market_phase(datetime(2026, 8, 3, 15, 25, tzinfo=IST)),
            MarketPhase.CLOSING_AUCTION_ORDER_ENTRY,
        )
        self.assertEqual(
            calendar.market_phase(datetime(2026, 8, 3, 15, 32, tzinfo=IST)),
            MarketPhase.CLOSING_AUCTION_MATCHING,
        )
        self.assertEqual(
            calendar.market_phase(datetime(2026, 8, 3, 15, 35, tzinfo=IST)),
            MarketPhase.CLOSED,
        )


class SessionNormalizationTests(TestCase):
    def test_cas_observations_are_separate_from_complete_continuous_history(self) -> None:
        config_path = Path("config/nse-calendar-through-2026-08-25.json")
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        calendar = load_nse_calendar(config_path)
        session_date = date(2026, 8, 3)
        session_open = datetime.combine(session_date, time(9, 15), IST)
        continuous = tuple(
            _bar_at(session_open + timedelta(minutes=index)) for index in range(360)
        )
        auction = tuple(
            _bar_at(datetime.combine(session_date, auction_time, IST))
            for auction_time in (time(15, 15), time(15, 29), time(15, 30))
        )
        result = normalize_to_continuous_sessions(
            _batch(continuous + auction, session_date),
            calendar=calendar,
            calendar_metadata=metadata,
        )
        diagnostics = diagnose_expected_minute_coverage(
            result.batch,
            calendar=calendar,
        )

        self.assertEqual(len(result.batch.rows), 360)
        self.assertEqual(result.excluded_row_count, 3)
        self.assertEqual(
            dict(result.excluded_reasons),
            {"CLOSING_AUCTION_OBSERVATION_NOT_CONTINUOUS_CANDLE": 3},
        )
        self.assertEqual(diagnostics.missing_minutes, 0)
        self.assertTrue(diagnostics.research_acceptable)

    def test_observed_provider_only_muhurat_minutes_are_explicitly_excluded(self) -> None:
        config_path = Path("config/nse-calendar-through-2026-08-25.json")
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        calendar = load_nse_calendar(config_path)
        session_date = date(2025, 10, 21)
        provider_only = (
            datetime.combine(session_date, time(11, 17), IST),
            datetime.combine(session_date, time(11, 40), IST),
            datetime.combine(session_date, time(14, 45), IST),
            datetime.combine(session_date, time(14, 46), IST),
        )
        normal = datetime.combine(session_date, time(13, 45), IST)
        batch = _batch(
            tuple(_bar_at(item) for item in provider_only + (normal,)),
            session_date,
        )

        result = normalize_to_continuous_sessions(
            batch,
            calendar=calendar,
            calendar_metadata=metadata,
        )

        self.assertEqual(result.excluded_row_count, 4)
        self.assertEqual(
            tuple(item.opens_at for item in result.batch.rows),
            (normal.astimezone(timezone.utc),),
        )

    def test_only_explicit_muhurat_preopen_observations_are_excluded(self) -> None:
        session_date = date(2025, 10, 21)
        calendar = NseSessionCalendar(
            special_sessions={session_date: (time(13, 45), time(14, 45))}
        )
        preopen = datetime.combine(session_date, time(13, 33), IST)
        normal = datetime.combine(session_date, time(13, 45), IST)
        batch = _batch(( _bar_at(preopen), _bar_at(normal) ), session_date)
        result = normalize_to_continuous_sessions(
            batch,
            calendar=calendar,
            calendar_metadata={
                "ignored_provider_observation_windows": {
                    "2025-10-21": [
                        {
                            "open": "13:30",
                            "close": "13:45",
                            "reason": "CAPITAL_MARKET_PRE_OPEN_NOT_CONTINUOUS_TRADING",
                        }
                    ]
                }
            },
        )

        self.assertEqual(result.excluded_row_count, 1)
        self.assertEqual(tuple(item.opens_at for item in result.batch.rows), (normal.astimezone(timezone.utc),))
        self.assertNotEqual(result.batch.source_sha256, batch.source_sha256)

    def test_unauthorized_out_of_session_rows_stop_the_import(self) -> None:
        session_date = date(2026, 8, 24)
        calendar = NseSessionCalendar()
        unauthorized = datetime.combine(session_date, time(8, 59), IST)
        batch = _batch((_bar_at(unauthorized),), session_date)

        with self.assertRaisesRegex(ValueError, "outside the verified"):
            normalize_to_continuous_sessions(
                batch,
                calendar=calendar,
                calendar_metadata={},
            )

    def test_sparse_missing_minutes_are_reported_and_never_filled(self) -> None:
        session_date = date(2026, 8, 24)
        session_open = datetime.combine(session_date, time(9, 15), IST)
        rows = tuple(
            _bar_at(session_open + timedelta(minutes=index))
            for index in range(360)
            if index != 100
        )
        batch = _batch(rows, session_date)
        diagnostics = diagnose_expected_minute_coverage(
            batch,
            calendar=NseSessionCalendar(),
        )

        self.assertEqual(diagnostics.expected_minutes, 360)
        self.assertEqual(diagnostics.missing_minutes, 1)
        self.assertEqual(diagnostics.max_consecutive_missing_minutes, 1)
        self.assertTrue(diagnostics.research_acceptable)

    def test_source_incomplete_session_is_explicitly_quarantined(self) -> None:
        session_date = date(2026, 8, 24)
        session_open = datetime.combine(session_date, time(9, 15), IST)
        batch = _batch(
            tuple(_bar_at(session_open + timedelta(minutes=index)) for index in range(360)),
            session_date,
        )
        metadata = {
            "research_excluded_sessions": {
                session_date.isoformat(): {"reason": "TEST_SOURCE_GAP"}
            }
        }

        result = normalize_to_continuous_sessions(
            batch,
            calendar=NseSessionCalendar(),
            calendar_metadata=metadata,
        )
        diagnostics = diagnose_expected_minute_coverage(
            result.batch,
            calendar=NseSessionCalendar(),
            calendar_metadata=metadata,
        )

        self.assertEqual(len(result.batch.rows), 0)
        self.assertEqual(result.excluded_row_count, 360)
        self.assertEqual(dict(result.excluded_reasons), {"TEST_SOURCE_GAP": 360})
        self.assertEqual(diagnostics.expected_minutes, 0)
        self.assertEqual(diagnostics.missing_minutes, 0)


def _request(*, days: int) -> HistoricalRequest:
    return HistoricalRequest(
        instrument_id="NIFTY50_SPOT",
        timeframe=Timeframe.M1,
        starts_at=START,
        ends_at=START + timedelta(days=days),
    )


def _bar(day: int, *, minute: int = 0) -> FinalizedMinuteBarInput:
    opens_at = START + timedelta(days=day, minutes=minute)
    price = Decimal("24500") + Decimal(day)
    return FinalizedMinuteBarInput(
        provider_bar_id=f"angelone-{day}-{minute}",
        provider="angelone",
        instrument_id="NIFTY50_SPOT",
        opens_at=opens_at,
        closes_at=opens_at + timedelta(minutes=1),
        open=price,
        high=price + Decimal("2"),
        low=price - Decimal("1"),
        close=price + Decimal("1"),
        volume=None,
        provider_revision=1,
        finalized_at=ACQUIRED,
        source_watermark=f"watermark-{day}-{minute}",
    )


def _bar_at(opens_at: datetime) -> FinalizedMinuteBarInput:
    utc_open = opens_at.astimezone(timezone.utc)
    suffix = utc_open.isoformat()
    return FinalizedMinuteBarInput(
        provider_bar_id=f"angelone-{suffix}",
        provider="angelone",
        instrument_id="NIFTY50_SPOT",
        opens_at=utc_open,
        closes_at=utc_open + timedelta(minutes=1),
        open=Decimal("24500"),
        high=Decimal("24502"),
        low=Decimal("24499"),
        close=Decimal("24501"),
        volume=None,
        provider_revision=1,
        finalized_at=ACQUIRED,
        source_watermark=f"watermark-{suffix}",
    )


def _batch(
    rows: tuple[FinalizedMinuteBarInput, ...],
    session_date: date,
) -> HistoricalBatch:
    request = HistoricalRequest(
        instrument_id="NIFTY50_SPOT",
        timeframe=Timeframe.M1,
        starts_at=datetime.combine(session_date, time.min, IST).astimezone(timezone.utc),
        ends_at=datetime.combine(
            session_date + timedelta(days=1), time.min, IST
        ).astimezone(timezone.utc),
    )
    return HistoricalBatch(
        provider="angelone",
        source_label="test",
        source_sha256="0" * 64,
        acquired_at=ACQUIRED,
        request=request,
        rows=rows,
    )
