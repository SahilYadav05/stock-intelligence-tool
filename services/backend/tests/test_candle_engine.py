from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import TestCase

from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.candles.engine import CandleEngine
from nifty_terminal.candles.store import InMemoryCandleStore
from nifty_terminal.domain.candle import CandleStatus, FinalizedMinuteBarInput, Timeframe
from nifty_terminal.domain.instruments import build_mvp_instrument_registry


SESSION_OPEN = datetime(2026, 8, 24, 3, 45, tzinfo=timezone.utc)


def minute_bar(index: int, *, revision: int = 1, bump: Decimal = Decimal("0")) -> FinalizedMinuteBarInput:
    opens_at = SESSION_OPEN + timedelta(minutes=index)
    base = Decimal("25000") + Decimal(index) + bump
    return FinalizedMinuteBarInput(
        provider_bar_id=f"replay-minute-{index}-r{revision}",
        provider="replay",
        instrument_id="NIFTY50_SPOT",
        opens_at=opens_at,
        closes_at=opens_at + timedelta(minutes=1),
        open=base,
        high=base + Decimal("2"),
        low=base - Decimal("1"),
        close=base + Decimal("1"),
        volume=None,
        provider_revision=revision,
        finalized_at=opens_at + timedelta(minutes=1, seconds=1),
        source_watermark=f"watermark-{index}-r{revision}",
    )


class CandleEngineTests(TestCase):
    def setUp(self) -> None:
        self.store = InMemoryCandleStore()
        self.engine = CandleEngine(
            calendar=NseSessionCalendar(),
            registry=build_mvp_instrument_registry(),
            store=self.store,
        )

    def ingest(self, count: int) -> list:
        return [self.engine.ingest_finalized_minute(minute_bar(index)) for index in range(count)]

    def test_five_minute_candle_requires_all_five_finalized_minutes(self) -> None:
        results = self.ingest(4)
        self.assertTrue(all(not item.finalized_candles for item in results))

        result = self.engine.ingest_finalized_minute(minute_bar(4))
        five = next(item for item in result.finalized_candles if item.timeframe is Timeframe.M5)

        self.assertEqual(five.status, CandleStatus.FINALIZED)
        self.assertEqual(five.open, Decimal("25000"))
        self.assertEqual(five.close, Decimal("25005"))
        self.assertIsNone(five.volume)
        self.assertEqual(len(five.component_candle_ids), 5)

    def test_fifteen_minute_and_hour_context_require_complete_buckets(self) -> None:
        results = self.ingest(60)
        all_created = [candle for result in results for candle in result.finalized_candles]

        self.assertEqual(sum(item.timeframe is Timeframe.M5 for item in all_created), 12)
        self.assertEqual(sum(item.timeframe is Timeframe.M15 for item in all_created), 4)
        self.assertEqual(sum(item.timeframe is Timeframe.H1 for item in all_created), 1)

    def test_provider_correction_appends_revisions_and_propagates(self) -> None:
        self.ingest(5)
        original_minute = self.store.latest("NIFTY50_SPOT", Timeframe.M1, SESSION_OPEN + timedelta(minutes=2))
        original_five = self.store.latest("NIFTY50_SPOT", Timeframe.M5, SESSION_OPEN)

        result = self.engine.ingest_finalized_minute(
            minute_bar(2, revision=2, bump=Decimal("10"))
        )
        revised_five = next(
            item for item in result.finalized_candles if item.timeframe is Timeframe.M5
        )

        self.assertEqual(result.minute_candle.revision, 2)
        self.assertEqual(result.minute_candle.supersedes_candle_id, original_minute.candle_id)  # type: ignore[union-attr]
        self.assertEqual(revised_five.revision, 2)
        self.assertEqual(revised_five.supersedes_candle_id, original_five.candle_id)  # type: ignore[union-attr]
        self.assertEqual(len(self.store.revisions("NIFTY50_SPOT", Timeframe.M5, SESSION_OPEN)), 2)
        self.assertEqual(original_five.revision, 1)  # type: ignore[union-attr]

    def test_invalid_ohlc_and_fake_index_volume_fail_closed(self) -> None:
        invalid = replace(minute_bar(0), low=Decimal("26000"))
        with self.assertRaisesRegex(ValueError, "OHLC"):
            self.engine.ingest_finalized_minute(invalid)

        with_volume = replace(minute_bar(0), volume=Decimal("100"))
        with self.assertRaisesRegex(ValueError, "Volume"):
            self.engine.ingest_finalized_minute(with_volume)
