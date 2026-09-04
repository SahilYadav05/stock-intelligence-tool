from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from unittest import TestCase

from factories import raw_index_event
from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.candles.developing import DevelopingCandleEngine
from nifty_terminal.domain.candle import CandleStatus, Timeframe
from nifty_terminal.domain.instruments import build_mvp_instrument_registry
from nifty_terminal.ingestion.normalizer import MarketEventNormalizer


class DevelopingCandleTests(TestCase):
    def setUp(self) -> None:
        self.normalizer = MarketEventNormalizer(build_mvp_instrument_registry())
        self.engine = DevelopingCandleEngine(NseSessionCalendar())

    def test_updates_visual_ohlc_without_finalizing_or_creating_volume(self) -> None:
        first = self.normalizer.normalize(raw_index_event())
        second = self.normalizer.normalize(
            raw_index_event(
                provider_sequence=1002,
                provider_event_time=datetime(2026, 8, 24, 3, 46, tzinfo=timezone.utc),
                provider_send_time=datetime(2026, 8, 24, 3, 46, tzinfo=timezone.utc),
                server_arrival_time=datetime(2026, 8, 24, 3, 46, 0, 10_000, tzinfo=timezone.utc),
                price=Decimal("25012.50"),
                raw_payload={"fixture": True, "sequence": 1002},
            )
        )

        opening = self.engine.apply(first, Timeframe.M5)
        updated = self.engine.apply(second, Timeframe.M5)

        self.assertIsNotNone(opening)
        self.assertIsNotNone(updated)
        self.assertEqual(updated.status, CandleStatus.DEVELOPING)  # type: ignore[union-attr]
        self.assertIsNone(updated.finalized_at)  # type: ignore[union-attr]
        self.assertIsNone(updated.volume)  # type: ignore[union-attr]
        self.assertEqual(updated.open, Decimal("25000.00"))  # type: ignore[union-attr]
        self.assertEqual(updated.high, Decimal("25012.50"))  # type: ignore[union-attr]
        self.assertEqual(updated.close, Decimal("25012.50"))  # type: ignore[union-attr]

    def test_new_bucket_does_not_promote_previous_visual_candle(self) -> None:
        first = self.normalizer.normalize(raw_index_event())
        later = self.normalizer.normalize(
            raw_index_event(
                provider_sequence=1006,
                provider_event_time=datetime(2026, 8, 24, 3, 50, tzinfo=timezone.utc),
                server_arrival_time=datetime(2026, 8, 24, 3, 50, 0, 10_000, tzinfo=timezone.utc),
                price=Decimal("25020"),
                raw_payload={"fixture": True, "sequence": 1006},
            )
        )

        previous = self.engine.apply(first, Timeframe.M5)
        current = self.engine.apply(later, Timeframe.M5)

        self.assertNotEqual(previous.opens_at, current.opens_at)  # type: ignore[union-attr]
        self.assertEqual(previous.status, CandleStatus.DEVELOPING)  # type: ignore[union-attr]
        self.assertEqual(current.status, CandleStatus.DEVELOPING)  # type: ignore[union-attr]
