from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import TestCase

from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.candles.engine import CandleEngine
from nifty_terminal.candles.store import InMemoryCandleStore
from nifty_terminal.domain.candle import (
    Candle,
    CandleSource,
    CandleStatus,
    FinalizedMinuteBarInput,
    Timeframe,
)
from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.domain.instruments import build_mvp_instrument_registry
from nifty_terminal.snapshots.builder import MarketStateSnapshotBuilder
from nifty_terminal.snapshots.models import DataMode
from nifty_terminal.snapshots.store import InMemorySnapshotStore


SESSION_OPEN = datetime(2026, 8, 24, 3, 45, tzinfo=timezone.utc)


def bar(index: int, revision: int = 1, adjustment: Decimal = Decimal("0")) -> FinalizedMinuteBarInput:
    opens_at = SESSION_OPEN + timedelta(minutes=index)
    price = Decimal("25000") + Decimal(index) + adjustment
    return FinalizedMinuteBarInput(
        provider_bar_id=f"bar-{index}-r{revision}",
        provider="replay",
        instrument_id="NIFTY50_SPOT",
        opens_at=opens_at,
        closes_at=opens_at + timedelta(minutes=1),
        open=price,
        high=price + 2,
        low=price - 1,
        close=price + 1,
        volume=None,
        provider_revision=revision,
        finalized_at=opens_at + timedelta(minutes=1, seconds=1),
        source_watermark=f"wm-{index}-r{revision}",
    )


class SnapshotTests(TestCase):
    def setUp(self) -> None:
        self.candles = InMemoryCandleStore()
        self.snapshots = InMemorySnapshotStore()
        self.engine = CandleEngine(
            calendar=NseSessionCalendar(),
            registry=build_mvp_instrument_registry(),
            store=self.candles,
        )
        self.builder = MarketStateSnapshotBuilder(
            candle_store=self.candles,
            snapshot_store=self.snapshots,
        )

    def _ingest(self, count: int) -> Candle:
        latest_five = None
        for index in range(count):
            result = self.engine.ingest_finalized_minute(bar(index))
            for candle in result.finalized_candles:
                if candle.timeframe is Timeframe.M5:
                    latest_five = candle
        assert latest_five is not None
        return latest_five

    def test_context_is_only_visible_after_its_finalized_close(self) -> None:
        five = self._ingest(5)
        snapshot = self.builder.build(
            primary_candle=five,
            created_at=five.closes_at + timedelta(seconds=2),
            data_mode=DataMode.REPLAY,
            data_status=ConnectionState.LIVE,
        )

        self.assertIsNone(snapshot.context_15m_candle_id)
        self.assertIsNone(snapshot.context_1h_candle_id)
        self.assertIn("FINALIZED_15M_CONTEXT_UNAVAILABLE", snapshot.blockers)
        self.assertFalse(snapshot.live_inference_eligible)

    def test_finalized_context_is_selected_at_or_before_decision_time(self) -> None:
        latest_five = self._ingest(60)
        snapshot = self.builder.build(
            primary_candle=latest_five,
            created_at=latest_five.closes_at + timedelta(seconds=2),
            data_mode=DataMode.LIVE,
            data_status=ConnectionState.LIVE,
        )

        self.assertIsNotNone(snapshot.context_15m_candle_id)
        self.assertIsNotNone(snapshot.context_1h_candle_id)
        self.assertTrue(snapshot.live_inference_eligible)
        self.assertEqual(snapshot.blockers, ())

    def test_developing_candle_is_visible_but_excluded_from_model_inputs(self) -> None:
        five = self._ingest(5)
        developing = Candle(
            schema_version=1,
            candle_id="8a54192e-5d13-5aa5-ae15-c95c734098ed",
            instrument_id="NIFTY50_SPOT",
            timeframe=Timeframe.M5,
            opens_at=five.closes_at,
            closes_at=five.closes_at + timedelta(minutes=5),
            open=Decimal("25006"), high=Decimal("25007"), low=Decimal("25005"), close=Decimal("25006"),
            volume=None, status=CandleStatus.DEVELOPING, revision=1,
            source=CandleSource.PROVISIONAL_EVENTS, provider="replay", source_revision=1, finalized_at=None,
            component_candle_ids=("visual-event",), source_watermark="visual-event",
        )
        snapshot = self.builder.build(
            primary_candle=five,
            developing_candle=developing,
            created_at=five.closes_at + timedelta(seconds=2),
            data_mode=DataMode.REPLAY,
            data_status=ConnectionState.LIVE,
        )

        self.assertEqual(snapshot.developing_candle_id, developing.candle_id)
        self.assertNotIn(developing.candle_id, snapshot.model_input_candle_ids)

    def test_correction_creates_new_snapshot_without_mutating_original(self) -> None:
        original_five = self._ingest(5)
        first = self.builder.build(
            primary_candle=original_five,
            created_at=original_five.closes_at + timedelta(seconds=2),
            data_mode=DataMode.REPLAY,
            data_status=ConnectionState.LIVE,
        )
        correction = self.engine.ingest_finalized_minute(bar(2, 2, Decimal("10")))
        revised_five = next(item for item in correction.finalized_candles if item.timeframe is Timeframe.M5)
        second = self.builder.build(
            primary_candle=revised_five,
            created_at=revised_five.closes_at + timedelta(seconds=3),
            data_mode=DataMode.REPLAY,
            data_status=ConnectionState.LIVE,
        )

        self.assertNotEqual(first.snapshot_id, second.snapshot_id)
        self.assertNotEqual(first.candle_revision_checksum, second.candle_revision_checksum)
        self.assertEqual(len(self.snapshots.all()), 2)
        self.assertEqual(first.primary_candle_id, original_five.candle_id)
        with self.assertRaises(FrozenInstanceError):
            first.primary_candle_id = revised_five.candle_id  # type: ignore[misc]
