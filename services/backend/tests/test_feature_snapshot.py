from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from unittest import TestCase

from nifty_terminal.calendar.nse import IST, NseSessionCalendar
from nifty_terminal.candles.engine import CandleEngine
from nifty_terminal.candles.store import InMemoryCandleStore
from nifty_terminal.delivery.models import MarketStateView
from nifty_terminal.domain.candle import Candle, FinalizedMinuteBarInput, Timeframe
from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.domain.instruments import build_mvp_instrument_registry
from nifty_terminal.features.definitions import FEATURE_SET_HASH, FEATURE_VERSION
from nifty_terminal.features.snapshot import SnapshotFeatureAssembler
from nifty_terminal.snapshots.builder import MarketStateSnapshotBuilder
from nifty_terminal.snapshots.models import DataMode
from nifty_terminal.snapshots.store import InMemorySnapshotStore


class FeatureSnapshotTests(TestCase):
    def build_view(self, trading_days: int) -> MarketStateView:
        calendar = NseSessionCalendar()
        store = InMemoryCandleStore()
        engine = CandleEngine(
            calendar=calendar,
            registry=build_mvp_instrument_registry(),
            store=store,
        )
        current_date = date(2026, 8, 24)
        completed_days = 0
        global_index = 0
        latest_five: Candle | None = None
        while completed_days < trading_days:
            if current_date.weekday() < 5:
                session = calendar.session_for_date(current_date)
                assert session is not None
                session_open = session.opens_at.astimezone(timezone.utc)
                expected_minutes = int(
                    (session.closes_at - session.opens_at).total_seconds() // 60
                )
                for minute in range(expected_minutes):
                    opens_at = session_open + timedelta(minutes=minute)
                    price = Decimal("25000") + Decimal(global_index) / Decimal("10")
                    result = engine.ingest_finalized_minute(
                        FinalizedMinuteBarInput(
                            provider_bar_id=f"long-history-{global_index}",
                            provider="replay",
                            instrument_id="NIFTY50_SPOT",
                            opens_at=opens_at,
                            closes_at=opens_at + timedelta(minutes=1),
                            open=price,
                            high=price + 2,
                            low=price - 1,
                            close=price + Decimal("0.5"),
                            volume=None,
                            provider_revision=1,
                            finalized_at=opens_at + timedelta(minutes=1, seconds=1),
                            source_watermark=f"long-watermark-{global_index}",
                        )
                    )
                    for candle in result.finalized_candles:
                        if candle.timeframe is Timeframe.M5:
                            latest_five = candle
                    global_index += 1
                completed_days += 1
            current_date += timedelta(days=1)
        assert latest_five is not None

        snapshot = MarketStateSnapshotBuilder(
            candle_store=store,
            snapshot_store=InMemorySnapshotStore(),
        ).build(
            primary_candle=latest_five,
            created_at=latest_five.closes_at + timedelta(seconds=2),
            data_mode=DataMode.REPLAY,
            data_status=ConnectionState.LIVE,
        )
        input_ids = set(snapshot.model_input_candle_ids)
        finalized = tuple(
            candle
            for timeframe in (Timeframe.M5, Timeframe.M15, Timeframe.H1)
            for candle in store.latest_series(
                "NIFTY50_SPOT",
                timeframe,
                closes_at_or_before=snapshot.decision_time,
            )
            if candle.candle_id in input_ids
        )
        return MarketStateView(
            schema_version=1,
            snapshot=snapshot,
            finalized_candles=finalized,
            developing_candle=None,
            published_at=latest_five.closes_at + timedelta(seconds=3),
        )

    def test_multi_timeframe_feature_snapshot_is_reproducible(self) -> None:
        view = self.build_view(9)
        assembler = SnapshotFeatureAssembler(NseSessionCalendar())

        first = assembler.assemble(view)
        second = assembler.assemble(view)

        self.assertEqual(first, second)
        self.assertTrue(first.is_ready)
        self.assertEqual(first.feature_version, FEATURE_VERSION)
        self.assertEqual(first.feature_set_hash, FEATURE_SET_HASH)
        self.assertEqual(first.input_revision_checksum, view.snapshot.candle_revision_checksum)
        names = {name for name, _ in first.values}
        self.assertIn("primary_5m__rsi_14", names)
        self.assertIn("context_15m__atr_pct", names)
        self.assertIn("context_1h__trend_ema20_above_ema50", names)

    def test_insufficient_context_history_fails_closed(self) -> None:
        feature_snapshot = SnapshotFeatureAssembler(NseSessionCalendar()).assemble(
            self.build_view(1)
        )

        self.assertFalse(feature_snapshot.is_ready)
        self.assertTrue(
            any("CONTEXT_1H" in blocker and "INSUFFICIENT_HISTORY" in blocker
                for blocker in feature_snapshot.blockers)
        )
