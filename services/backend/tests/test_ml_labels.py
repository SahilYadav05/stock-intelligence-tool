from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import TestCase

from history_fixture import SESSION_OPEN
from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.domain.candle import Candle, CandleSource, CandleStatus, Timeframe
from nifty_terminal.features.definitions import FEATURE_SET_HASH, FEATURE_VERSION
from nifty_terminal.features.models import PriceFeatureRow
from nifty_terminal.ml.definitions import HORIZON_MINUTES, LABEL_VERSION
from nifty_terminal.ml.labels import FirstTouchLabeler
from nifty_terminal.ml.models import TargetOutcome


class FirstTouchLabelTests(TestCase):
    def setUp(self) -> None:
        self.labeler = FirstTouchLabeler(NseSessionCalendar())

    def test_up_barrier_touched_first(self) -> None:
        candles = _five_minute_candles(13)
        candles = (candles[0], replace(candles[1], high=Decimal("102.1"))) + candles[2:]

        label = self.labeler.build(
            dataset_id="00000000-0000-5000-8000-000000000001",
            primary_candles=candles,
            primary_features=(_atr_feature(candles[0], Decimal("2")),),
        )[0]

        self.assertEqual(label.label_version, LABEL_VERSION)
        self.assertEqual(label.outcome, TargetOutcome.UP)
        self.assertTrue(label.eligible)
        self.assertEqual(label.window_ends_at, candles[0].closes_at + timedelta(minutes=60))

    def test_one_minute_order_resolves_a_double_touch(self) -> None:
        candles = _five_minute_candles(13)
        candles = (candles[0], replace(candles[1], high=Decimal("102.5"), low=Decimal("97.5"))) + candles[2:]
        minutes = list(_minute_candles(candles[1].opens_at, 5))
        minutes[0] = replace(minutes[0], low=Decimal("97.5"))
        minutes[1] = replace(minutes[1], high=Decimal("102.5"))

        label = self.labeler.build(
            dataset_id="00000000-0000-5000-8000-000000000002",
            primary_candles=candles,
            primary_features=(_atr_feature(candles[0], Decimal("2")),),
            minute_candles=tuple(minutes),
        )[0]

        self.assertEqual(label.outcome, TargetOutcome.DOWN)
        self.assertEqual(label.first_touch_candle_id, minutes[0].candle_id)
        self.assertTrue(label.eligible)

    def test_unresolved_same_minute_double_touch_is_ambiguous(self) -> None:
        candles = _five_minute_candles(13)
        candles = (candles[0], replace(candles[1], high=Decimal("102.5"), low=Decimal("97.5"))) + candles[2:]
        minutes = list(_minute_candles(candles[1].opens_at, 5))
        minutes[0] = replace(minutes[0], high=Decimal("102.5"), low=Decimal("97.5"))

        label = self.labeler.build(
            dataset_id="00000000-0000-5000-8000-000000000003",
            primary_candles=candles,
            primary_features=(_atr_feature(candles[0], Decimal("2")),),
            minute_candles=tuple(minutes),
        )[0]

        self.assertEqual(label.outcome, TargetOutcome.AMBIGUOUS)
        self.assertFalse(label.eligible)
        self.assertEqual(label.exclusion_reason, "AMBIGUOUS_INTRABAR_ORDER")

    def test_decision_too_close_to_session_close_is_excluded(self) -> None:
        late_open = datetime(2026, 8, 24, 9, 30, tzinfo=timezone.utc)
        candles = _five_minute_candles(13, starts_at=late_open)

        label = self.labeler.build(
            dataset_id="00000000-0000-5000-8000-000000000004",
            primary_candles=candles,
            primary_features=(_atr_feature(candles[0], Decimal("2")),),
        )[0]

        self.assertIsNone(label.outcome)
        self.assertFalse(label.eligible)
        self.assertEqual(label.exclusion_reason, "OUTCOME_WINDOW_CROSSES_SESSION_CLOSE")

    def test_last_standard_cas_day_decision_is_exactly_1415_ist(self) -> None:
        last_allowed_open = datetime(2026, 8, 24, 8, 40, tzinfo=timezone.utc)
        allowed = _five_minute_candles(13, starts_at=last_allowed_open)
        allowed_label = self.labeler.build(
            dataset_id="00000000-0000-5000-8000-000000000005",
            primary_candles=allowed,
            primary_features=(_atr_feature(allowed[0], Decimal("2")),),
        )[0]
        self.assertTrue(allowed_label.eligible)
        self.assertEqual(allowed_label.window_ends_at.hour, 9)
        self.assertEqual(allowed_label.window_ends_at.minute, 45)

        blocked = _five_minute_candles(
            13,
            starts_at=datetime(2026, 8, 24, 8, 45, tzinfo=timezone.utc),
        )
        blocked_label = self.labeler.build(
            dataset_id="00000000-0000-5000-8000-000000000006",
            primary_candles=blocked,
            primary_features=(_atr_feature(blocked[0], Decimal("2")),),
        )[0]
        self.assertFalse(blocked_label.eligible)
        self.assertEqual(
            blocked_label.exclusion_reason,
            "OUTCOME_WINDOW_CROSSES_SESSION_CLOSE",
        )

    def test_horizon_is_locked_to_sixty_minutes(self) -> None:
        self.assertEqual(HORIZON_MINUTES, 60)


def _five_minute_candles(
    count: int,
    *,
    starts_at: datetime = SESSION_OPEN,
) -> tuple[Candle, ...]:
    return tuple(
        _candle(
            index=index,
            timeframe=Timeframe.M5,
            opens_at=starts_at + timedelta(minutes=index * 5),
        )
        for index in range(count)
    )


def _minute_candles(starts_at: datetime, count: int) -> tuple[Candle, ...]:
    return tuple(
        _candle(
            index=10_000 + index,
            timeframe=Timeframe.M1,
            opens_at=starts_at + timedelta(minutes=index),
        )
        for index in range(count)
    )


def _candle(*, index: int, timeframe: Timeframe, opens_at: datetime) -> Candle:
    return Candle(
        schema_version=1,
        candle_id=f"00000000-0000-5000-8000-{index:012d}",
        instrument_id="NIFTY50_SPOT",
        timeframe=timeframe,
        opens_at=opens_at,
        closes_at=opens_at + timedelta(minutes=timeframe.minutes),
        open=Decimal("100"),
        high=Decimal("100.5"),
        low=Decimal("99.5"),
        close=Decimal("100"),
        volume=None,
        status=CandleStatus.FINALIZED,
        revision=1,
        source=CandleSource.AGGREGATED,
        provider="test",
        source_revision=1,
        finalized_at=opens_at + timedelta(minutes=timeframe.minutes, seconds=1),
        component_candle_ids=(f"component-{index}",),
        source_watermark=f"watermark-{index}",
    )


def _atr_feature(candle: Candle, atr: Decimal) -> PriceFeatureRow:
    return PriceFeatureRow(
        schema_version=1,
        feature_version=FEATURE_VERSION,
        feature_set_hash=FEATURE_SET_HASH,
        source_candle_id=candle.candle_id,
        instrument_id=candle.instrument_id,
        timeframe=candle.timeframe,
        decision_time=candle.closes_at,
        values=(("atr_14", atr),),
        is_ready=True,
        blockers=(),
    )
