from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest import TestCase

from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.domain.candle import Candle, CandleSource, CandleStatus, Timeframe
from nifty_terminal.features.definitions import FEATURE_SET_HASH, FEATURE_VERSION
from nifty_terminal.features.models import PriceFeatureRow
from nifty_terminal.ml.dataset import TrainingDatasetAssembler
from nifty_terminal.ml.definitions import LABEL_DEFINITION_HASH, LABEL_VERSION
from nifty_terminal.ml.models import FirstTouchLabel, TargetOutcome


class TrainingDatasetAssemblerTests(TestCase):
    def test_future_higher_timeframe_rows_cannot_enter_primary_decision(self) -> None:
        decision_time = datetime(2026, 8, 24, 4, 45, tzinfo=timezone.utc)
        primary = _candle("primary", Timeframe.M5, decision_time - timedelta(minutes=5))
        context_15 = _candle("context-15", Timeframe.M15, decision_time - timedelta(minutes=15))
        future_15 = _candle("future-15", Timeframe.M15, decision_time)
        context_1h = _candle("context-1h", Timeframe.H1, decision_time - timedelta(hours=1))
        rows = {
            Timeframe.M5: (_row(primary, Decimal("1")),),
            Timeframe.M15: (
                _row(context_15, Decimal("2")),
                _row(future_15, Decimal("999")),
            ),
            Timeframe.H1: (_row(context_1h, Decimal("3")),),
        }
        label = FirstTouchLabel(
            schema_version=1,
            label_id="label-primary",
            label_version=LABEL_VERSION,
            label_definition_hash=LABEL_DEFINITION_HASH,
            dataset_id="dataset",
            instrument_id="NIFTY50_SPOT",
            decision_candle_id=primary.candle_id,
            decision_time=decision_time,
            reference_close=primary.close,
            atr_at_decision=Decimal("1"),
            up_barrier=primary.close + 1,
            down_barrier=primary.close - 1,
            window_ends_at=decision_time + timedelta(minutes=60),
            outcome=TargetOutcome.UP,
            first_touch_at=decision_time + timedelta(minutes=5),
            first_touch_candle_id="touch",
            future_candle_ids=("touch",),
            eligible=True,
            exclusion_reason=None,
        )
        assembler = TrainingDatasetAssembler(NseSessionCalendar())
        assembler._feature_engine = _FeatureEngine(rows)
        assembler._labeler = _Labeler((label,))

        report = assembler.assemble(
            dataset_id="dataset",
            minute_candles=(),
            primary_candles=(primary,),
            context_15m_candles=(context_15, future_15),
            context_1h_candles=(context_1h,),
        )

        self.assertEqual(report.eligible_samples, 1)
        sample = report.samples[0]
        self.assertEqual(sample.context_15m_candle_id, context_15.candle_id)
        self.assertNotEqual(sample.context_15m_candle_id, future_15.candle_id)
        self.assertEqual(sample.feature_values, (1.0, 2.0, 3.0))


class _FeatureEngine:
    def __init__(self, rows: dict[Timeframe, tuple[PriceFeatureRow, ...]]) -> None:
        self._rows = rows

    def calculate(self, candles: tuple[Candle, ...]) -> tuple[PriceFeatureRow, ...]:
        return self._rows[candles[0].timeframe]


class _Labeler:
    def __init__(self, labels: tuple[FirstTouchLabel, ...]) -> None:
        self._labels = labels

    def build(self, **_: object) -> tuple[FirstTouchLabel, ...]:
        return self._labels


def _candle(candle_id: str, timeframe: Timeframe, opens_at: datetime) -> Candle:
    return Candle(
        schema_version=1,
        candle_id=candle_id,
        instrument_id="NIFTY50_SPOT",
        timeframe=timeframe,
        opens_at=opens_at,
        closes_at=opens_at + timedelta(minutes=timeframe.minutes),
        open=Decimal("100"),
        high=Decimal("101"),
        low=Decimal("99"),
        close=Decimal("100"),
        volume=None,
        status=CandleStatus.FINALIZED,
        revision=1,
        source=CandleSource.AGGREGATED,
        provider="test",
        source_revision=1,
        finalized_at=opens_at + timedelta(minutes=timeframe.minutes, seconds=1),
        component_candle_ids=(f"component-{candle_id}",),
        source_watermark=f"watermark-{candle_id}",
    )


def _row(candle: Candle, value: Decimal) -> PriceFeatureRow:
    return PriceFeatureRow(
        schema_version=1,
        feature_version=FEATURE_VERSION,
        feature_set_hash=FEATURE_SET_HASH,
        source_candle_id=candle.candle_id,
        instrument_id=candle.instrument_id,
        timeframe=candle.timeframe,
        decision_time=candle.closes_at,
        values=(("test_feature", value),),
        is_ready=True,
        blockers=(),
    )
