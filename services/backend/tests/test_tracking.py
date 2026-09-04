from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest import TestCase

from dashboard_fixture import build_analysis_view
from market_state_fixture import build_market_state_view
from nifty_terminal.ml.definitions import LABEL_VERSION
from nifty_terminal.ml.models import TargetOutcome
from nifty_terminal.signals.models import (
    RiskLevels,
    SignalDirection,
    SignalLifecycleStatus,
)
from nifty_terminal.tracking.analytics import build_prediction_analytics
from nifty_terminal.tracking.models import (
    EvidenceStatus,
    PaperTradeStatus,
    PredictionAssessment,
    TrackedPrediction,
)
from nifty_terminal.tracking.service import TrackingService
from nifty_terminal.tracking.sqlite_repository import SQLiteTrackingLedger


class TrackingTests(TestCase):
    def test_wait_is_tracked_but_never_creates_a_paper_trade(self) -> None:
        service = TrackingService()
        analysis = build_analysis_view(build_market_state_view())

        tracked = service.register_analysis(analysis)

        self.assertEqual(tracked.direction, SignalDirection.WAIT)
        self.assertEqual(service.read_model.trades("NIFTY50_SPOT"), ())

    def test_active_signal_creates_conservative_paper_lifecycle(self) -> None:
        service = TrackingService()
        analysis = _active_analysis()
        service.register_analysis(analysis)
        decision_time = analysis.decision_time

        opened = service.assess_paper_trades(
            "NIFTY50_SPOT",
            observed_at=decision_time + timedelta(minutes=5),
            high=Decimal("25020"),
            low=Decimal("25005"),
            close=Decimal("25015"),
        )
        closed = service.assess_paper_trades(
            "NIFTY50_SPOT",
            observed_at=decision_time + timedelta(minutes=10),
            high=Decimal("25120"),
            low=Decimal("25000"),
            close=Decimal("25100"),
        )

        self.assertEqual(opened, 1)
        self.assertEqual(closed, 1)
        events = service.read_model.events("NIFTY50_SPOT")
        self.assertEqual(events[0].status, PaperTradeStatus.TARGET_1_HIT)
        self.assertEqual(events[0].pnl_points, Decimal("90"))
        self.assertFalse(events[0].to_contract().get("automatic_execution", False))

    def test_metrics_are_hidden_until_minimum_sample(self) -> None:
        generated_at = datetime(2026, 8, 24, 12, tzinfo=timezone.utc)
        predictions = tuple(_prediction(index, generated_at) for index in range(3))
        assessments = tuple(
            _assessment(item, TargetOutcome.UP if index < 2 else TargetOutcome.DOWN)
            for index, item in enumerate(predictions)
        )

        hidden = build_prediction_analytics(
            instrument_id="NIFTY50_SPOT",
            generated_at=generated_at + timedelta(hours=2),
            predictions=predictions,
            assessments=assessments[:2],
            paper_trades=(),
            paper_events=(),
            minimum_sample=3,
        )
        ready = build_prediction_analytics(
            instrument_id="NIFTY50_SPOT",
            generated_at=generated_at + timedelta(hours=2),
            predictions=predictions,
            assessments=assessments,
            paper_trades=(),
            paper_events=(),
            minimum_sample=3,
        )

        self.assertEqual(hidden.metrics_status, EvidenceStatus.INSUFFICIENT_SAMPLE)
        self.assertIsNone(hidden.accuracy)
        self.assertEqual(ready.metrics_status, EvidenceStatus.READY)
        self.assertAlmostEqual(ready.accuracy or 0, 2 / 3)
        self.assertIsNotNone(ready.multiclass_brier_score)
        self.assertIsNotNone(ready.expected_calibration_error)
        self.assertFalse(ready.to_contract()["performance_claim_allowed"])

    def test_sqlite_tracking_ledger_is_idempotent_and_append_only(self) -> None:
        item = _prediction(1, datetime(2026, 8, 24, 12, tzinfo=timezone.utc))
        with TemporaryDirectory() as directory:
            ledger = SQLiteTrackingLedger(Path(directory) / "tracking.sqlite3")
            self.assertTrue(ledger.append_prediction(item))
            self.assertFalse(ledger.append_prediction(item))
            with self.assertRaisesRegex(ValueError, "immutable tracking identity"):
                ledger.append_prediction(replace(item, model_version="different"))
            contracts = ledger.load_contracts(
                "tracked_predictions", instrument_id="NIFTY50_SPOT"
            )
            self.assertEqual(contracts[0]["prediction_id"], item.prediction_id)
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                with ledger._connect() as connection:
                    connection.execute(
                        "UPDATE tracked_predictions SET instrument_id = 'OTHER'"
                    )


def _active_analysis():
    analysis = build_analysis_view(build_market_state_view())
    risk = RiskLevels(
        entry_low=Decimal("25000"),
        entry_high=Decimal("25010"),
        stop=Decimal("24900"),
        invalidation=Decimal("24900"),
        target1=Decimal("25100"),
        target2=Decimal("25200"),
        target3=Decimal("25300"),
        target1_reward_risk=1.0,
    )
    signal = replace(
        analysis.signal,
        direction=SignalDirection.BUY,
        lifecycle_status=SignalLifecycleStatus.ACTIVE,
        probabilities=(("DOWN", 0.1), ("NEITHER", 0.2), ("UP", 0.7)),
        expected_atr=0.5,
        risk_levels=risk,
        blockers=(),
    )
    return replace(analysis, signal=signal)


def _prediction(index: int, created_at: datetime) -> TrackedPrediction:
    decision = created_at + timedelta(minutes=index * 5)
    return TrackedPrediction(
        prediction_id=f"prediction-{index}",
        signal_id=f"signal-{index}",
        snapshot_id=f"snapshot-{index}",
        instrument_id="NIFTY50_SPOT",
        decision_time=decision,
        registered_at=decision + timedelta(seconds=1),
        direction=SignalDirection.BUY,
        predicted_outcome=TargetOutcome.UP,
        probabilities=(("DOWN", 0.1), ("NEITHER", 0.2), ("UP", 0.7)),
        model_version="model.v1",
        calibration_version="calibration.v1",
        feature_version="price_features.v1",
        signal_policy_version="wait_first_atr_policy.v1",
        input_revision_checksum=f"{index:064x}",
    )


def _assessment(
    prediction: TrackedPrediction,
    actual: TargetOutcome,
) -> PredictionAssessment:
    assessed_at = prediction.decision_time + timedelta(minutes=60)
    return PredictionAssessment(
        assessment_id=f"assessment-{prediction.prediction_id}",
        prediction_id=prediction.prediction_id,
        signal_id=prediction.signal_id,
        snapshot_id=prediction.snapshot_id,
        instrument_id=prediction.instrument_id,
        decision_time=prediction.decision_time,
        assessed_at=assessed_at,
        predicted_outcome=TargetOutcome.UP,
        actual_outcome=actual,
        probabilities=prediction.probabilities or (),
        correct=actual is TargetOutcome.UP,
        first_touch_at=prediction.decision_time + timedelta(minutes=20),
        label_version=LABEL_VERSION,
        input_revision_checksum=prediction.input_revision_checksum,
    )
