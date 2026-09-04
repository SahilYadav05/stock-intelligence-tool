from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest import TestCase

from nifty_terminal.ml.metrics import calculate_metrics
from nifty_terminal.ml.models import (
    DatasetBuildReport,
    FirstTouchLabel,
    TargetOutcome,
    TrainingSample,
    WalkForwardConfig,
)
from nifty_terminal.history.models import (
    HistoricalBatch,
    HistoricalQualityReport,
    HistoricalRequest,
    QualityStatus,
)
from nifty_terminal.history.sqlite_repository import SQLiteHistoricalRepository
from nifty_terminal.domain.candle import Timeframe
from nifty_terminal.ml.definitions import LABEL_DEFINITION_HASH, LABEL_VERSION
from nifty_terminal.ml.split import PurgedWalkForwardSplitter
from nifty_terminal.ml.training import MLResearchRunner


class MLTrainingTests(TestCase):
    def setUp(self) -> None:
        self.samples = _samples(260)
        self.config = WalkForwardConfig(
            n_splits=2,
            minimum_train_samples=100,
            test_samples=50,
            purge_bars=12,
            embargo_bars=1,
            minimum_train_class_samples=20,
        )

    def test_walk_forward_folds_are_purged_embargoed_and_chronological(self) -> None:
        folds = PurgedWalkForwardSplitter().split(self.samples, self.config)

        self.assertEqual(len(folds), 2)
        for fold in folds:
            self.assertLess(fold.train_ends_at, fold.test_starts_at)
            self.assertLessEqual(fold.maximum_train_label_end, fold.embargo_cutoff)
            self.assertTrue(set(fold.train_indices).isdisjoint(fold.test_indices))

    def test_training_creates_only_raw_oos_predictions_and_later_assessments(self) -> None:
        report = MLResearchRunner().run(
            dataset_report=_dataset_report(self.samples),
            config=self.config,
            created_at=datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc),
        )

        self.assertFalse(report.calibrated)
        self.assertFalse(report.approved_for_live_inference)
        self.assertEqual(len(report.candidates), 2)
        self.assertEqual(len(report.replay_assessments), 200)
        for candidate in report.candidates:
            self.assertEqual(len(candidate.predictions), 100)
            self.assertTrue(
                all(item.calibration_status == "UNCALIBRATED" for item in candidate.predictions)
            )
            self.assertTrue(
                all(item.generated_at == item.decision_time for item in candidate.predictions)
            )
        prediction_ids = {
            item.prediction_id
            for candidate in report.candidates
            for item in candidate.predictions
        }
        self.assertEqual(
            {item.prediction_id for item in report.replay_assessments},
            prediction_ids,
        )
        self.assertTrue(
            all(item.assessed_at > report.candidates[0].predictions[0].generated_at
                for item in report.replay_assessments)
        )
        contract = report.to_contract()
        self.assertFalse(contract["official_signal_available"])

    def test_metrics_reject_probability_rows_that_do_not_sum_to_one(self) -> None:
        import numpy as np

        with self.assertRaisesRegex(ValueError, "sum to one"):
            calculate_metrics(("UP",), np.asarray([[0.2, 0.2, 0.2]]))

    def test_sqlite_research_records_are_append_only_and_idempotent(self) -> None:
        dataset_report = _dataset_report(self.samples, include_labels=True)
        report = MLResearchRunner().run(
            dataset_report=dataset_report,
            config=self.config,
            created_at=datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc),
        )
        with TemporaryDirectory() as directory:
            repository = SQLiteHistoricalRepository(Path(directory) / "research.sqlite3")
            request = HistoricalRequest(
                instrument_id="NIFTY50_SPOT",
                timeframe=Timeframe.M1,
                starts_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                ends_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
            )
            repository.save_dataset(
                dataset_id=dataset_report.dataset_id,
                batch=HistoricalBatch(
                    provider="test",
                    source_label="test",
                    source_sha256="0" * 64,
                    acquired_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
                    request=request,
                    rows=(),
                ),
                quality=HistoricalQualityReport(
                    status=QualityStatus.PASS,
                    total_rows=0,
                    unique_minute_buckets=0,
                    correction_rows=0,
                    missing_minutes=0,
                    duplicate_provider_ids=0,
                    out_of_order_rows=0,
                    out_of_request_rows=0,
                    errors=(),
                    warnings=(),
                ),
                candles=(),
            )

            self.assertTrue(repository.save_ml_research_run(report))
            self.assertFalse(repository.save_ml_research_run(report))
            with repository._connect() as connection:
                prediction_count = connection.execute(
                    "SELECT COUNT(*) FROM ml_replay_predictions"
                ).fetchone()[0]
                assessment_count = connection.execute(
                    "SELECT COUNT(*) FROM ml_replay_assessments"
                ).fetchone()[0]
                with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                    connection.execute(
                        "UPDATE ml_replay_predictions SET predicted_outcome = 'UP'"
                    )

            self.assertEqual(prediction_count, 200)
            self.assertEqual(assessment_count, 200)


def _samples(count: int) -> tuple[TrainingSample, ...]:
    starts_at = datetime(2026, 1, 1, 4, 0, tzinfo=timezone.utc)
    outcomes = (TargetOutcome.DOWN, TargetOutcome.NEITHER, TargetOutcome.UP)
    samples = []
    for index in range(count):
        outcome = outcomes[index % len(outcomes)]
        decision_time = starts_at + timedelta(minutes=index * 5)
        signal = float(index % 3 - 1)
        samples.append(
            TrainingSample(
                sample_id=f"sample-{index}",
                dataset_id="00000000-0000-5000-8000-000000000010",
                instrument_id="NIFTY50_SPOT",
                decision_time=decision_time,
                label_window_end=decision_time + timedelta(minutes=60),
                label_id=f"label-{index}",
                outcome=outcome,
                primary_candle_id=f"primary-{index}",
                context_15m_candle_id=f"context-15m-{index // 3}",
                context_1h_candle_id=f"context-1h-{index // 12}",
                input_revision_checksum=f"{index:064x}",
                feature_names=("signal", "cycle", "trend", "noise"),
                feature_values=(
                    signal,
                    float(index % 9),
                    float((index // 20) % 2),
                    float((index * 17) % 11) / 10,
                ),
            )
        )
    return tuple(samples)


def _dataset_report(
    samples: tuple[TrainingSample, ...],
    *,
    include_labels: bool = False,
) -> DatasetBuildReport:
    support = tuple(
        (outcome.value, sum(item.outcome is outcome for item in samples))
        for outcome in (TargetOutcome.DOWN, TargetOutcome.NEITHER, TargetOutcome.UP)
    )
    return DatasetBuildReport(
        dataset_id=samples[0].dataset_id,
        candidate_decisions=len(samples),
        eligible_samples=len(samples),
        outcome_support=support,
        ambiguous_labels=0,
        unavailable_labels=0,
        excluded_feature_rows=0,
        exclusion_counts=(),
        feature_names=samples[0].feature_names,
        labels=_labels(samples) if include_labels else (),
        samples=samples,
    )


def _labels(samples: tuple[TrainingSample, ...]) -> tuple[FirstTouchLabel, ...]:
    return tuple(
        FirstTouchLabel(
            schema_version=1,
            label_id=sample.label_id,
            label_version=LABEL_VERSION,
            label_definition_hash=LABEL_DEFINITION_HASH,
            dataset_id=sample.dataset_id,
            instrument_id=sample.instrument_id,
            decision_candle_id=sample.primary_candle_id,
            decision_time=sample.decision_time,
            reference_close=Decimal("25000"),
            atr_at_decision=Decimal("100"),
            up_barrier=Decimal("25100"),
            down_barrier=Decimal("24900"),
            window_ends_at=sample.label_window_end,
            outcome=sample.outcome,
            first_touch_at=sample.label_window_end,
            first_touch_candle_id=f"touch-{sample.sample_id}",
            future_candle_ids=tuple(f"future-{sample.sample_id}-{index}" for index in range(12)),
            eligible=True,
            exclusion_reason=None,
        )
        for sample in samples
    )
