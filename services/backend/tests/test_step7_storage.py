from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import sqlite3
from tempfile import TemporaryDirectory
from unittest import TestCase

from nifty_terminal.calibration.research import Step7ResearchPipeline
from nifty_terminal.domain.candle import Timeframe
from nifty_terminal.history.models import (
    HistoricalBatch,
    HistoricalQualityReport,
    HistoricalRequest,
    QualityStatus,
)
from nifty_terminal.history.sqlite_repository import SQLiteHistoricalRepository
from nifty_terminal.ml.models import WalkForwardConfig
from nifty_terminal.ml.training import MLResearchRunner
from nifty_terminal.signals.lifecycle import assess_signal

from test_ml_training import _dataset_report, _samples


class Step7StorageTests(TestCase):
    def test_step7_tables_are_append_only(self) -> None:
        with TemporaryDirectory() as directory:
            repository = SQLiteHistoricalRepository(Path(directory) / "research.sqlite3")
            with repository._connect() as connection:
                tables = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    ).fetchall()
                }
                self.assertTrue(
                    {
                        "calibration_runs",
                        "calibrated_predictions",
                        "signal_decisions",
                        "signal_lifecycle_events",
                    }.issubset(tables)
                )
                triggers = {
                    row[0]
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'trigger'"
                    ).fetchall()
                }
                self.assertIn("signal_decisions_no_update", triggers)
                self.assertIn("signal_decisions_no_delete", triggers)
                self.assertIn("signal_lifecycle_events_no_update", triggers)

    def test_step7_outputs_and_lifecycle_events_persist_separately(self) -> None:
        samples = _samples(260)
        dataset_report = _dataset_report(samples, include_labels=True)
        training = MLResearchRunner().run(
            dataset_report=dataset_report,
            config=WalkForwardConfig(
                n_splits=2,
                minimum_train_samples=100,
                test_samples=50,
                purge_bars=12,
                embargo_bars=1,
                minimum_train_class_samples=20,
            ),
            created_at=datetime(2026, 8, 24, 12, tzinfo=timezone.utc),
        )
        with TemporaryDirectory() as directory:
            repository = SQLiteHistoricalRepository(Path(directory) / "research.sqlite3")
            repository.save_dataset(
                dataset_id=dataset_report.dataset_id,
                batch=HistoricalBatch(
                    provider="test",
                    source_label="test",
                    source_sha256="0" * 64,
                    acquired_at=datetime(2026, 8, 24, tzinfo=timezone.utc),
                    request=HistoricalRequest(
                        instrument_id="NIFTY50_SPOT",
                        timeframe=Timeframe.M1,
                        starts_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        ends_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                    ),
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
            self.assertTrue(repository.save_ml_research_run(training))
            observations, replay_inputs = repository.load_step7_source(
                run_id=training.run_id
            )
            report = Step7ResearchPipeline().run(
                observations=observations,
                replay_inputs=replay_inputs,
                created_at=datetime(2026, 8, 24, 13, tzinfo=timezone.utc),
            )

            self.assertFalse(report.calibration.release_gate_passed)
            self.assertTrue(repository.save_step7_research_report(report))
            self.assertFalse(repository.save_step7_research_report(report))
            event = assess_signal(
                report.decisions[0],
                observed_at=report.decisions[0].decision_time,
                high=dataset_report.labels[0].reference_close,
                low=dataset_report.labels[0].reference_close,
            )
            self.assertTrue(repository.append_signal_lifecycle_event(event))
            self.assertFalse(repository.append_signal_lifecycle_event(event))

            with repository._connect() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM ml_replay_predictions"
                    ).fetchone()[0],
                    200,
                )
                self.assertEqual(
                    connection.execute(
                        "SELECT COUNT(*) FROM calibrated_predictions"
                    ).fetchone()[0],
                    40,
                )
                self.assertEqual(
                    connection.execute("SELECT COUNT(*) FROM signal_decisions").fetchone()[0],
                    40,
                )
                with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                    connection.execute(
                        "UPDATE calibrated_predictions SET actual_outcome = 'UP'"
                    )
