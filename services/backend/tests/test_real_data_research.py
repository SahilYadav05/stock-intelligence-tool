from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from nifty_terminal.calibration.research import Step7ResearchPipeline
from nifty_terminal.cli.run_real_data_research import (
    DEFAULT_CALENDAR,
    REAL_DATA_WALK_FORWARD_CONFIG,
    _require_live_signal_kill_switch,
)
from nifty_terminal.domain.candle import Timeframe
from nifty_terminal.history.models import (
    HistoricalBatch,
    HistoricalQualityReport,
    HistoricalRequest,
    QualityStatus,
)
from nifty_terminal.history.calendar_loader import load_nse_calendar
from nifty_terminal.history.sqlite_repository import SQLiteHistoricalRepository
from nifty_terminal.ml.models import TargetOutcome, WalkForwardConfig
from nifty_terminal.ml.split import PurgedWalkForwardSplitter
from nifty_terminal.ml.training import MLResearchRunner
from nifty_terminal.research.real_data import evaluate_real_data_research

from test_ml_training import _dataset_report, _samples


class RealDataResearchTests(TestCase):
    def test_training_support_floor_is_distinct_from_final_evidence_gate(self) -> None:
        self.assertEqual(
            REAL_DATA_WALK_FORWARD_CONFIG.minimum_train_class_samples,
            25,
        )

    def test_real_data_folds_can_measure_a_rare_but_trainable_neither_class(self) -> None:
        source = _samples(20_100)
        samples = tuple(
            replace(
                sample,
                outcome=(
                    TargetOutcome.NEITHER
                    if index < 30
                    else TargetOutcome.UP if index % 2 else TargetOutcome.DOWN
                ),
            )
            for index, sample in enumerate(source)
        )

        folds = PurgedWalkForwardSplitter().split(
            samples,
            REAL_DATA_WALK_FORWARD_CONFIG,
        )

        self.assertEqual(len(folds), 5)
        self.assertEqual(dict(folds[0].train_class_support)["NEITHER"], 30)

    def test_default_calendar_preserves_special_session_parity(self) -> None:
        root = Path(__file__).resolve().parents[3]
        calendar = load_nse_calendar(root / DEFAULT_CALENDAR)

        saturday_session = calendar.session_for_date(datetime(2025, 2, 1).date())
        self.assertIsNotNone(saturday_session)
        self.assertEqual(
            saturday_session.opens_at.hour,  # type: ignore[union-attr]
            9,
        )

    def test_step14_contract_is_versioned_and_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[3]
        with (root / "contracts" / "real-data-research.v1.schema.json").open(
            "r", encoding="utf-8"
        ) as file:
            schema = json.load(file)

        self.assertEqual(schema["properties"]["schema_version"]["const"], 1)
        self.assertFalse(
            schema["properties"]["approved_for_live_inference"]["const"]
        )

    def test_small_fixture_fails_closed_and_never_approves_live_inference(self) -> None:
        training = MLResearchRunner().run(
            dataset_report=_dataset_report(_samples(260), include_labels=True),
            config=WalkForwardConfig(
                n_splits=2,
                minimum_train_samples=100,
                test_samples=50,
                purge_bars=12,
                embargo_bars=12,
                minimum_train_class_samples=20,
            ),
            created_at=datetime(2026, 8, 25, 12, tzinfo=timezone.utc),
        )
        observations = tuple(
            item
            for item in _observations(training)
        )
        replay_inputs = tuple(_replay_inputs(training))
        calibration = Step7ResearchPipeline().run(
            observations=observations,
            replay_inputs=replay_inputs,
            created_at=datetime(2026, 8, 25, 13, tzinfo=timezone.utc),
        ).calibration

        gate = evaluate_real_data_research(training=training, calibration=calibration)

        self.assertFalse(gate.passed)
        self.assertIn("INSUFFICIENT_ELIGIBLE_REAL_DATA_SAMPLES", gate.blockers)
        self.assertIn("INSUFFICIENT_CHRONOLOGICAL_FOLDS", gate.blockers)
        self.assertFalse(gate.to_contract()["approved_for_live_inference"])

    def test_latest_pass_dataset_ignores_newer_degraded_dataset(self) -> None:
        with TemporaryDirectory() as directory:
            repository = SQLiteHistoricalRepository(Path(directory) / "research.sqlite3")
            self._save_dataset(repository, "pass-old", QualityStatus.PASS, day=1)
            self._save_dataset(repository, "degraded-new", QualityStatus.DEGRADED, day=2)

            self.assertEqual(
                repository.latest_pass_dataset_id(instrument_id="NIFTY50_SPOT"),
                "pass-old",
            )

    def test_step14_requires_kill_switch(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("LIVE_SIGNAL_KILL_SWITCH=false\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "requires LIVE_SIGNAL_KILL_SWITCH=true"):
                _require_live_signal_kill_switch(path)
            path.write_text("LIVE_SIGNAL_KILL_SWITCH=true\n", encoding="utf-8")
            _require_live_signal_kill_switch(path)

    def _save_dataset(
        self,
        repository: SQLiteHistoricalRepository,
        dataset_id: str,
        status: QualityStatus,
        *,
        day: int,
    ) -> None:
        moment = datetime(2026, 8, day, tzinfo=timezone.utc)
        repository.save_dataset(
            dataset_id=dataset_id,
            batch=HistoricalBatch(
                provider="test",
                source_label=dataset_id,
                source_sha256=str(day) * 64,
                acquired_at=moment,
                request=HistoricalRequest(
                    instrument_id="NIFTY50_SPOT",
                    timeframe=Timeframe.M1,
                    starts_at=moment,
                    ends_at=moment + timedelta(minutes=1),
                ),
                rows=(),
            ),
            quality=HistoricalQualityReport(
                status=status,
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


def _observations(training):
    from nifty_terminal.calibration.models import CalibrationObservation

    selected = next(
        item
        for item in training.candidates
        if item.candidate_name == training.selected_research_candidate
    )
    return (
        CalibrationObservation(
            prediction_id=item.prediction_id,
            run_id=item.run_id,
            candidate_name=item.candidate_name,
            fold_index=item.fold_index,
            decision_time=item.decision_time,
            raw_probabilities=item.raw_probabilities,
            actual_outcome=item.actual_outcome,
        )
        for item in selected.predictions
    )


def _replay_inputs(training):
    from nifty_terminal.signals.replay import SignalReplayInput

    labels = {item.label_id: item for item in training.dataset_report.labels}
    selected = next(
        item
        for item in training.candidates
        if item.candidate_name == training.selected_research_candidate
    )
    return (
        SignalReplayInput(
            prediction_id=item.prediction_id,
            snapshot_id=f"fixture:{item.prediction_id}",
            instrument_id="NIFTY50_SPOT",
            decision_time=item.decision_time,
            input_revision_checksum=item.input_revision_checksum,
            reference_close=labels[item.label_id].reference_close,
            atr=labels[item.label_id].atr_at_decision,
        )
        for item in selected.predictions
        if labels[item.label_id].atr_at_decision is not None
    )
