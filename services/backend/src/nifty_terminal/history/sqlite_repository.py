"""Zero-cost local SQLite implementation of the historical repository."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
import sqlite3

from nifty_terminal.domain.candle import Candle, CandleSource, CandleStatus, Timeframe
from nifty_terminal.features.models import PriceFeatureRow
from nifty_terminal.history.models import HistoricalBatch, HistoricalQualityReport, QualityStatus
from nifty_terminal.history.repository import HistoricalRepository
from nifty_terminal.ml.models import TrainingRunReport
from nifty_terminal.calibration.models import CalibrationObservation
from nifty_terminal.ml.models import TargetOutcome
from nifty_terminal.signals.models import SignalLifecycleEvent
from nifty_terminal.signals.replay import SignalReplayInput, Step7ResearchReport


class SQLiteHistoricalRepository(HistoricalRepository):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def load_dataset_quality_status(self, *, dataset_id: str) -> QualityStatus | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT quality_status FROM historical_datasets WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
        return QualityStatus(row["quality_status"]) if row else None

    def latest_pass_dataset_id(self, *, instrument_id: str) -> str | None:
        """Return the newest immutable PASS dataset for one canonical instrument."""
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT dataset_id
                FROM historical_datasets
                WHERE instrument_id = ? AND quality_status = 'PASS'
                ORDER BY ends_at DESC, acquired_at DESC, dataset_id DESC
                LIMIT 1
                """,
                (instrument_id,),
            ).fetchone()
        return str(row["dataset_id"]) if row else None

    def save_dataset(
        self,
        *,
        dataset_id: str,
        batch: HistoricalBatch,
        quality: HistoricalQualityReport,
        candles: tuple[Candle, ...],
    ) -> bool:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM historical_datasets WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
            if exists:
                return False
            connection.execute(
                """
                INSERT INTO historical_datasets (
                    dataset_id, instrument_id, source_timeframe, provider, source_label,
                    source_sha256, acquired_at, starts_at, ends_at, quality_status,
                    quality_json, source_row_count, candle_revision_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    batch.request.instrument_id,
                    batch.request.timeframe.value,
                    batch.provider,
                    batch.source_label,
                    batch.source_sha256,
                    _time(batch.acquired_at),
                    _time(batch.request.starts_at),
                    _time(batch.request.ends_at),
                    quality.status.value,
                    json.dumps(quality.to_contract(), sort_keys=True, separators=(",", ":")),
                    len(batch.rows),
                    len(candles),
                ),
            )
            connection.executemany(
                """
                INSERT INTO historical_candle_revisions (
                    dataset_id, candle_id, instrument_id, timeframe, opens_at, closes_at,
                    open, high, low, close, volume, status, revision, source, provider,
                    source_revision, finalized_at, component_candle_ids, source_watermark,
                    supersedes_candle_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        dataset_id,
                        item.candle_id,
                        item.instrument_id,
                        item.timeframe.value,
                        _time(item.opens_at),
                        _time(item.closes_at),
                        _decimal(item.open),
                        _decimal(item.high),
                        _decimal(item.low),
                        _decimal(item.close),
                        _decimal(item.volume),
                        item.status.value,
                        item.revision,
                        item.source.value,
                        item.provider,
                        item.source_revision,
                        _time(item.finalized_at),
                        json.dumps(item.component_candle_ids, separators=(",", ":")),
                        item.source_watermark,
                        item.supersedes_candle_id,
                    )
                    for item in candles
                ],
            )
        return True

    def load_latest_candles(
        self,
        *,
        dataset_id: str,
        instrument_id: str,
        timeframe: Timeframe,
    ) -> tuple[Candle, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM (
                    SELECT c.*,
                        ROW_NUMBER() OVER (
                            PARTITION BY instrument_id, timeframe, opens_at
                            ORDER BY revision DESC
                        ) AS revision_rank
                    FROM historical_candle_revisions c
                    WHERE dataset_id = ? AND instrument_id = ? AND timeframe = ?
                ) ranked
                WHERE revision_rank = 1
                ORDER BY opens_at ASC
                """,
                (dataset_id, instrument_id, timeframe.value),
            ).fetchall()
        return tuple(_candle_from_row(row) for row in rows)

    def save_feature_rows(
        self,
        *,
        dataset_id: str,
        rows: tuple[PriceFeatureRow, ...],
    ) -> int:
        if not rows:
            return 0
        inserted = 0
        with self._connect() as connection:
            for row in rows:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO historical_feature_rows (
                        dataset_id, feature_version, feature_set_hash, source_candle_id,
                        instrument_id, timeframe, decision_time, is_ready, values_json,
                        blockers_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        dataset_id,
                        row.feature_version,
                        row.feature_set_hash,
                        row.source_candle_id,
                        row.instrument_id,
                        row.timeframe.value,
                        _time(row.decision_time),
                        int(row.is_ready),
                        json.dumps(row.contract_values(), sort_keys=True, separators=(",", ":")),
                        json.dumps(row.blockers, separators=(",", ":")),
                    ),
                )
                inserted += cursor.rowcount
        return inserted

    def count_feature_rows(
        self,
        *,
        dataset_id: str,
        feature_version: str,
        timeframe: Timeframe | None = None,
    ) -> int:
        with self._connect() as connection:
            if timeframe is None:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS row_count FROM historical_feature_rows
                    WHERE dataset_id = ? AND feature_version = ?
                    """,
                    (dataset_id, feature_version),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS row_count FROM historical_feature_rows
                    WHERE dataset_id = ? AND feature_version = ? AND timeframe = ?
                    """,
                    (dataset_id, feature_version, timeframe.value),
                ).fetchone()
        return int(row["row_count"])

    def save_ml_research_run(self, report: TrainingRunReport) -> bool:
        """Persist labels, predictions, and later assessments as append-only records."""
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM ml_research_runs WHERE run_id = ?",
                (report.run_id,),
            ).fetchone()
            if exists:
                return False
            connection.executemany(
                """
                INSERT OR IGNORE INTO ml_first_touch_labels (
                    label_id, dataset_id, label_version, label_definition_hash,
                    instrument_id, decision_candle_id, decision_time, reference_close,
                    atr_at_decision, up_barrier, down_barrier, window_ends_at, outcome,
                    first_touch_at, first_touch_candle_id, future_candle_ids, eligible,
                    exclusion_reason, label_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        label.label_id,
                        label.dataset_id,
                        label.label_version,
                        label.label_definition_hash,
                        label.instrument_id,
                        label.decision_candle_id,
                        _time(label.decision_time),
                        _decimal(label.reference_close),
                        _decimal(label.atr_at_decision),
                        _decimal(label.up_barrier),
                        _decimal(label.down_barrier),
                        _time(label.window_ends_at),
                        label.outcome.value if label.outcome else None,
                        _time(label.first_touch_at),
                        label.first_touch_candle_id,
                        json.dumps(label.future_candle_ids, separators=(",", ":")),
                        int(label.eligible),
                        label.exclusion_reason,
                        json.dumps(label.to_contract(), sort_keys=True, separators=(",", ":")),
                    )
                    for label in report.dataset_report.labels
                ],
            )
            connection.execute(
                """
                INSERT INTO ml_research_runs (
                    run_id, dataset_id, research_version, research_identity, created_at,
                    feature_version, feature_set_hash, label_version, label_definition_hash,
                    selected_research_candidate, calibrated, approved_for_live_inference,
                    config_json, report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    report.run_id,
                    report.dataset_id,
                    report.research_version,
                    report.research_identity,
                    _time(report.created_at),
                    report.feature_version,
                    report.feature_set_hash,
                    report.label_version,
                    report.label_definition_hash,
                    report.selected_research_candidate,
                    int(report.calibrated),
                    int(report.approved_for_live_inference),
                    json.dumps(report.config.to_contract(), sort_keys=True, separators=(",", ":")),
                    json.dumps(report.to_contract(), sort_keys=True, separators=(",", ":")),
                ),
            )
            for candidate in report.candidates:
                metrics_by_fold = dict(candidate.fold_metrics)
                for fold in report.folds:
                    connection.execute(
                        """
                        INSERT INTO ml_fold_results (
                            run_id, candidate_name, fold_index, train_starts_at,
                            train_ends_at, maximum_train_label_end, test_starts_at,
                            test_ends_at, embargo_cutoff, train_count, test_count,
                            metrics_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            report.run_id,
                            candidate.candidate_name,
                            fold.fold_index,
                            _time(fold.train_starts_at),
                            _time(fold.train_ends_at),
                            _time(fold.maximum_train_label_end),
                            _time(fold.test_starts_at),
                            _time(fold.test_ends_at),
                            _time(fold.embargo_cutoff),
                            len(fold.train_indices),
                            len(fold.test_indices),
                            json.dumps(
                                metrics_by_fold[fold.fold_index].to_contract(),
                                sort_keys=True,
                                separators=(",", ":"),
                            ),
                        ),
                    )
                connection.executemany(
                    """
                    INSERT INTO ml_replay_predictions (
                        prediction_id, run_id, model_id, candidate_name, fold_index,
                        sample_id, label_id, decision_time, data_as_of, generated_at,
                        input_revision_checksum, raw_probabilities_json,
                        predicted_outcome, calibration_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item.prediction_id,
                            item.run_id,
                            item.model_id,
                            item.candidate_name,
                            item.fold_index,
                            item.sample_id,
                            item.label_id,
                            _time(item.decision_time),
                            _time(item.decision_time),
                            _time(item.generated_at),
                            item.input_revision_checksum,
                            json.dumps(dict(item.raw_probabilities), sort_keys=True, separators=(",", ":")),
                            item.predicted_outcome.value,
                            item.calibration_status,
                        )
                        for item in candidate.predictions
                    ],
                )
            connection.executemany(
                """
                INSERT INTO ml_replay_assessments (
                    assessment_id, prediction_id, label_id, assessed_at,
                    actual_outcome, predicted_outcome, correct, assessment_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.assessment_id,
                        item.prediction_id,
                        item.label_id,
                        _time(item.assessed_at),
                        item.actual_outcome.value,
                        item.predicted_outcome.value,
                        int(item.correct),
                        json.dumps(item.to_contract(), sort_keys=True, separators=(",", ":")),
                    )
                    for item in report.replay_assessments
                ],
            )
        return True

    def load_step7_source(
        self,
        *,
        run_id: str,
    ) -> tuple[tuple[CalibrationObservation, ...], tuple[SignalReplayInput, ...]]:
        """Load immutable Step 6 OOS rows for only the selected research candidate."""
        with self._connect() as connection:
            run = connection.execute(
                "SELECT selected_research_candidate FROM ml_research_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise ValueError("ML research run does not exist")
            rows = connection.execute(
                """
                SELECT p.*, a.actual_outcome, l.instrument_id, l.reference_close,
                       l.atr_at_decision
                FROM ml_replay_predictions p
                JOIN ml_replay_assessments a ON a.prediction_id = p.prediction_id
                JOIN ml_first_touch_labels l ON l.label_id = p.label_id
                WHERE p.run_id = ? AND p.candidate_name = ?
                ORDER BY p.decision_time ASC, p.prediction_id ASC
                """,
                (run_id, run["selected_research_candidate"]),
            ).fetchall()
        observations = tuple(
            CalibrationObservation(
                prediction_id=row["prediction_id"],
                run_id=run_id,
                candidate_name=row["candidate_name"],
                fold_index=int(row["fold_index"]),
                decision_time=_parse_time(row["decision_time"]),
                raw_probabilities=tuple(
                    (name, float(value))
                    for name, value in sorted(json.loads(row["raw_probabilities_json"]).items())
                ),
                actual_outcome=TargetOutcome(row["actual_outcome"]),
            )
            for row in rows
        )
        replay_inputs = tuple(
            SignalReplayInput(
                prediction_id=row["prediction_id"],
                snapshot_id=f"historical-replay:{row['prediction_id']}",
                instrument_id=row["instrument_id"],
                decision_time=_parse_time(row["decision_time"]),
                input_revision_checksum=row["input_revision_checksum"],
                reference_close=Decimal(row["reference_close"]),
                atr=Decimal(row["atr_at_decision"]),
            )
            for row in rows
            if row["atr_at_decision"] is not None
        )
        return observations, replay_inputs

    def save_step7_research_report(self, report: Step7ResearchReport) -> bool:
        """Append calibration outputs and policy decisions without updating Step 6 rows."""
        calibration = report.calibration
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM calibration_runs WHERE calibration_id = ?",
                (calibration.calibration_id,),
            ).fetchone()
            if exists:
                return False
            connection.execute(
                """
                INSERT INTO calibration_runs (
                    calibration_id, source_run_id, candidate_name, calibration_version,
                    calibration_identity, created_at, fit_ends_at, evaluation_starts_at,
                    temperature, release_gate_passed, supported_bins_json, blockers_json,
                    report_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    calibration.calibration_id,
                    calibration.artifact.source_run_id,
                    calibration.artifact.candidate_name,
                    calibration.artifact.calibration_version,
                    calibration.artifact.calibration_identity,
                    _time(calibration.created_at),
                    _time(calibration.artifact.fit_ends_at),
                    _time(calibration.artifact.evaluation_starts_at),
                    calibration.artifact.temperature,
                    int(calibration.release_gate_passed),
                    json.dumps(calibration.supported_confidence_bins, separators=(",", ":")),
                    json.dumps(calibration.blockers, separators=(",", ":")),
                    json.dumps(calibration.to_contract(), sort_keys=True, separators=(",", ":")),
                ),
            )
            connection.executemany(
                """
                INSERT INTO calibrated_predictions (
                    calibrated_prediction_id, calibration_id, source_prediction_id,
                    decision_time, calibrated_probabilities_json, actual_outcome,
                    evaluation_partition
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.calibrated_prediction_id,
                        item.calibration_id,
                        item.source_prediction_id,
                        _time(item.decision_time),
                        json.dumps(
                            dict(item.calibrated_probabilities),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        item.actual_outcome.value,
                        int(item.evaluation_partition),
                    )
                    for item in calibration.predictions
                ],
            )
            connection.executemany(
                """
                INSERT INTO signal_decisions (
                    signal_id, calibration_id, source_prediction_id, snapshot_id,
                    instrument_id, decision_time, created_at, expires_at, direction,
                    lifecycle_status, probabilities_json, expected_atr, risk_levels_json,
                    blockers_json, signal_policy_version, risk_policy_version,
                    input_revision_checksum, decision_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.signal_id,
                        item.calibration_id,
                        item.prediction_id,
                        item.snapshot_id,
                        item.instrument_id,
                        _time(item.decision_time),
                        _time(item.created_at),
                        _time(item.expires_at),
                        item.direction.value,
                        item.lifecycle_status.value,
                        (
                            json.dumps(
                                dict(item.probabilities),
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            if item.probabilities
                            else None
                        ),
                        item.expected_atr,
                        (
                            json.dumps(
                                item.risk_levels.to_contract(),
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            if item.risk_levels
                            else None
                        ),
                        json.dumps(item.blockers, separators=(",", ":")),
                        item.signal_policy_version,
                        item.risk_policy_version,
                        item.input_revision_checksum,
                        json.dumps(item.to_contract(), sort_keys=True, separators=(",", ":")),
                    )
                    for item in report.decisions
                ],
            )
        return True

    def append_signal_lifecycle_event(self, event: SignalLifecycleEvent) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO signal_lifecycle_events (
                    event_id, signal_id, event_type, status, occurred_at,
                    observed_price, reason, event_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.signal_id,
                    event.event_type.value,
                    event.status.value,
                    _time(event.occurred_at),
                    _decimal(event.observed_price),
                    event.reason,
                    json.dumps(event.to_contract(), sort_keys=True, separators=(",", ":")),
                ),
            )
        return cursor.rowcount == 1

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_datasets (
                    dataset_id TEXT PRIMARY KEY,
                    instrument_id TEXT NOT NULL,
                    source_timeframe TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    source_label TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    acquired_at TEXT NOT NULL,
                    starts_at TEXT NOT NULL,
                    ends_at TEXT NOT NULL,
                    quality_status TEXT NOT NULL,
                    quality_json TEXT NOT NULL,
                    source_row_count INTEGER NOT NULL,
                    candle_revision_count INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_candle_revisions (
                    dataset_id TEXT NOT NULL,
                    candle_id TEXT NOT NULL,
                    instrument_id TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    opens_at TEXT NOT NULL,
                    closes_at TEXT NOT NULL,
                    open TEXT NOT NULL,
                    high TEXT NOT NULL,
                    low TEXT NOT NULL,
                    close TEXT NOT NULL,
                    volume TEXT,
                    status TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    source_revision INTEGER NOT NULL,
                    finalized_at TEXT,
                    component_candle_ids TEXT NOT NULL,
                    source_watermark TEXT NOT NULL,
                    supersedes_candle_id TEXT,
                    PRIMARY KEY (dataset_id, candle_id),
                    UNIQUE (dataset_id, instrument_id, timeframe, opens_at, revision),
                    FOREIGN KEY (dataset_id) REFERENCES historical_datasets(dataset_id)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS historical_candle_lookup_idx
                ON historical_candle_revisions (
                    dataset_id, instrument_id, timeframe, opens_at, revision
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_feature_rows (
                    dataset_id TEXT NOT NULL,
                    feature_version TEXT NOT NULL,
                    feature_set_hash TEXT NOT NULL,
                    source_candle_id TEXT NOT NULL,
                    instrument_id TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    decision_time TEXT NOT NULL,
                    is_ready INTEGER NOT NULL,
                    values_json TEXT NOT NULL,
                    blockers_json TEXT NOT NULL,
                    PRIMARY KEY (dataset_id, feature_version, source_candle_id),
                    FOREIGN KEY (dataset_id) REFERENCES historical_datasets(dataset_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_first_touch_labels (
                    label_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    label_version TEXT NOT NULL,
                    label_definition_hash TEXT NOT NULL,
                    instrument_id TEXT NOT NULL,
                    decision_candle_id TEXT NOT NULL,
                    decision_time TEXT NOT NULL,
                    reference_close TEXT NOT NULL,
                    atr_at_decision TEXT,
                    up_barrier TEXT,
                    down_barrier TEXT,
                    window_ends_at TEXT NOT NULL,
                    outcome TEXT,
                    first_touch_at TEXT,
                    first_touch_candle_id TEXT,
                    future_candle_ids TEXT NOT NULL,
                    eligible INTEGER NOT NULL,
                    exclusion_reason TEXT,
                    label_json TEXT NOT NULL,
                    UNIQUE (dataset_id, decision_candle_id, label_version),
                    FOREIGN KEY (dataset_id) REFERENCES historical_datasets(dataset_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_research_runs (
                    run_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    research_version TEXT NOT NULL,
                    research_identity TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    feature_version TEXT NOT NULL,
                    feature_set_hash TEXT NOT NULL,
                    label_version TEXT NOT NULL,
                    label_definition_hash TEXT NOT NULL,
                    selected_research_candidate TEXT NOT NULL,
                    calibrated INTEGER NOT NULL,
                    approved_for_live_inference INTEGER NOT NULL,
                    config_json TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    FOREIGN KEY (dataset_id) REFERENCES historical_datasets(dataset_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_fold_results (
                    run_id TEXT NOT NULL,
                    candidate_name TEXT NOT NULL,
                    fold_index INTEGER NOT NULL,
                    train_starts_at TEXT NOT NULL,
                    train_ends_at TEXT NOT NULL,
                    maximum_train_label_end TEXT NOT NULL,
                    test_starts_at TEXT NOT NULL,
                    test_ends_at TEXT NOT NULL,
                    embargo_cutoff TEXT NOT NULL,
                    train_count INTEGER NOT NULL,
                    test_count INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, candidate_name, fold_index),
                    FOREIGN KEY (run_id) REFERENCES ml_research_runs(run_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_replay_predictions (
                    prediction_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    candidate_name TEXT NOT NULL,
                    fold_index INTEGER NOT NULL,
                    sample_id TEXT NOT NULL,
                    label_id TEXT NOT NULL,
                    decision_time TEXT NOT NULL,
                    data_as_of TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    input_revision_checksum TEXT NOT NULL,
                    raw_probabilities_json TEXT NOT NULL,
                    predicted_outcome TEXT NOT NULL,
                    calibration_status TEXT NOT NULL CHECK (calibration_status = 'UNCALIBRATED'),
                    FOREIGN KEY (run_id) REFERENCES ml_research_runs(run_id),
                    FOREIGN KEY (label_id) REFERENCES ml_first_touch_labels(label_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_replay_assessments (
                    assessment_id TEXT PRIMARY KEY,
                    prediction_id TEXT NOT NULL,
                    label_id TEXT NOT NULL,
                    assessed_at TEXT NOT NULL,
                    actual_outcome TEXT NOT NULL,
                    predicted_outcome TEXT NOT NULL,
                    correct INTEGER NOT NULL,
                    assessment_json TEXT NOT NULL,
                    FOREIGN KEY (prediction_id) REFERENCES ml_replay_predictions(prediction_id),
                    FOREIGN KEY (label_id) REFERENCES ml_first_touch_labels(label_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS calibration_runs (
                    calibration_id TEXT PRIMARY KEY,
                    source_run_id TEXT NOT NULL,
                    candidate_name TEXT NOT NULL,
                    calibration_version TEXT NOT NULL,
                    calibration_identity TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    fit_ends_at TEXT NOT NULL,
                    evaluation_starts_at TEXT NOT NULL,
                    temperature REAL NOT NULL CHECK (temperature > 0),
                    release_gate_passed INTEGER NOT NULL,
                    supported_bins_json TEXT NOT NULL,
                    blockers_json TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    FOREIGN KEY (source_run_id) REFERENCES ml_research_runs(run_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS calibrated_predictions (
                    calibrated_prediction_id TEXT PRIMARY KEY,
                    calibration_id TEXT NOT NULL,
                    source_prediction_id TEXT NOT NULL,
                    decision_time TEXT NOT NULL,
                    calibrated_probabilities_json TEXT NOT NULL,
                    actual_outcome TEXT NOT NULL,
                    evaluation_partition INTEGER NOT NULL,
                    FOREIGN KEY (calibration_id) REFERENCES calibration_runs(calibration_id),
                    FOREIGN KEY (source_prediction_id) REFERENCES ml_replay_predictions(prediction_id),
                    UNIQUE (calibration_id, source_prediction_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_decisions (
                    signal_id TEXT PRIMARY KEY,
                    calibration_id TEXT NOT NULL,
                    source_prediction_id TEXT NOT NULL,
                    snapshot_id TEXT NOT NULL,
                    instrument_id TEXT NOT NULL,
                    decision_time TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    lifecycle_status TEXT NOT NULL,
                    probabilities_json TEXT,
                    expected_atr REAL,
                    risk_levels_json TEXT,
                    blockers_json TEXT NOT NULL,
                    signal_policy_version TEXT NOT NULL,
                    risk_policy_version TEXT NOT NULL,
                    input_revision_checksum TEXT NOT NULL,
                    decision_json TEXT NOT NULL,
                    FOREIGN KEY (calibration_id) REFERENCES calibration_runs(calibration_id),
                    FOREIGN KEY (source_prediction_id) REFERENCES ml_replay_predictions(prediction_id),
                    UNIQUE (calibration_id, source_prediction_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS signal_lifecycle_events (
                    event_id TEXT PRIMARY KEY,
                    signal_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    observed_price TEXT,
                    reason TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    FOREIGN KEY (signal_id) REFERENCES signal_decisions(signal_id)
                )
                """
            )
            for table in (
                "historical_datasets",
                "historical_candle_revisions",
                "historical_feature_rows",
                "ml_first_touch_labels",
                "ml_research_runs",
                "ml_fold_results",
                "ml_replay_predictions",
                "ml_replay_assessments",
                "calibration_runs",
                "calibrated_predictions",
                "signal_decisions",
                "signal_lifecycle_events",
            ):
                connection.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_no_update
                    BEFORE UPDATE ON {table}
                    BEGIN SELECT RAISE(ABORT, 'append-only table'); END
                    """
                )
                connection.execute(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_no_delete
                    BEFORE DELETE ON {table}
                    BEGIN SELECT RAISE(ABORT, 'append-only table'); END
                    """
                )


def _candle_from_row(row: sqlite3.Row) -> Candle:
    return Candle(
        schema_version=1,
        candle_id=row["candle_id"],
        instrument_id=row["instrument_id"],
        timeframe=Timeframe(row["timeframe"]),
        opens_at=_parse_time(row["opens_at"]),
        closes_at=_parse_time(row["closes_at"]),
        open=Decimal(row["open"]),
        high=Decimal(row["high"]),
        low=Decimal(row["low"]),
        close=Decimal(row["close"]),
        volume=Decimal(row["volume"]) if row["volume"] is not None else None,
        status=CandleStatus(row["status"]),
        revision=int(row["revision"]),
        source=CandleSource(row["source"]),
        provider=row["provider"],
        source_revision=int(row["source_revision"]),
        finalized_at=_parse_time(row["finalized_at"]) if row["finalized_at"] else None,
        component_candle_ids=tuple(json.loads(row["component_candle_ids"])),
        source_watermark=row["source_watermark"],
        supersedes_candle_id=row["supersedes_candle_id"],
    )


def _decimal(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None


def _time(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
