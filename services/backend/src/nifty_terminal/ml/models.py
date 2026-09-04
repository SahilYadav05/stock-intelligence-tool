"""Immutable contracts for labels, training, evaluation, and replay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum


class TargetOutcome(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    NEITHER = "NEITHER"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True, slots=True)
class FirstTouchLabel:
    schema_version: int
    label_id: str
    label_version: str
    label_definition_hash: str
    dataset_id: str
    instrument_id: str
    decision_candle_id: str
    decision_time: datetime
    reference_close: Decimal
    atr_at_decision: Decimal | None
    up_barrier: Decimal | None
    down_barrier: Decimal | None
    window_ends_at: datetime
    outcome: TargetOutcome | None
    first_touch_at: datetime | None
    first_touch_candle_id: str | None
    future_candle_ids: tuple[str, ...]
    eligible: bool
    exclusion_reason: str | None

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "label_id": self.label_id,
            "label_version": self.label_version,
            "label_definition_hash": self.label_definition_hash,
            "dataset_id": self.dataset_id,
            "instrument_id": self.instrument_id,
            "decision_candle_id": self.decision_candle_id,
            "decision_time": _time(self.decision_time),
            "reference_close": _decimal(self.reference_close),
            "atr_at_decision": _decimal(self.atr_at_decision),
            "up_barrier": _decimal(self.up_barrier),
            "down_barrier": _decimal(self.down_barrier),
            "window_ends_at": _time(self.window_ends_at),
            "outcome": self.outcome.value if self.outcome else None,
            "first_touch_at": _time(self.first_touch_at),
            "first_touch_candle_id": self.first_touch_candle_id,
            "future_candle_ids": list(self.future_candle_ids),
            "eligible": self.eligible,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True, slots=True)
class TrainingSample:
    sample_id: str
    dataset_id: str
    instrument_id: str
    decision_time: datetime
    label_window_end: datetime
    label_id: str
    outcome: TargetOutcome
    primary_candle_id: str
    context_15m_candle_id: str
    context_1h_candle_id: str
    input_revision_checksum: str
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class DatasetBuildReport:
    dataset_id: str
    candidate_decisions: int
    eligible_samples: int
    outcome_support: tuple[tuple[str, int], ...]
    ambiguous_labels: int
    unavailable_labels: int
    excluded_feature_rows: int
    exclusion_counts: tuple[tuple[str, int], ...]
    feature_names: tuple[str, ...]
    labels: tuple[FirstTouchLabel, ...]
    samples: tuple[TrainingSample, ...]

    def summary_contract(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "candidate_decisions": self.candidate_decisions,
            "eligible_samples": self.eligible_samples,
            "outcome_support": dict(self.outcome_support),
            "ambiguous_labels": self.ambiguous_labels,
            "unavailable_labels": self.unavailable_labels,
            "excluded_feature_rows": self.excluded_feature_rows,
            "exclusion_counts": dict(self.exclusion_counts),
            "feature_count": len(self.feature_names),
            "feature_names": list(self.feature_names),
        }


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    n_splits: int = 4
    minimum_train_samples: int = 1_500
    test_samples: int = 250
    purge_bars: int = 12
    embargo_bars: int = 12
    minimum_train_class_samples: int = 25

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("Walk-forward validation requires at least two folds")
        if min(
            self.minimum_train_samples,
            self.test_samples,
            self.purge_bars,
            self.embargo_bars,
            self.minimum_train_class_samples,
        ) < 1:
            raise ValueError("Walk-forward configuration values must be positive")

    def to_contract(self) -> dict[str, int]:
        return {
            "n_splits": self.n_splits,
            "minimum_train_samples": self.minimum_train_samples,
            "test_samples": self.test_samples,
            "purge_bars": self.purge_bars,
            "embargo_bars": self.embargo_bars,
            "minimum_train_class_samples": self.minimum_train_class_samples,
        }


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold_index: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_starts_at: datetime
    train_ends_at: datetime
    maximum_train_label_end: datetime
    test_starts_at: datetime
    test_ends_at: datetime
    embargo_cutoff: datetime
    train_class_support: tuple[tuple[str, int], ...]

    def to_contract(self) -> dict[str, object]:
        return {
            "fold_index": self.fold_index,
            "train_count": len(self.train_indices),
            "test_count": len(self.test_indices),
            "train_starts_at": _time(self.train_starts_at),
            "train_ends_at": _time(self.train_ends_at),
            "maximum_train_label_end": _time(self.maximum_train_label_end),
            "test_starts_at": _time(self.test_starts_at),
            "test_ends_at": _time(self.test_ends_at),
            "embargo_cutoff": _time(self.embargo_cutoff),
            "train_class_support": dict(self.train_class_support),
        }


@dataclass(frozen=True, slots=True)
class MetricSummary:
    sample_count: int
    accuracy: float
    balanced_accuracy: float
    multiclass_brier: float
    log_loss: float
    raw_ece_10_bin: float
    class_support: tuple[tuple[str, int], ...]
    class_recall: tuple[tuple[str, float], ...]

    def to_contract(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "multiclass_brier": self.multiclass_brier,
            "log_loss": self.log_loss,
            "raw_ece_10_bin": self.raw_ece_10_bin,
            "class_support": dict(self.class_support),
            "class_recall": dict(self.class_recall),
        }


@dataclass(frozen=True, slots=True)
class ResearchPrediction:
    prediction_id: str
    run_id: str
    model_id: str
    candidate_name: str
    fold_index: int
    sample_id: str
    label_id: str
    decision_time: datetime
    label_window_end: datetime
    generated_at: datetime
    input_revision_checksum: str
    raw_probabilities: tuple[tuple[str, float], ...]
    predicted_outcome: TargetOutcome
    actual_outcome: TargetOutcome
    calibration_status: str = "UNCALIBRATED"

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "prediction_id": self.prediction_id,
            "run_id": self.run_id,
            "model_id": self.model_id,
            "candidate_name": self.candidate_name,
            "fold_index": self.fold_index,
            "sample_id": self.sample_id,
            "label_id": self.label_id,
            "decision_time": _time(self.decision_time),
            "data_as_of": _time(self.decision_time),
            "generated_at": _time(self.generated_at),
            "input_revision_checksum": self.input_revision_checksum,
            "raw_probabilities": dict(self.raw_probabilities),
            "predicted_outcome": self.predicted_outcome.value,
            "calibration_status": self.calibration_status,
            "official_signal": None,
        }


@dataclass(frozen=True, slots=True)
class ReplayAssessment:
    assessment_id: str
    prediction_id: str
    label_id: str
    assessed_at: datetime
    actual_outcome: TargetOutcome
    predicted_outcome: TargetOutcome
    correct: bool

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "assessment_id": self.assessment_id,
            "prediction_id": self.prediction_id,
            "label_id": self.label_id,
            "assessed_at": _time(self.assessed_at),
            "actual_outcome": self.actual_outcome.value,
            "predicted_outcome": self.predicted_outcome.value,
            "correct": self.correct,
        }


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate_name: str
    aggregate_metrics: MetricSummary
    fold_metrics: tuple[tuple[int, MetricSummary], ...]
    predictions: tuple[ResearchPrediction, ...]
    fit_latency_ms: float
    inference_latency_ms: float

    def summary_contract(self) -> dict[str, object]:
        return {
            "candidate_name": self.candidate_name,
            "aggregate_metrics": self.aggregate_metrics.to_contract(),
            "fold_metrics": [
                {"fold_index": index, "metrics": metrics.to_contract()}
                for index, metrics in self.fold_metrics
            ],
            "fit_latency_ms": self.fit_latency_ms,
            "inference_latency_ms": self.inference_latency_ms,
            "prediction_count": len(self.predictions),
            "probability_status": "RAW_UNCALIBRATED_RESEARCH_ONLY",
        }


@dataclass(frozen=True, slots=True)
class TrainingRunReport:
    schema_version: int
    run_id: str
    research_version: str
    research_identity: str
    dataset_id: str
    created_at: datetime
    feature_version: str
    feature_set_hash: str
    label_version: str
    label_definition_hash: str
    runtime_versions: tuple[tuple[str, str], ...]
    config: WalkForwardConfig
    dataset_report: DatasetBuildReport
    folds: tuple[WalkForwardFold, ...]
    prior_baseline_metrics: MetricSummary
    candidates: tuple[CandidateEvaluation, ...]
    selected_research_candidate: str
    replay_assessments: tuple[ReplayAssessment, ...]
    approved_for_live_inference: bool = False
    calibrated: bool = False

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "research_version": self.research_version,
            "research_identity": self.research_identity,
            "dataset_id": self.dataset_id,
            "created_at": _time(self.created_at),
            "feature_version": self.feature_version,
            "feature_set_hash": self.feature_set_hash,
            "label_version": self.label_version,
            "label_definition_hash": self.label_definition_hash,
            "runtime_versions": dict(self.runtime_versions),
            "config": self.config.to_contract(),
            "dataset": self.dataset_report.summary_contract(),
            "folds": [item.to_contract() for item in self.folds],
            "prior_baseline_metrics": self.prior_baseline_metrics.to_contract(),
            "candidates": [item.summary_contract() for item in self.candidates],
            "selected_research_candidate": self.selected_research_candidate,
            "replay_prediction_count": sum(len(item.predictions) for item in self.candidates),
            "replay_assessment_count": len(self.replay_assessments),
            "approved_for_live_inference": self.approved_for_live_inference,
            "calibrated": self.calibrated,
            "official_signal_available": False,
        }


@dataclass(frozen=True, slots=True)
class MLResearchPipelineResult:
    report: TrainingRunReport
    persisted: bool


def _time(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def _decimal(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None
