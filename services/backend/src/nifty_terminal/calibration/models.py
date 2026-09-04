"""Immutable calibration observations, artifacts, results, and gate reports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nifty_terminal.ml.models import MetricSummary, TargetOutcome


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    fit_fraction: float = 0.60
    minimum_total_predictions: int = 500
    minimum_fit_class_support: int = 50
    minimum_evaluation_class_support: int = 30
    minimum_supported_probability_bin: int = 30
    minimum_supported_probability_bins: int = 2
    maximum_ece: float = 0.05
    minimum_brier_skill: float = 0.0
    maximum_slice_ece: float = 0.10
    minimum_slice_brier_skill: float = -0.10
    minimum_slice_samples: int = 50

    def __post_init__(self) -> None:
        if not 0.50 <= self.fit_fraction < 0.90:
            raise ValueError("Calibration fit_fraction must be in [0.50, 0.90)")
        integer_values = (
            self.minimum_total_predictions,
            self.minimum_fit_class_support,
            self.minimum_evaluation_class_support,
            self.minimum_supported_probability_bin,
            self.minimum_supported_probability_bins,
            self.minimum_slice_samples,
        )
        if min(integer_values) < 1:
            raise ValueError("Calibration support requirements must be positive")
        if not 0.0 <= self.maximum_ece <= 1.0:
            raise ValueError("maximum_ece must be in [0, 1]")

    def to_contract(self) -> dict[str, object]:
        return {
            "fit_fraction": self.fit_fraction,
            "minimum_total_predictions": self.minimum_total_predictions,
            "minimum_fit_class_support": self.minimum_fit_class_support,
            "minimum_evaluation_class_support": self.minimum_evaluation_class_support,
            "minimum_supported_probability_bin": self.minimum_supported_probability_bin,
            "minimum_supported_probability_bins": self.minimum_supported_probability_bins,
            "maximum_ece": self.maximum_ece,
            "minimum_brier_skill": self.minimum_brier_skill,
            "maximum_slice_ece": self.maximum_slice_ece,
            "minimum_slice_brier_skill": self.minimum_slice_brier_skill,
            "minimum_slice_samples": self.minimum_slice_samples,
        }


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    prediction_id: str
    run_id: str
    candidate_name: str
    fold_index: int
    decision_time: datetime
    raw_probabilities: tuple[tuple[str, float], ...]
    actual_outcome: TargetOutcome


@dataclass(frozen=True, slots=True)
class TemperatureArtifact:
    schema_version: int
    calibration_id: str
    source_run_id: str
    candidate_name: str
    calibration_version: str
    calibration_identity: str
    fitted_at: datetime
    fit_ends_at: datetime
    evaluation_starts_at: datetime
    temperature: float

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "calibration_id": self.calibration_id,
            "source_run_id": self.source_run_id,
            "candidate_name": self.candidate_name,
            "calibration_version": self.calibration_version,
            "calibration_identity": self.calibration_identity,
            "fitted_at": _time(self.fitted_at),
            "fit_ends_at": _time(self.fit_ends_at),
            "evaluation_starts_at": _time(self.evaluation_starts_at),
            "method": "MULTICLASS_TEMPERATURE_SCALING",
            "temperature": self.temperature,
            "serialization": "SAFE_JSON_PARAMETERS_ONLY",
        }


@dataclass(frozen=True, slots=True)
class CalibratedPrediction:
    calibrated_prediction_id: str
    calibration_id: str
    source_prediction_id: str
    decision_time: datetime
    calibrated_probabilities: tuple[tuple[str, float], ...]
    actual_outcome: TargetOutcome
    evaluation_partition: bool

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "calibrated_prediction_id": self.calibrated_prediction_id,
            "calibration_id": self.calibration_id,
            "source_prediction_id": self.source_prediction_id,
            "decision_time": _time(self.decision_time),
            "calibrated_probabilities": dict(self.calibrated_probabilities),
            "actual_outcome": self.actual_outcome.value,
            "evaluation_partition": self.evaluation_partition,
        }


@dataclass(frozen=True, slots=True)
class SliceGate:
    name: str
    sample_count: int
    ece: float | None
    brier_skill: float | None
    passed: bool

    def to_contract(self) -> dict[str, object]:
        return {
            "name": self.name,
            "sample_count": self.sample_count,
            "ece": self.ece,
            "brier_skill": self.brier_skill,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    schema_version: int
    calibration_id: str
    created_at: datetime
    artifact: TemperatureArtifact
    config: CalibrationConfig
    total_observations: int
    fit_observations: int
    evaluation_observations: int
    raw_evaluation_metrics: MetricSummary
    calibrated_evaluation_metrics: MetricSummary
    prior_evaluation_metrics: MetricSummary
    brier_skill: float
    supported_confidence_bins: tuple[str, ...]
    slice_gates: tuple[SliceGate, ...]
    release_gate_passed: bool
    blockers: tuple[str, ...]
    predictions: tuple[CalibratedPrediction, ...]

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "calibration_id": self.calibration_id,
            "created_at": _time(self.created_at),
            "artifact": self.artifact.to_contract(),
            "config": self.config.to_contract(),
            "total_observations": self.total_observations,
            "fit_observations": self.fit_observations,
            "evaluation_observations": self.evaluation_observations,
            "raw_evaluation_metrics": self.raw_evaluation_metrics.to_contract(),
            "calibrated_evaluation_metrics": self.calibrated_evaluation_metrics.to_contract(),
            "prior_evaluation_metrics": self.prior_evaluation_metrics.to_contract(),
            "brier_skill": self.brier_skill,
            "supported_confidence_bins": list(self.supported_confidence_bins),
            "slice_gates": [item.to_contract() for item in self.slice_gates],
            "release_gate_passed": self.release_gate_passed,
            "precise_probability_display_allowed": self.release_gate_passed,
            "blockers": list(self.blockers),
            "predictions": [item.to_contract() for item in self.predictions],
        }


def _time(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
