"""Conservative evidence gate for one chronological real-data experiment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from nifty_terminal.calibration.models import CalibrationReport
from nifty_terminal.ml.definitions import CLASS_ORDER
from nifty_terminal.ml.models import CandidateEvaluation, TrainingRunReport


REAL_DATA_RESEARCH_GATE_VERSION = "real_data_research_gate.v1"
REAL_DATA_RESEARCH_GATE_DEFINITION = {
    "minimum_eligible_samples": 15_000,
    "minimum_oos_predictions": 7_500,
    "minimum_class_support": 1_000,
    "minimum_folds": 5,
    "minimum_positive_balanced_accuracy_folds": 4,
    "minimum_balanced_accuracy": 1.0 / 3.0,
    "minimum_brier_skill_vs_prior": 0.0,
    "require_log_loss_better_than_prior": True,
    "require_calibration_release_gate": True,
    "live_inference_approval": False,
}
REAL_DATA_RESEARCH_GATE_IDENTITY = hashlib.sha256(
    json.dumps(
        REAL_DATA_RESEARCH_GATE_DEFINITION,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True, slots=True)
class RealDataResearchGate:
    schema_version: int
    gate_version: str
    gate_identity: str
    passed: bool
    blockers: tuple[str, ...]
    selected_candidate: str
    oos_predictions: int
    brier_skill_vs_prior: float
    positive_balanced_accuracy_folds: int
    total_folds: int

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "gate_version": self.gate_version,
            "gate_identity": self.gate_identity,
            "definition": REAL_DATA_RESEARCH_GATE_DEFINITION,
            "passed": self.passed,
            "blockers": list(self.blockers),
            "selected_candidate": self.selected_candidate,
            "oos_predictions": self.oos_predictions,
            "brier_skill_vs_prior": self.brier_skill_vs_prior,
            "positive_balanced_accuracy_folds": self.positive_balanced_accuracy_folds,
            "total_folds": self.total_folds,
            "approved_for_live_inference": False,
        }


@dataclass(frozen=True, slots=True)
class RealDataResearchResult:
    training: TrainingRunReport
    calibration: CalibrationReport
    gate: RealDataResearchGate
    training_persisted: bool
    calibration_persisted: bool

    def summary_contract(self) -> dict[str, object]:
        selected = _selected_candidate(self.training)
        return {
            "schema_version": 1,
            "dataset_id": self.training.dataset_id,
            "run_id": self.training.run_id,
            "calibration_id": self.calibration.calibration_id,
            "feature_version": self.training.feature_version,
            "label_version": self.training.label_version,
            "selected_candidate": selected.candidate_name,
            "dataset": self.training.dataset_report.summary_contract(),
            "prior_baseline_metrics": self.training.prior_baseline_metrics.to_contract(),
            "selected_candidate_metrics": selected.aggregate_metrics.to_contract(),
            "calibration": {
                "release_gate_passed": self.calibration.release_gate_passed,
                "blockers": list(self.calibration.blockers),
                "temperature": self.calibration.artifact.temperature,
                "fit_observations": self.calibration.fit_observations,
                "evaluation_observations": self.calibration.evaluation_observations,
                "raw_evaluation_metrics": self.calibration.raw_evaluation_metrics.to_contract(),
                "calibrated_evaluation_metrics": (
                    self.calibration.calibrated_evaluation_metrics.to_contract()
                ),
                "prior_evaluation_metrics": self.calibration.prior_evaluation_metrics.to_contract(),
                "brier_skill": self.calibration.brier_skill,
                "supported_confidence_bins": list(
                    self.calibration.supported_confidence_bins
                ),
            },
            "research_gate": self.gate.to_contract(),
            "persistence": {
                "training": self.training_persisted,
                "calibration": self.calibration_persisted,
            },
            "market_inputs": {
                "finalized_5m_primary": True,
                "finalized_15m_context": True,
                "finalized_1h_context": True,
                "developing_candle_used": False,
                "nifty_spot_volume_used": False,
                "vwap_used": False,
                "news_used": False,
            },
            "news_status": "NOT_INCLUDED_IN_STEP_14",
            "official_signal_available": False,
            "approved_for_live_inference": False,
            "automatic_trading_enabled": False,
        }


def evaluate_real_data_research(
    *,
    training: TrainingRunReport,
    calibration: CalibrationReport,
) -> RealDataResearchGate:
    selected = _selected_candidate(training)
    prior = training.prior_baseline_metrics
    candidate = selected.aggregate_metrics
    brier_skill = (
        1.0 - candidate.multiclass_brier / prior.multiclass_brier
        if prior.multiclass_brier > 0
        else float("-inf")
    )
    positive_folds = sum(
        metrics.balanced_accuracy
        > REAL_DATA_RESEARCH_GATE_DEFINITION["minimum_balanced_accuracy"]
        for _, metrics in selected.fold_metrics
    )
    support = dict(training.dataset_report.outcome_support)
    blockers: list[str] = []

    if training.dataset_report.eligible_samples < 15_000:
        blockers.append("INSUFFICIENT_ELIGIBLE_REAL_DATA_SAMPLES")
    if len(selected.predictions) < 7_500:
        blockers.append("INSUFFICIENT_OUT_OF_SAMPLE_PREDICTIONS")
    if any(support.get(name, 0) < 1_000 for name in CLASS_ORDER):
        blockers.append("INSUFFICIENT_TARGET_CLASS_SUPPORT")
    if len(training.folds) < 5:
        blockers.append("INSUFFICIENT_CHRONOLOGICAL_FOLDS")
    if positive_folds < 4:
        blockers.append("BALANCED_ACCURACY_NOT_STABLE_ACROSS_FOLDS")
    if candidate.balanced_accuracy <= 1.0 / 3.0:
        blockers.append("BALANCED_ACCURACY_DOES_NOT_BEAT_RANDOM_CLASS_BASELINE")
    if brier_skill <= 0.0:
        blockers.append("NO_POSITIVE_BRIER_SKILL_VS_PRIOR")
    if candidate.log_loss >= prior.log_loss:
        blockers.append("LOG_LOSS_DOES_NOT_BEAT_PRIOR")
    if not calibration.release_gate_passed:
        blockers.append("CALIBRATION_RELEASE_GATE_NOT_PASSED")

    return RealDataResearchGate(
        schema_version=1,
        gate_version=REAL_DATA_RESEARCH_GATE_VERSION,
        gate_identity=REAL_DATA_RESEARCH_GATE_IDENTITY,
        passed=not blockers,
        blockers=tuple(blockers),
        selected_candidate=selected.candidate_name,
        oos_predictions=len(selected.predictions),
        brier_skill_vs_prior=brier_skill,
        positive_balanced_accuracy_folds=positive_folds,
        total_folds=len(training.folds),
    )


def _selected_candidate(training: TrainingRunReport) -> CandidateEvaluation:
    for candidate in training.candidates:
        if candidate.candidate_name == training.selected_research_candidate:
            return candidate
    raise ValueError("Selected research candidate is missing from the training report")
