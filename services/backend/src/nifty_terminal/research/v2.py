"""Nested chronological screening of label balance and probability-safe models."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import json
import platform

import numpy as np
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nifty_terminal.calibration.temperature import apply_temperature, fit_temperature
from nifty_terminal.ml.definitions import CLASS_ORDER, RANDOM_SEED
from nifty_terminal.ml.metrics import calculate_metrics, prior_probabilities
from nifty_terminal.ml.models import DatasetBuildReport, MetricSummary, WalkForwardConfig
from nifty_terminal.ml.split import PurgedWalkForwardSplitter


RESEARCH_V2_VERSION = "probability_research_v2.v1"
BARRIER_MULTIPLIERS = (Decimal("1.0"), Decimal("1.25"), Decimal("1.5"))
CANDIDATE_NAMES = (
    "multinomial_logistic_unweighted",
    "multinomial_logistic_balanced",
    "hist_gradient_boosting_unweighted",
    "hist_gradient_boosting_balanced",
)
NESTED_WALK_FORWARD_CONFIG = WalkForwardConfig(
    n_splits=5,
    minimum_train_samples=10_000,
    test_samples=2_000,
    purge_bars=12,
    embargo_bars=12,
    minimum_train_class_samples=25,
)
SCREENING_GATE_DEFINITION = {
    "minimum_full_class_support": 1_000,
    "minimum_final_class_support": 50,
    "minimum_final_balanced_accuracy": 1.0 / 3.0,
    "minimum_final_brier_skill_vs_prior": 0.0,
    "maximum_final_ece": 0.05,
    "calibrated_log_loss_must_beat_prior": True,
    "calibration_must_not_degrade_raw_proper_scores": True,
    "candidate_selection_folds": [0, 1, 2],
    "calibration_fit_fold": 3,
    "final_screening_fold": 4,
    "live_inference_approval": False,
}
RESEARCH_V2_IDENTITY = hashlib.sha256(
    json.dumps(
        {
            "version": RESEARCH_V2_VERSION,
            "barriers": [format(item, "f") for item in BARRIER_MULTIPLIERS],
            "walk_forward": NESTED_WALK_FORWARD_CONFIG.to_contract(),
            "gate": SCREENING_GATE_DEFINITION,
            "candidates": list(CANDIDATE_NAMES),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True, slots=True)
class CandidateScreening:
    name: str
    selection_metrics: MetricSummary
    fold_metrics: tuple[tuple[int, MetricSummary], ...]

    def to_contract(self) -> dict[str, object]:
        return {
            "name": self.name,
            "selection_metrics": self.selection_metrics.to_contract(),
            "fold_metrics": [
                {"fold_index": index, "metrics": metrics.to_contract()}
                for index, metrics in self.fold_metrics
            ],
        }


@dataclass(frozen=True, slots=True)
class TargetScreening:
    atr_multiplier: Decimal
    label_version: str
    label_definition_hash: str
    eligible_samples: int
    outcome_support: tuple[tuple[str, int], ...]
    selected_candidate: str
    candidates: tuple[CandidateScreening, ...]
    temperature: float
    selection_ends_at: str
    calibration_fit_starts_at: str
    calibration_fit_ends_at: str
    final_evaluation_starts_at: str
    final_evaluation_ends_at: str
    raw_final_metrics: MetricSummary
    calibrated_final_metrics: MetricSummary
    prior_final_metrics: MetricSummary
    brier_skill_vs_prior: float
    screening_gate_passed: bool
    blockers: tuple[str, ...]

    def to_contract(self) -> dict[str, object]:
        return {
            "atr_multiplier": format(self.atr_multiplier, "f"),
            "label_version": self.label_version,
            "label_definition_hash": self.label_definition_hash,
            "eligible_samples": self.eligible_samples,
            "outcome_support": dict(self.outcome_support),
            "selected_candidate": self.selected_candidate,
            "candidate_selection": [item.to_contract() for item in self.candidates],
            "nested_timeline": {
                "selection_ends_at": self.selection_ends_at,
                "calibration_fit_starts_at": self.calibration_fit_starts_at,
                "calibration_fit_ends_at": self.calibration_fit_ends_at,
                "final_evaluation_starts_at": self.final_evaluation_starts_at,
                "final_evaluation_ends_at": self.final_evaluation_ends_at,
            },
            "temperature": self.temperature,
            "raw_final_metrics": self.raw_final_metrics.to_contract(),
            "calibrated_final_metrics": self.calibrated_final_metrics.to_contract(),
            "prior_final_metrics": self.prior_final_metrics.to_contract(),
            "brier_skill_vs_prior": self.brier_skill_vs_prior,
            "screening_gate_passed": self.screening_gate_passed,
            "blockers": list(self.blockers),
            "approved_for_live_inference": False,
        }


@dataclass(frozen=True, slots=True)
class ResearchV2Report:
    dataset_id: str
    feature_version: str
    feature_set_hash: str
    targets: tuple[TargetScreening, ...]

    def to_contract(self) -> dict[str, object]:
        passing = tuple(item for item in self.targets if item.screening_gate_passed)
        leader_pool = passing or self.targets
        leader = min(
            leader_pool,
            key=_target_ranking_key,
        )
        return {
            "schema_version": 1,
            "research_version": RESEARCH_V2_VERSION,
            "research_identity": RESEARCH_V2_IDENTITY,
            "dataset_id": self.dataset_id,
            "feature_version": self.feature_version,
            "feature_set_hash": self.feature_set_hash,
            "runtime_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "walk_forward": NESTED_WALK_FORWARD_CONFIG.to_contract(),
            "screening_gate_definition": SCREENING_GATE_DEFINITION,
            "targets": [item.to_contract() for item in self.targets],
            "screening_leader": {
                "atr_multiplier": format(leader.atr_multiplier, "f"),
                "selected_candidate": leader.selected_candidate,
                "screening_gate_passed": leader.screening_gate_passed,
                "ranking_metric": "BRIER_SKILL_VS_TARGET_SPECIFIC_PRIOR",
                "status": "SCREENING_ONLY_REQUIRES_LOCKED_CONFIRMATION_RUN",
            },
            "target_selection_used_final_screening_results": True,
            "approved_for_live_inference": False,
            "official_signal_available": False,
            "automatic_trading_enabled": False,
            "news_used": False,
            "nifty_spot_volume_used": False,
        }


def _target_ranking_key(item: TargetScreening) -> tuple[float, float, float, Decimal]:
    """Compare different labels by normalized skill, never raw Brier scale."""
    return (
        -item.brier_skill_vs_prior,
        item.calibrated_final_metrics.log_loss,
        -item.calibrated_final_metrics.balanced_accuracy,
        item.atr_multiplier,
    )


def screen_target(dataset: DatasetBuildReport, multiplier: Decimal) -> TargetScreening:
    samples = dataset.samples
    folds = PurgedWalkForwardSplitter().split(samples, NESTED_WALK_FORWARD_CONFIG)
    actual_by_fold = {
        fold.fold_index: tuple(samples[index].outcome.value for index in fold.test_indices)
        for fold in folds
    }
    probabilities_by_candidate: dict[str, dict[int, np.ndarray]] = {}
    candidate_reports: list[CandidateScreening] = []

    for candidate_name, factory in _candidate_factories():
        by_fold: dict[int, np.ndarray] = {}
        fold_metrics = []
        for fold in folds:
            train_x = np.asarray(
                [samples[index].feature_values for index in fold.train_indices],
                dtype=float,
            )
            train_y = np.asarray(
                [samples[index].outcome.value for index in fold.train_indices]
            )
            test_x = np.asarray(
                [samples[index].feature_values for index in fold.test_indices],
                dtype=float,
            )
            estimator = factory()
            estimator.fit(train_x, train_y)
            probabilities = _ordered_probabilities(estimator, estimator.predict_proba(test_x))
            by_fold[fold.fold_index] = probabilities
            fold_metrics.append(
                (
                    fold.fold_index,
                    calculate_metrics(actual_by_fold[fold.fold_index], probabilities),
                )
            )
        selection_probabilities = np.vstack([by_fold[index] for index in (0, 1, 2)])
        selection_actual = tuple(
            outcome for index in (0, 1, 2) for outcome in actual_by_fold[index]
        )
        candidate_reports.append(
            CandidateScreening(
                name=candidate_name,
                selection_metrics=calculate_metrics(
                    selection_actual,
                    selection_probabilities,
                ),
                fold_metrics=tuple(fold_metrics),
            )
        )
        probabilities_by_candidate[candidate_name] = by_fold

    selected = min(
        candidate_reports,
        key=lambda item: (
            item.selection_metrics.multiclass_brier,
            item.selection_metrics.log_loss,
            -item.selection_metrics.balanced_accuracy,
            item.name,
        ),
    )
    selected_probabilities = probabilities_by_candidate[selected.name]
    fit_probabilities = selected_probabilities[3]
    fit_actual = actual_by_fold[3]
    class_index = {name: index for index, name in enumerate(CLASS_ORDER)}
    fit_indices = np.asarray([class_index[name] for name in fit_actual], dtype=int)
    temperature = fit_temperature(fit_probabilities, fit_indices)

    raw_final = selected_probabilities[4]
    calibrated_final = apply_temperature(raw_final, temperature)
    final_actual = actual_by_fold[4]
    final_fold = folds[4]
    training_actual = tuple(
        samples[index].outcome.value for index in final_fold.train_indices
    )
    prior_final = prior_probabilities(training_actual, len(final_actual))
    raw_metrics = calculate_metrics(final_actual, raw_final)
    calibrated_metrics = calculate_metrics(final_actual, calibrated_final)
    prior_metrics = calculate_metrics(final_actual, prior_final)
    brier_skill = 1.0 - calibrated_metrics.multiclass_brier / prior_metrics.multiclass_brier
    blockers = _screening_blockers(
        full_support=dict(dataset.outcome_support),
        final_support=Counter(final_actual),
        raw=raw_metrics,
        calibrated=calibrated_metrics,
        prior=prior_metrics,
        brier_skill=brier_skill,
    )
    labels = {item.label_version: item.label_definition_hash for item in dataset.labels}
    if len(labels) != 1:
        raise ValueError("One target-screening dataset must contain one label definition")
    label_version, label_hash = next(iter(labels.items()))

    return TargetScreening(
        atr_multiplier=multiplier,
        label_version=label_version,
        label_definition_hash=label_hash,
        eligible_samples=dataset.eligible_samples,
        outcome_support=dataset.outcome_support,
        selected_candidate=selected.name,
        candidates=tuple(candidate_reports),
        temperature=temperature,
        selection_ends_at=folds[2].test_ends_at.isoformat(),
        calibration_fit_starts_at=folds[3].test_starts_at.isoformat(),
        calibration_fit_ends_at=folds[3].test_ends_at.isoformat(),
        final_evaluation_starts_at=folds[4].test_starts_at.isoformat(),
        final_evaluation_ends_at=folds[4].test_ends_at.isoformat(),
        raw_final_metrics=raw_metrics,
        calibrated_final_metrics=calibrated_metrics,
        prior_final_metrics=prior_metrics,
        brier_skill_vs_prior=brier_skill,
        screening_gate_passed=not blockers,
        blockers=tuple(blockers),
    )


def _screening_blockers(
    *,
    full_support: dict[str, int],
    final_support: Counter[str],
    raw: MetricSummary,
    calibrated: MetricSummary,
    prior: MetricSummary,
    brier_skill: float,
) -> list[str]:
    blockers = []
    if any(full_support.get(name, 0) < 1_000 for name in CLASS_ORDER):
        blockers.append("INSUFFICIENT_FULL_TARGET_CLASS_SUPPORT")
    if any(final_support[name] < 50 for name in CLASS_ORDER):
        blockers.append("INSUFFICIENT_FINAL_SCREENING_CLASS_SUPPORT")
    if calibrated.balanced_accuracy <= 1.0 / 3.0:
        blockers.append("FINAL_BALANCED_ACCURACY_GATE_FAILED")
    if brier_skill <= 0.0:
        blockers.append("FINAL_BRIER_SKILL_GATE_FAILED")
    if calibrated.log_loss >= prior.log_loss:
        blockers.append("FINAL_LOG_LOSS_GATE_FAILED")
    if calibrated.raw_ece_10_bin > 0.05:
        blockers.append("FINAL_ECE_GATE_FAILED")
    if (
        calibrated.multiclass_brier > raw.multiclass_brier
        or calibrated.log_loss > raw.log_loss
    ):
        blockers.append("OUT_OF_TIME_CALIBRATION_DEGRADATION")
    return blockers


def _candidate_factories() -> tuple[
    tuple[str, Callable[[], ClassifierMixin]], ...
]:
    def logistic(class_weight: str | None) -> Pipeline:
        return Pipeline(
            steps=(
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight=class_weight,
                        max_iter=2_000,
                        random_state=RANDOM_SEED,
                        solver="lbfgs",
                    ),
                ),
            )
        )

    def histogram(class_weight: str | None) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            class_weight=class_weight,
            early_stopping=False,
            l2_regularization=1.0,
            learning_rate=0.05,
            max_iter=160,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            random_state=RANDOM_SEED,
        )

    return (
        ("multinomial_logistic_unweighted", lambda: logistic(None)),
        ("multinomial_logistic_balanced", lambda: logistic("balanced")),
        ("hist_gradient_boosting_unweighted", lambda: histogram(None)),
        ("hist_gradient_boosting_balanced", lambda: histogram("balanced")),
    )


def _ordered_probabilities(
    estimator: ClassifierMixin,
    probabilities: np.ndarray,
) -> np.ndarray:
    classes = tuple(str(item) for item in estimator.classes_)  # type: ignore[attr-defined]
    if set(classes) != set(CLASS_ORDER):
        raise ValueError("Every research-v2 fold must contain all target classes")
    return probabilities[:, [classes.index(name) for name in CLASS_ORDER]]
