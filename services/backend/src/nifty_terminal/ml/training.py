"""Deterministic baseline training over purged chronological folds."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import hashlib
import json
import platform
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nifty_terminal.features.definitions import FEATURE_SET_HASH, FEATURE_VERSION
from nifty_terminal.ml.definitions import (
    CLASS_ORDER,
    LABEL_DEFINITION_HASH,
    LABEL_VERSION,
    RANDOM_SEED,
    RESEARCH_IDENTITY,
    RESEARCH_VERSION,
)
from nifty_terminal.ml.metrics import calculate_metrics, prior_probabilities
from nifty_terminal.ml.models import (
    CandidateEvaluation,
    DatasetBuildReport,
    ResearchPrediction,
    TargetOutcome,
    TrainingSample,
    TrainingRunReport,
    WalkForwardConfig,
)
from nifty_terminal.ml.replay import build_replay_assessments
from nifty_terminal.ml.split import PurgedWalkForwardSplitter


class MLResearchRunner:
    """Trains fixed baselines; it never approves or calibrates a live model."""

    def run(
        self,
        *,
        dataset_report: DatasetBuildReport,
        config: WalkForwardConfig,
        created_at: datetime | None = None,
    ) -> TrainingRunReport:
        timestamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        samples = dataset_report.samples
        folds = PurgedWalkForwardSplitter().split(samples, config)
        run_id = _run_id(dataset_report, config, timestamp)
        baseline_actual: list[str] = []
        baseline_probabilities: list[np.ndarray] = []
        candidates: list[CandidateEvaluation] = []

        for candidate_name, factory in _candidate_factories():
            fold_metrics = []
            all_predictions: list[ResearchPrediction] = []
            fit_latency = 0.0
            inference_latency = 0.0
            for fold in folds:
                training = tuple(samples[index] for index in fold.train_indices)
                testing = tuple(samples[index] for index in fold.test_indices)
                train_x = np.asarray([item.feature_values for item in training], dtype=float)
                train_y = np.asarray([item.outcome.value for item in training])
                test_x = np.asarray([item.feature_values for item in testing], dtype=float)
                test_y = tuple(item.outcome.value for item in testing)

                if candidate_name == "multinomial_logistic":
                    baseline_actual.extend(test_y)
                    baseline_probabilities.append(
                        prior_probabilities(tuple(train_y.tolist()), len(testing))
                    )

                estimator = factory()
                started = perf_counter()
                estimator.fit(train_x, train_y)
                fit_latency += (perf_counter() - started) * 1_000
                started = perf_counter()
                raw = estimator.predict_proba(test_x)
                inference_latency += (perf_counter() - started) * 1_000
                probabilities = _ordered_probabilities(estimator, raw)
                metrics = calculate_metrics(test_y, probabilities)
                fold_metrics.append((fold.fold_index, metrics))
                model_id = _model_id(candidate_name, fold.fold_index, training)
                for row_index, sample in enumerate(testing):
                    probability_row = probabilities[row_index]
                    predicted = TargetOutcome(CLASS_ORDER[int(probability_row.argmax())])
                    prediction_id = str(
                        uuid5(
                            NAMESPACE_URL,
                            f"replay-prediction:{run_id}:{model_id}:{sample.sample_id}",
                        )
                    )
                    all_predictions.append(
                        ResearchPrediction(
                            prediction_id=prediction_id,
                            run_id=run_id,
                            model_id=model_id,
                            candidate_name=candidate_name,
                            fold_index=fold.fold_index,
                            sample_id=sample.sample_id,
                            label_id=sample.label_id,
                            decision_time=sample.decision_time,
                            label_window_end=sample.label_window_end,
                            generated_at=sample.decision_time,
                            input_revision_checksum=sample.input_revision_checksum,
                            raw_probabilities=tuple(
                                (name, float(probability_row[index]))
                                for index, name in enumerate(CLASS_ORDER)
                            ),
                            predicted_outcome=predicted,
                            actual_outcome=sample.outcome,
                        )
                    )

            aggregate_probabilities = np.asarray(
                [
                    [dict(item.raw_probabilities)[name] for name in CLASS_ORDER]
                    for item in all_predictions
                ],
                dtype=float,
            )
            aggregate_actual = tuple(item.actual_outcome.value for item in all_predictions)
            candidates.append(
                CandidateEvaluation(
                    candidate_name=candidate_name,
                    aggregate_metrics=calculate_metrics(
                        aggregate_actual, aggregate_probabilities
                    ),
                    fold_metrics=tuple(fold_metrics),
                    predictions=tuple(all_predictions),
                    fit_latency_ms=fit_latency,
                    inference_latency_ms=inference_latency,
                )
            )

        baseline_matrix = np.vstack(baseline_probabilities)
        baseline_metrics = calculate_metrics(tuple(baseline_actual), baseline_matrix)
        selected = min(
            candidates,
            key=lambda item: (
                item.aggregate_metrics.multiclass_brier,
                item.aggregate_metrics.log_loss,
                -item.aggregate_metrics.balanced_accuracy,
                item.candidate_name,
            ),
        )
        every_prediction = tuple(
            prediction for candidate in candidates for prediction in candidate.predictions
        )
        return TrainingRunReport(
            schema_version=1,
            run_id=run_id,
            research_version=RESEARCH_VERSION,
            research_identity=RESEARCH_IDENTITY,
            dataset_id=dataset_report.dataset_id,
            created_at=timestamp,
            feature_version=FEATURE_VERSION,
            feature_set_hash=FEATURE_SET_HASH,
            label_version=LABEL_VERSION,
            label_definition_hash=LABEL_DEFINITION_HASH,
            runtime_versions=(
                ("python", platform.python_version()),
                ("numpy", np.__version__),
                ("scikit_learn", sklearn.__version__),
            ),
            config=config,
            dataset_report=dataset_report,
            folds=folds,
            prior_baseline_metrics=baseline_metrics,
            candidates=tuple(candidates),
            selected_research_candidate=selected.candidate_name,
            replay_assessments=build_replay_assessments(every_prediction),
        )


def _candidate_factories() -> tuple[tuple[str, Callable[[], ClassifierMixin]], ...]:
    def logistic() -> Pipeline:
        return Pipeline(
            steps=(
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2_000,
                        random_state=RANDOM_SEED,
                        solver="lbfgs",
                    ),
                ),
            )
        )

    def histogram_boosting() -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            class_weight="balanced",
            early_stopping=False,
            l2_regularization=1.0,
            learning_rate=0.05,
            max_iter=160,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            random_state=RANDOM_SEED,
        )

    return (
        ("multinomial_logistic", logistic),
        ("hist_gradient_boosting", histogram_boosting),
    )


def _ordered_probabilities(
    estimator: ClassifierMixin,
    probabilities: np.ndarray,
) -> np.ndarray:
    classes = tuple(str(item) for item in estimator.classes_)  # type: ignore[attr-defined]
    if set(classes) != set(CLASS_ORDER):
        raise ValueError("Every training fold must contain all target classes")
    return probabilities[:, [classes.index(name) for name in CLASS_ORDER]]


def _model_id(
    candidate_name: str,
    fold_index: int,
    training: tuple[TrainingSample, ...],
) -> str:
    sample_ids = [item.sample_id for item in training]
    checksum = hashlib.sha256("|".join(sample_ids).encode("utf-8")).hexdigest()
    return str(
        uuid5(
            NAMESPACE_URL,
            f"research-model:{RESEARCH_IDENTITY}:{candidate_name}:{fold_index}:{checksum}",
        )
    )


def _run_id(
    dataset_report: DatasetBuildReport,
    config: WalkForwardConfig,
    created_at: datetime,
) -> str:
    identity = json.dumps(
        {
            "dataset_id": dataset_report.dataset_id,
            "research_identity": RESEARCH_IDENTITY,
            "config": config.to_contract(),
            "created_at": created_at.isoformat(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return str(uuid5(NAMESPACE_URL, f"training-run:{identity}"))
