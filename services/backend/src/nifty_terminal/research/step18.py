"""Step 18 hierarchical model-v2 research with explicit collapse diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import hashlib
import json
import platform

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nifty_terminal.features.enhanced import (
    ENHANCED_FEATURE_SET_HASH,
    ENHANCED_FEATURE_VERSION,
    enhance_dataset,
)
from nifty_terminal.ml.definitions import CLASS_ORDER, RANDOM_SEED
from nifty_terminal.ml.metrics import calculate_metrics, prior_probabilities
from nifty_terminal.ml.models import DatasetBuildReport, MetricSummary, TrainingSample
from nifty_terminal.ml.split import PurgedWalkForwardSplitter
from nifty_terminal.research.step16 import (
    CALIBRATION_METHODS,
    CalibrationArtifactV2,
    apply_calibrator,
    fit_calibrator,
)
from nifty_terminal.research.v2 import NESTED_WALK_FORWARD_CONFIG


STEP18_VERSION = "hierarchical_model_v2_research.v1"
ARCHITECTURE_SELECTION_FOLDS = (0, 1)
CALIBRATION_FIT_FOLD = 2
CALIBRATION_SELECTION_FOLD = 3
HISTORICAL_DIAGNOSTIC_FOLD = 4
CANDIDATE_NAMES = (
    "direct_logistic_v1",
    "direct_logistic_v2",
    "direct_logistic_balanced_v2",
    "hierarchical_logistic_v2",
    "hierarchical_logistic_balanced_v2",
    "hierarchical_hgb_balanced_v2",
)
COLLAPSE_GATE = {
    "minimum_up_prediction_share": 0.05,
    "minimum_down_prediction_share": 0.05,
    "minimum_up_recall": 0.10,
    "minimum_down_recall": 0.10,
}
FINAL_RESEARCH_GATE = {
    **COLLAPSE_GATE,
    "minimum_balanced_accuracy": 1.0 / 3.0,
    "minimum_brier_skill_vs_prior": 0.0,
    "maximum_ece": 0.05,
    "log_loss_must_beat_prior": True,
    "calibration_must_not_degrade_raw_proper_scores": True,
    "live_inference_approval": False,
}
RESEARCH_IDENTITY = hashlib.sha256(
    json.dumps(
        {
            "version": STEP18_VERSION,
            "features": ENHANCED_FEATURE_SET_HASH,
            "candidates": CANDIDATE_NAMES,
            "walk_forward": NESTED_WALK_FORWARD_CONFIG.to_contract(),
            "timeline": {
                "architecture_selection": ARCHITECTURE_SELECTION_FOLDS,
                "calibration_fit": CALIBRATION_FIT_FOLD,
                "calibration_selection": CALIBRATION_SELECTION_FOLD,
                "historical_diagnostic": HISTORICAL_DIAGNOSTIC_FOLD,
            },
            "gate": FINAL_RESEARCH_GATE,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def run_model_v2_research(dataset: DatasetBuildReport) -> dict[str, object]:
    """Compare causal architectures without changing any deployed runtime artifact."""
    enhanced = enhance_dataset(dataset)
    samples = enhanced.samples
    folds = PurgedWalkForwardSplitter().split(samples, NESTED_WALK_FORWARD_CONFIG)
    actual = {
        fold.fold_index: tuple(samples[index].outcome.value for index in fold.test_indices)
        for fold in folds
    }
    all_probabilities: dict[str, dict[int, np.ndarray]] = {}
    candidate_reports = []
    for candidate_name in CANDIDATE_NAMES:
        by_fold = {}
        fold_reports = []
        for fold in folds:
            probabilities = _fit_predict(
                candidate_name,
                tuple(samples[index] for index in fold.train_indices),
                tuple(samples[index] for index in fold.test_indices),
            )
            by_fold[fold.fold_index] = probabilities
            fold_reports.append(
                {
                    "fold_index": fold.fold_index,
                    "metrics": calculate_metrics(
                        actual[fold.fold_index], probabilities
                    ).to_contract(),
                    "collapse_diagnostics": collapse_diagnostics(
                        actual[fold.fold_index], probabilities
                    ),
                }
            )
        selection_probabilities = np.vstack(
            [by_fold[index] for index in ARCHITECTURE_SELECTION_FOLDS]
        )
        selection_actual = tuple(
            value
            for index in ARCHITECTURE_SELECTION_FOLDS
            for value in actual[index]
        )
        selection_metrics = calculate_metrics(selection_actual, selection_probabilities)
        diagnostics = collapse_diagnostics(selection_actual, selection_probabilities)
        blockers = collapse_blockers(diagnostics)
        candidate_reports.append(
            {
                "name": candidate_name,
                "architecture": (
                    "HIERARCHICAL_ACTIONABLE_THEN_DIRECTION"
                    if candidate_name.startswith("hierarchical")
                    else "DIRECT_MULTICLASS"
                ),
                "feature_version": (
                    "price_features.v1"
                    if candidate_name == "direct_logistic_v1"
                    else ENHANCED_FEATURE_VERSION
                ),
                "selection_metrics": selection_metrics.to_contract(),
                "selection_collapse_diagnostics": diagnostics,
                "selection_blockers": blockers,
                "selection_viable": not blockers,
                "folds": fold_reports,
            }
        )
        all_probabilities[candidate_name] = by_fold

    viable = [item for item in candidate_reports if item["selection_viable"]]
    pool = viable or candidate_reports
    selected_report = min(pool, key=_candidate_ranking_key)
    selected_name = str(selected_report["name"])
    selected_probabilities = all_probabilities[selected_name]

    calibration_prior = _prior_vector(
        tuple(samples[index].outcome.value for index in folds[CALIBRATION_FIT_FOLD].train_indices)
    )
    calibration_reports = []
    fitted_calibrators: dict[str, CalibrationArtifactV2] = {}
    raw_calibration_selection = calculate_metrics(
        actual[CALIBRATION_SELECTION_FOLD],
        selected_probabilities[CALIBRATION_SELECTION_FOLD],
    )
    for method in CALIBRATION_METHODS:
        artifact = fit_calibrator(
            method=method,
            probabilities=selected_probabilities[CALIBRATION_FIT_FOLD],
            actual=actual[CALIBRATION_FIT_FOLD],
            prior=calibration_prior,
        )
        fitted_calibrators[method] = artifact
        transformed = apply_calibrator(
            artifact, selected_probabilities[CALIBRATION_SELECTION_FOLD]
        )
        metrics = calculate_metrics(actual[CALIBRATION_SELECTION_FOLD], transformed)
        diagnostics = collapse_diagnostics(actual[CALIBRATION_SELECTION_FOLD], transformed)
        blockers = collapse_blockers(diagnostics)
        if metrics.multiclass_brier > raw_calibration_selection.multiclass_brier:
            blockers.append("CALIBRATION_BRIER_DEGRADATION")
        if metrics.log_loss > raw_calibration_selection.log_loss:
            blockers.append("CALIBRATION_LOG_LOSS_DEGRADATION")
        calibration_reports.append(
            {
                "method": method,
                "metrics": metrics.to_contract(),
                "collapse_diagnostics": diagnostics,
                "selection_blockers": blockers,
                "selection_viable": not blockers,
                "artifact": artifact.to_contract(),
            }
        )
    viable_calibrations = [
        item for item in calibration_reports if item["selection_viable"]
    ]
    calibration_pool = viable_calibrations or [
        item for item in calibration_reports if item["method"] == "identity"
    ]
    selected_calibration_report = min(
        calibration_pool,
        key=lambda item: (
            item["metrics"]["multiclass_brier"],
            item["metrics"]["log_loss"],
            item["metrics"]["raw_ece_10_bin"],
            item["method"],
        ),
    )
    selected_calibration = fitted_calibrators[str(selected_calibration_report["method"])]

    diagnostic_fold = folds[HISTORICAL_DIAGNOSTIC_FOLD]
    raw_final = selected_probabilities[HISTORICAL_DIAGNOSTIC_FOLD]
    calibrated_final = apply_calibrator(selected_calibration, raw_final)
    final_actual = actual[HISTORICAL_DIAGNOSTIC_FOLD]
    final_prior_probabilities = prior_probabilities(
        tuple(samples[index].outcome.value for index in diagnostic_fold.train_indices),
        len(diagnostic_fold.test_indices),
    )
    raw_metrics = calculate_metrics(final_actual, raw_final)
    calibrated_metrics = calculate_metrics(final_actual, calibrated_final)
    prior_metrics = calculate_metrics(final_actual, final_prior_probabilities)
    brier_skill = (
        1.0 - calibrated_metrics.multiclass_brier / prior_metrics.multiclass_brier
    )
    final_diagnostics = collapse_diagnostics(final_actual, calibrated_final)
    final_blockers = _final_blockers(
        raw=raw_metrics,
        calibrated=calibrated_metrics,
        prior=prior_metrics,
        brier_skill=brier_skill,
        collapse=final_diagnostics,
    )
    final_samples = tuple(samples[index] for index in diagnostic_fold.test_indices)

    return {
        "schema_version": 1,
        "step18_version": STEP18_VERSION,
        "research_identity": RESEARCH_IDENTITY,
        "dataset_id": dataset.dataset_id,
        "target": {
            "outcomes": list(CLASS_ORDER),
            "horizon_minutes": 60,
            "barriers": "SYMMETRIC_1.5_ATR_FIRST_TOUCH",
            "locked_from_step15": True,
        },
        "feature_architecture": {
            "base_feature_count": len(dataset.feature_names),
            "enhanced_feature_count": len(enhanced.feature_names),
            "added_feature_count": len(enhanced.feature_names) - len(dataset.feature_names),
            "feature_version": ENHANCED_FEATURE_VERSION,
            "feature_set_hash": ENHANCED_FEATURE_SET_HASH,
            "inputs": "FINALIZED_5M_15M_1H_ONLY",
            "developing_candle_used": False,
            "nifty_spot_volume_used": False,
            "vwap_used": False,
            "news_used": False,
            "live_backtest_parity": "SAME_PURE_ENHANCEMENT_FUNCTION_REQUIRED",
        },
        "chronology": {
            "walk_forward": NESTED_WALK_FORWARD_CONFIG.to_contract(),
            "architecture_selection_folds": list(ARCHITECTURE_SELECTION_FOLDS),
            "calibration_fit_fold": CALIBRATION_FIT_FOLD,
            "calibration_selection_fold": CALIBRATION_SELECTION_FOLD,
            "historical_diagnostic_fold": HISTORICAL_DIAGNOSTIC_FOLD,
            "historical_diagnostic_starts_at": diagnostic_fold.test_starts_at.isoformat(),
            "historical_diagnostic_ends_at": diagnostic_fold.test_ends_at.isoformat(),
            "historical_period_previously_seen": True,
        },
        "candidate_comparison": candidate_reports,
        "selected_candidate": selected_name,
        "selected_candidate_selection_viable": bool(selected_report["selection_viable"]),
        "calibration_comparison": calibration_reports,
        "selected_calibration_method": selected_calibration.method,
        "historical_diagnostic": {
            "raw_metrics": raw_metrics.to_contract(),
            "calibrated_metrics": calibrated_metrics.to_contract(),
            "prior_metrics": prior_metrics.to_contract(),
            "brier_skill_vs_prior": brier_skill,
            "collapse_diagnostics": final_diagnostics,
            "regime_diagnostics": regime_diagnostics(
                samples=final_samples,
                actual=final_actual,
                probabilities=calibrated_final,
            ),
            "research_gate_passed": not [
                item
                for item in final_blockers
                if item not in {
                    "HISTORICAL_PERIOD_PREVIOUSLY_USED",
                    "FORWARD_CONFIRMATION_NOT_COMPLETED",
                }
            ],
            "blockers": final_blockers,
        },
        "final_research_gate": FINAL_RESEARCH_GATE,
        "forward_confirmation": {
            "required": True,
            "minimum_eligible_predictions": 2_000,
            "minimum_distinct_sessions": 60,
            "begins_after": diagnostic_fold.test_ends_at.isoformat(),
            "architecture_feature_or_threshold_change_restarts_confirmation": True,
        },
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "existing_step17_runtime_modified": False,
        "model_artifact_created": False,
        "approved_for_live_inference": False,
        "precise_probability_display_allowed": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
    }


def collapse_diagnostics(
    actual: tuple[str, ...], probabilities: np.ndarray
) -> dict[str, object]:
    metrics = calculate_metrics(actual, probabilities)
    predicted = tuple(CLASS_ORDER[index] for index in probabilities.argmax(axis=1))
    counts = Counter(predicted)
    total = len(predicted)
    recalls = dict(metrics.class_recall)
    return {
        "predicted_class_counts": {name: counts[name] for name in CLASS_ORDER},
        "predicted_class_share": {
            name: counts[name] / total for name in CLASS_ORDER
        },
        "up_recall": recalls["UP"],
        "down_recall": recalls["DOWN"],
        "neither_recall": recalls["NEITHER"],
        "directional_collapse_detected": (
            counts["UP"] / total < COLLAPSE_GATE["minimum_up_prediction_share"]
            or counts["DOWN"] / total < COLLAPSE_GATE["minimum_down_prediction_share"]
            or recalls["UP"] < COLLAPSE_GATE["minimum_up_recall"]
            or recalls["DOWN"] < COLLAPSE_GATE["minimum_down_recall"]
        ),
    }


def collapse_blockers(diagnostics: dict[str, object]) -> list[str]:
    share = diagnostics["predicted_class_share"]
    blockers = []
    if share["UP"] < COLLAPSE_GATE["minimum_up_prediction_share"]:
        blockers.append("UP_PREDICTION_SHARE_TOO_LOW")
    if share["DOWN"] < COLLAPSE_GATE["minimum_down_prediction_share"]:
        blockers.append("DOWN_PREDICTION_SHARE_TOO_LOW")
    if diagnostics["up_recall"] < COLLAPSE_GATE["minimum_up_recall"]:
        blockers.append("UP_RECALL_TOO_LOW")
    if diagnostics["down_recall"] < COLLAPSE_GATE["minimum_down_recall"]:
        blockers.append("DOWN_RECALL_TOO_LOW")
    return blockers


def combine_hierarchical_probabilities(
    actionable_probability: np.ndarray,
    up_given_actionable_probability: np.ndarray,
) -> np.ndarray:
    actionable = np.clip(np.asarray(actionable_probability, dtype=float), 0.0, 1.0)
    up_given = np.clip(
        np.asarray(up_given_actionable_probability, dtype=float), 0.0, 1.0
    )
    if actionable.shape != up_given.shape:
        raise ValueError("Hierarchical probability vectors must have matching shapes")
    result = np.column_stack(
        (
            actionable * (1.0 - up_given),
            1.0 - actionable,
            actionable * up_given,
        )
    )
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-9):
        raise AssertionError("Combined hierarchical probabilities must sum to one")
    return result


def regime_diagnostics(
    *,
    samples: tuple[TrainingSample, ...],
    actual: tuple[str, ...],
    probabilities: np.ndarray,
) -> list[dict[str, object]]:
    name = "enhanced__trend_alignment"
    index = samples[0].feature_names.index(name)
    values = np.asarray([item.feature_values[index] for item in samples])
    regimes = (
        ("BEAR_ALIGNED", values <= -0.66),
        ("MIXED", (values > -0.66) & (values < 0.66)),
        ("BULL_ALIGNED", values >= 0.66),
    )
    reports = []
    for regime_name, mask in regimes:
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            reports.append({"regime": regime_name, "sample_count": 0})
            continue
        subset_actual = tuple(actual[index] for index in indices)
        reports.append(
            {
                "regime": regime_name,
                "sample_count": len(indices),
                "metrics": calculate_metrics(
                    subset_actual, probabilities[indices]
                ).to_contract(),
                "collapse_diagnostics": collapse_diagnostics(
                    subset_actual, probabilities[indices]
                ),
            }
        )
    return reports


def _fit_predict(
    candidate_name: str,
    train: tuple[TrainingSample, ...],
    test: tuple[TrainingSample, ...],
) -> np.ndarray:
    if candidate_name == "direct_logistic_v1":
        feature_indices = tuple(
            index
            for index, name in enumerate(train[0].feature_names)
            if not name.startswith("enhanced__")
        )
        return _direct_probabilities(
            train, test, feature_indices, _logistic_factory(None)
        )
    feature_indices = tuple(range(len(train[0].feature_names)))
    if candidate_name == "direct_logistic_v2":
        return _direct_probabilities(
            train, test, feature_indices, _logistic_factory(None)
        )
    if candidate_name == "direct_logistic_balanced_v2":
        return _direct_probabilities(
            train, test, feature_indices, _logistic_factory("balanced")
        )
    if candidate_name == "hierarchical_logistic_v2":
        return _hierarchical_probabilities(
            train,
            test,
            feature_indices,
            _logistic_factory(None),
            _logistic_factory(None),
        )
    if candidate_name == "hierarchical_logistic_balanced_v2":
        return _hierarchical_probabilities(
            train,
            test,
            feature_indices,
            _logistic_factory("balanced"),
            _logistic_factory("balanced"),
        )
    if candidate_name == "hierarchical_hgb_balanced_v2":
        return _hierarchical_probabilities(
            train,
            test,
            feature_indices,
            _hgb_factory(),
            _hgb_factory(),
            balanced_sample_weights=True,
        )
    raise ValueError(f"Unknown Step 18 candidate: {candidate_name}")


def _direct_probabilities(
    train: tuple[TrainingSample, ...],
    test: tuple[TrainingSample, ...],
    feature_indices: tuple[int, ...],
    factory: Callable[[], object],
) -> np.ndarray:
    model = factory()
    train_x = _matrix(train, feature_indices)
    test_x = _matrix(test, feature_indices)
    train_y = np.asarray([item.outcome.value for item in train])
    model.fit(train_x, train_y)
    raw = model.predict_proba(test_x)
    classes = tuple(str(item) for item in model.classes_)
    return raw[:, [classes.index(name) for name in CLASS_ORDER]]


def _hierarchical_probabilities(
    train: tuple[TrainingSample, ...],
    test: tuple[TrainingSample, ...],
    feature_indices: tuple[int, ...],
    opportunity_factory: Callable[[], object],
    direction_factory: Callable[[], object],
    *,
    balanced_sample_weights: bool = False,
) -> np.ndarray:
    train_x = _matrix(train, feature_indices)
    test_x = _matrix(test, feature_indices)
    outcomes = np.asarray([item.outcome.value for item in train])
    actionable = np.asarray(outcomes != "NEITHER", dtype=int)
    opportunity = opportunity_factory()
    opportunity_weights = _balanced_weights(actionable) if balanced_sample_weights else None
    opportunity.fit(train_x, actionable, **_fit_kwargs(opportunity_weights))
    actionable_probability = _binary_positive_probability(
        opportunity, opportunity.predict_proba(test_x), 1
    )

    directional_mask = outcomes != "NEITHER"
    direction_labels = np.asarray(outcomes[directional_mask] == "UP", dtype=int)
    direction = direction_factory()
    direction_weights = (
        _balanced_weights(direction_labels) if balanced_sample_weights else None
    )
    direction.fit(
        train_x[directional_mask],
        direction_labels,
        **_fit_kwargs(direction_weights),
    )
    up_given_actionable = _binary_positive_probability(
        direction, direction.predict_proba(test_x), 1
    )
    return combine_hierarchical_probabilities(
        actionable_probability, up_given_actionable
    )


def _matrix(
    samples: tuple[TrainingSample, ...], feature_indices: tuple[int, ...]
) -> np.ndarray:
    return np.asarray(
        [[item.feature_values[index] for index in feature_indices] for item in samples],
        dtype=float,
    )


def _logistic_factory(class_weight: str | None) -> Callable[[], Pipeline]:
    return lambda: Pipeline(
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


def _hgb_factory() -> Callable[[], HistGradientBoostingClassifier]:
    return lambda: HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )


def _fit_kwargs(weights: np.ndarray | None) -> dict[str, object]:
    return {"sample_weight": weights} if weights is not None else {}


def _balanced_weights(labels: np.ndarray) -> np.ndarray:
    counts = Counter(int(item) for item in labels)
    return np.asarray(
        [len(labels) / (len(counts) * counts[int(item)]) for item in labels],
        dtype=float,
    )


def _binary_positive_probability(
    estimator: object, probabilities: np.ndarray, positive_class: int
) -> np.ndarray:
    classes = tuple(int(item) for item in estimator.classes_)
    return probabilities[:, classes.index(positive_class)]


def _prior_vector(actual: tuple[str, ...]) -> np.ndarray:
    return prior_probabilities(actual, 1)[0]


def _candidate_ranking_key(item: dict[str, object]) -> tuple[object, ...]:
    metrics = item["selection_metrics"]
    return (
        not bool(item["selection_viable"]),
        metrics["multiclass_brier"],
        metrics["log_loss"],
        -metrics["balanced_accuracy"],
        item["name"],
    )


def _final_blockers(
    *,
    raw: MetricSummary,
    calibrated: MetricSummary,
    prior: MetricSummary,
    brier_skill: float,
    collapse: dict[str, object],
) -> list[str]:
    blockers = collapse_blockers(collapse)
    if brier_skill <= FINAL_RESEARCH_GATE["minimum_brier_skill_vs_prior"]:
        blockers.append("BRIER_SKILL_GATE_FAILED")
    if calibrated.log_loss >= prior.log_loss:
        blockers.append("LOG_LOSS_GATE_FAILED")
    if calibrated.balanced_accuracy <= FINAL_RESEARCH_GATE["minimum_balanced_accuracy"]:
        blockers.append("BALANCED_ACCURACY_GATE_FAILED")
    if calibrated.raw_ece_10_bin > FINAL_RESEARCH_GATE["maximum_ece"]:
        blockers.append("ECE_GATE_FAILED")
    if (
        calibrated.multiclass_brier > raw.multiclass_brier
        or calibrated.log_loss > raw.log_loss
    ):
        blockers.append("CALIBRATION_DEGRADES_RAW_PROPER_SCORES")
    blockers.extend(
        ("HISTORICAL_PERIOD_PREVIOUSLY_USED", "FORWARD_CONFIRMATION_NOT_COMPLETED")
    )
    return blockers
