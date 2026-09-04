"""Step 23: separate opportunity detection from direction selection.

The Step 20 pooled label asks one model to learn two different effects: whether
the next hour contains enough movement for either target, and which side reaches
its target first.  Because the 1R target and 0.75R stop geometry permits at most
one successful side at a decision timestamp, these are naturally a conditional
two-head problem.

This module is historical research only.  Its candidates, folds, calibration,
policy grid and gates are declared before the diagnostic folds are evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nifty_terminal.context.features import ContextFeatureBuild
from nifty_terminal.ml.definitions import RANDOM_SEED
from nifty_terminal.ml.models import WalkForwardConfig
from nifty_terminal.ml.split import PurgedWalkForwardSplitter
from nifty_terminal.research.step18b import (
    BinaryCalibrationArtifact,
    apply_binary_calibrator,
    binary_metrics,
    build_trade_paths,
    fit_binary_calibrator,
)
from nifty_terminal.research.step20 import (
    CoveragePolicy,
    _benchmarks_for_folds,
    _feature_indices as step20_feature_indices,
    _policy_blockers,
    _policy_rank,
    _reference_percentiles,
    _session_balanced_weights,
    simulate_coverage_policy,
)


STEP23_VERSION = "conditional_direction_research.v1"
WALK_FORWARD_CONFIG = WalkForwardConfig(
    n_splits=7,
    minimum_train_samples=10_000,
    test_samples=2_000,
    purge_bars=12,
    embargo_bars=12,
    minimum_train_class_samples=25,
)
MODEL_SELECTION_FOLDS = (0, 1, 2)
CALIBRATION_FOLD = 3
POLICY_SELECTION_FOLD = 4
HISTORICAL_DIAGNOSTIC_FOLDS = (5, 6)
MODEL_CANDIDATES = (
    "conditional_logistic_structure_c0p02",
    "conditional_logistic_core_c0p02",
    "conditional_hgb_core",
)
CALIBRATION_METHODS = ("identity", "platt")
ACTIVATION_PERCENTILES = (0.45, 0.60, 0.75)
DIRECTIONAL_MARGINS = (0.00, 0.05, 0.10)
MODEL_GATE = {
    "minimum_direction_auc": 0.515,
    "minimum_direction_brier_skill": 0.0,
    "minimum_opportunity_auc": 0.505,
    "minimum_opportunity_brier_skill": 0.0,
    "minimum_direction_probability_std": 0.01,
    "minimum_folds_above_random_direction_auc": 2,
}
SELECTION_POLICY_GATE = {
    "minimum_trades": 120,
    "minimum_trades_per_direction": 30,
    "minimum_sessions": 25,
    "minimum_trades_per_session": 3.0,
    "minimum_win_rate": 0.50,
    "minimum_profit_factor": 1.10,
    "minimum_average_r_lower_95": 0.0,
    "maximum_drawdown_r": 12.0,
    "maximum_session_trade_share": 0.08,
}
DIAGNOSTIC_POLICY_GATE = {
    "minimum_trades": 240,
    "minimum_trades_per_direction": 60,
    "minimum_sessions": 50,
    "minimum_trades_per_session": 3.0,
    "minimum_win_rate": 0.50,
    "minimum_profit_factor": 1.05,
    "minimum_average_r_lower_95": 0.0,
    "maximum_drawdown_r": 20.0,
    "maximum_session_trade_share": 0.05,
    "minimum_daily_r_uplift_lower_95": 0.0,
}

RESEARCH_IDENTITY = hashlib.sha256(
    json.dumps(
        {
            "version": STEP23_VERSION,
            "walk_forward": WALK_FORWARD_CONFIG.to_contract(),
            "models": MODEL_CANDIDATES,
            "calibration": CALIBRATION_METHODS,
            "activation_percentiles": ACTIVATION_PERCENTILES,
            "directional_margins": DIRECTIONAL_MARGINS,
            "model_gate": MODEL_GATE,
            "selection_policy_gate": SELECTION_POLICY_GATE,
            "diagnostic_policy_gate": DIAGNOSTIC_POLICY_GATE,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


@dataclass(frozen=True, slots=True)
class ConditionalLabels:
    opportunity: np.ndarray
    long_when_opportunity: np.ndarray


def conditional_labels(samples, long_paths, short_paths) -> ConditionalLabels:
    """Return P(any winning side) and P(long wins | exactly one side wins)."""
    long_success = np.asarray(
        [long_paths[item.sample_id].success for item in samples], dtype=int
    )
    short_success = np.asarray(
        [short_paths[item.sample_id].success for item in samples], dtype=int
    )
    if np.any((long_success == 1) & (short_success == 1)):
        raise ValueError("Barrier geometry unexpectedly produced two successful sides")
    return ConditionalLabels(
        opportunity=np.asarray(long_success | short_success, dtype=int),
        long_when_opportunity=long_success,
    )


def run_conditional_direction_research(
    *,
    context: ContextFeatureBuild,
    minute_candles,
    context_bundle_sha256: str,
) -> dict[str, object]:
    samples, matrix, long_paths, short_paths, exclusions = build_trade_paths(
        dataset=context.dataset,
        features=context.matrix,
        minute_candles=minute_candles,
    )
    if len(samples) < 24_000:
        raise ValueError("Step 23 requires at least 24,000 complete trade paths")
    folds = PurgedWalkForwardSplitter().split(samples, WALK_FORWARD_CONFIG)
    names = context.matrix.feature_names
    labels = conditional_labels(samples, long_paths, short_paths)

    comparisons: list[dict[str, object]] = []
    predictions: dict[str, dict[int, dict[str, np.ndarray | float]]] = {}
    for candidate in MODEL_CANDIDATES:
        by_fold: dict[int, dict[str, np.ndarray | float]] = {}
        fold_reports = []
        for fold_index in MODEL_SELECTION_FOLDS:
            fold = folds[fold_index]
            output = _fit_predict(
                candidate=candidate,
                samples=samples,
                matrix=matrix,
                names=names,
                labels=labels,
                train_indices=np.asarray(fold.train_indices),
                test_indices=np.asarray(fold.test_indices),
            )
            by_fold[fold_index] = output
            fold_reports.append(
                _head_report(
                    fold_index=fold_index,
                    output=output,
                    labels=labels,
                    test_indices=np.asarray(fold.test_indices),
                )
            )
        pooled = _pooled_head_report(
            by_fold=by_fold,
            folds=folds,
            labels=labels,
            fold_indices=MODEL_SELECTION_FOLDS,
        )
        blockers = _model_blockers(pooled, fold_reports)
        comparisons.append(
            {
                "candidate": candidate,
                "feature_count_per_head": len(_feature_indices(candidate, names)),
                "selection": pooled,
                "folds": fold_reports,
                "selection_viable": not blockers,
                "selection_blockers": blockers,
            }
        )
        predictions[candidate] = by_fold

    viable_models = [item for item in comparisons if item["selection_viable"]]
    selected_model_report = min(viable_models or comparisons, key=_model_rank)
    selected_candidate = str(selected_model_report["candidate"])
    for fold_index in (
        CALIBRATION_FOLD,
        POLICY_SELECTION_FOLD,
        *HISTORICAL_DIAGNOSTIC_FOLDS,
    ):
        fold = folds[fold_index]
        predictions[selected_candidate][fold_index] = _fit_predict(
            candidate=selected_candidate,
            samples=samples,
            matrix=matrix,
            names=names,
            labels=labels,
            train_indices=np.asarray(fold.train_indices),
            test_indices=np.asarray(fold.test_indices),
        )

    calibration_indices = np.asarray(folds[CALIBRATION_FOLD].test_indices)
    calibration_output = predictions[selected_candidate][CALIBRATION_FOLD]
    opportunity_calibration, opportunity_comparison = _select_calibration(
        probabilities=np.asarray(calibration_output["opportunity"]),
        actual=labels.opportunity[calibration_indices],
        prior=float(calibration_output["opportunity_prior"]),
    )
    opportunity_mask = labels.opportunity[calibration_indices] == 1
    direction_calibration, direction_comparison = _select_calibration(
        probabilities=np.asarray(calibration_output["direction"])[opportunity_mask],
        actual=labels.long_when_opportunity[calibration_indices][opportunity_mask],
        prior=float(calibration_output["direction_prior"]),
    )
    calibrated = {
        fold_index: _calibrated_scores(
            output,
            opportunity_calibration=opportunity_calibration,
            direction_calibration=direction_calibration,
        )
        for fold_index, output in predictions[selected_candidate].items()
    }

    reference = np.maximum(
        calibrated[CALIBRATION_FOLD]["LONG"],
        calibrated[CALIBRATION_FOLD]["SHORT"],
    )
    selection = calibrated[POLICY_SELECTION_FOLD]
    selection_scores = np.maximum(selection["LONG"], selection["SHORT"])
    selection_percentiles = _reference_percentiles(reference, selection_scores)
    selection_indices = np.asarray(folds[POLICY_SELECTION_FOLD].test_indices)
    selection_samples = tuple(samples[int(index)] for index in selection_indices)
    path_labels = {
        "LONG": np.asarray(
            [long_paths[item.sample_id].success for item in samples], dtype=int
        ),
        "SHORT": np.asarray(
            [short_paths[item.sample_id].success for item in samples], dtype=int
        ),
    }
    selection_benchmarks = _benchmarks_for_folds(
        fold_indices=(POLICY_SELECTION_FOLD,),
        folds=folds,
        samples=samples,
        matrix=matrix,
        names=names,
        labels=path_labels,
        long_paths=long_paths,
        short_paths=short_paths,
    )
    policies = []
    for activation in ACTIVATION_PERCENTILES:
        for margin in DIRECTIONAL_MARGINS:
            policy = CoveragePolicy(activation, margin)
            metrics = simulate_coverage_policy(
                samples=selection_samples,
                score_percentiles=selection_percentiles,
                long_probabilities=selection["LONG"],
                short_probabilities=selection["SHORT"],
                long_paths=long_paths,
                short_paths=short_paths,
                policy=policy,
                benchmarks=selection_benchmarks,
            )
            blockers = _policy_blockers(
                metrics, SELECTION_POLICY_GATE, require_benchmarks=False
            )
            if not selected_model_report["selection_viable"]:
                blockers = sorted(set(blockers + ["MODEL_SELECTION_GATE_FAILED"]))
            policies.append(
                {
                    "policy": {
                        "activation_percentile": activation,
                        "directional_margin": margin,
                    },
                    "metrics": metrics,
                    "selection_viable": not blockers,
                    "selection_blockers": blockers,
                }
            )
    viable_policies = [item for item in policies if item["selection_viable"]]
    exploratory_policy = min(policies, key=_policy_rank)
    selected_policy_report = (
        min(viable_policies, key=_policy_rank) if viable_policies else None
    )
    evaluation_policy_report = selected_policy_report or exploratory_policy
    evaluation_policy = CoveragePolicy(**evaluation_policy_report["policy"])

    diagnostic_indices = np.concatenate(
        [
            np.asarray(folds[index].test_indices)
            for index in HISTORICAL_DIAGNOSTIC_FOLDS
        ]
    )
    diagnostic_samples = tuple(samples[int(index)] for index in diagnostic_indices)
    diagnostic_long = np.concatenate(
        [calibrated[index]["LONG"] for index in HISTORICAL_DIAGNOSTIC_FOLDS]
    )
    diagnostic_short = np.concatenate(
        [calibrated[index]["SHORT"] for index in HISTORICAL_DIAGNOSTIC_FOLDS]
    )
    diagnostic_reference = np.concatenate((reference, selection_scores))
    diagnostic_percentiles = _reference_percentiles(
        diagnostic_reference, np.maximum(diagnostic_long, diagnostic_short)
    )
    diagnostic_benchmarks = _benchmarks_for_folds(
        fold_indices=HISTORICAL_DIAGNOSTIC_FOLDS,
        folds=folds,
        samples=samples,
        matrix=matrix,
        names=names,
        labels=path_labels,
        long_paths=long_paths,
        short_paths=short_paths,
    )
    diagnostic = simulate_coverage_policy(
        samples=diagnostic_samples,
        score_percentiles=diagnostic_percentiles,
        long_probabilities=diagnostic_long,
        short_probabilities=diagnostic_short,
        long_paths=long_paths,
        short_paths=short_paths,
        policy=evaluation_policy,
        benchmarks=diagnostic_benchmarks,
    )
    diagnostic_blockers = _policy_blockers(
        diagnostic, DIAGNOSTIC_POLICY_GATE, require_benchmarks=True
    )
    if selected_policy_report is None:
        diagnostic_blockers = sorted(
            set(diagnostic_blockers + ["NO_POLICY_PASSED_SELECTION_GATE"])
        )
    diagnostic_heads = _pooled_head_report(
        by_fold=predictions[selected_candidate],
        folds=folds,
        labels=labels,
        fold_indices=HISTORICAL_DIAGNOSTIC_FOLDS,
        calibrated=calibrated,
    )
    diagnostic_model_blockers = _diagnostic_model_blockers(diagnostic_heads)
    historical_blockers = sorted(
        set(
            ([] if selected_model_report["selection_viable"] else ["MODEL_SELECTION_GATE_FAILED"])
            + ([] if selected_policy_report is not None else ["NO_POLICY_PASSED_SELECTION_GATE"])
            + diagnostic_model_blockers
            + diagnostic_blockers
        )
    )
    return {
        "schema_version": 1,
        "step23_version": STEP23_VERSION,
        "research_identity": RESEARCH_IDENTITY,
        "dataset_id": context.dataset.dataset_id,
        "context_bundle_sha256": context_bundle_sha256,
        "objective": {
            "opportunity_head": "probability that exactly one direction reaches its 1R target before its 0.75R stop",
            "direction_head": "probability LONG is the successful side conditional on an opportunity",
            "joint_long_score": "P(opportunity) * P(LONG | opportunity)",
            "joint_short_score": "P(opportunity) * (1 - P(LONG | opportunity))",
            "entry": "next finalized 1m open",
            "horizon_minutes": 60,
            "same_minute_resolution": "STOP_FIRST_CONSERVATIVE",
            "slippage_points_one_way": 0.5,
        },
        "dataset": {
            "complete_trade_paths": len(samples),
            "opportunity_rows": int(np.sum(labels.opportunity)),
            "no_opportunity_rows": int(len(labels.opportunity) - np.sum(labels.opportunity)),
            "feature_count": matrix.shape[1],
            "excluded_trade_paths": exclusions,
        },
        "chronology": {
            "walk_forward": WALK_FORWARD_CONFIG.to_contract(),
            "model_selection_folds": list(MODEL_SELECTION_FOLDS),
            "calibration_fold": CALIBRATION_FOLD,
            "policy_selection_fold": POLICY_SELECTION_FOLD,
            "historical_diagnostic_folds": list(HISTORICAL_DIAGNOSTIC_FOLDS),
            "diagnostic_thresholds_locked_before_diagnostic": True,
            "future_labels_used_in_scores": False,
            "historical_period_previously_seen": True,
        },
        "anti_overfit_controls": {
            "conditional_objective_predeclared": True,
            "model_candidates": len(MODEL_CANDIDATES),
            "calibration_candidates_per_head": len(CALIBRATION_METHODS),
            "policy_candidates": len(ACTIVATION_PERCENTILES)
            * len(DIRECTIONAL_MARGINS),
            "purge_and_embargo_bars": 12,
            "non_overlapping_positions": True,
            "session_balanced_training_weights": True,
        },
        "model_comparison": comparisons,
        "selected_model": {
            "candidate": selected_candidate,
            "selection_viable": selected_model_report["selection_viable"],
            "selection_blockers": selected_model_report["selection_blockers"],
            "feature_count_per_head": selected_model_report["feature_count_per_head"],
        },
        "calibration": {
            "fit_fold": CALIBRATION_FOLD,
            "opportunity": {
                "selected": opportunity_calibration.to_contract(),
                "comparison": opportunity_comparison,
            },
            "direction": {
                "selected": direction_calibration.to_contract(),
                "comparison": direction_comparison,
            },
        },
        "policy_selection": {
            "candidate_count": len(policies),
            "candidates": policies,
            "passing_candidate_count": len(viable_policies),
            "selected": selected_policy_report,
            "best_exploratory_rejected": (
                None if selected_policy_report else exploratory_policy
            ),
            "gate": SELECTION_POLICY_GATE,
        },
        "historical_diagnostic": {
            "evaluated_policy_source": (
                "SELECTED" if selected_policy_report else "BEST_REJECTED_EXPLORATORY_ONLY"
            ),
            "metrics": diagnostic,
            "heads": diagnostic_heads,
            "gate": DIAGNOSTIC_POLICY_GATE,
            "gate_passed": not diagnostic_blockers and selected_policy_report is not None,
            "blockers": diagnostic_blockers,
        },
        "research_gate": {
            "model_gate": MODEL_GATE,
            "selection_policy_gate": SELECTION_POLICY_GATE,
            "diagnostic_policy_gate": DIAGNOSTIC_POLICY_GATE,
            "passed_before_mandatory_forward_blockers": not historical_blockers,
            "historical_blockers": historical_blockers,
            "blockers": sorted(
                set(
                    historical_blockers
                    + [
                        "HISTORICAL_PERIOD_USED_FOR_MODEL_DEVELOPMENT",
                        "FORWARD_CONFIRMATION_NOT_COMPLETED",
                    ]
                )
            ),
        },
        "model_artifact_created": False,
        "approved_for_live_inference": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
    }


def _fit_predict(
    *, candidate, samples, matrix, names, labels, train_indices, test_indices
):
    feature_indices = _feature_indices(candidate, names)
    train_x = matrix[train_indices][:, feature_indices]
    test_x = matrix[test_indices][:, feature_indices]
    opportunity_weights = _session_balanced_weights(samples, train_indices)
    opportunity_model = _candidate(candidate)
    _fit(
        opportunity_model,
        train_x,
        labels.opportunity[train_indices],
        opportunity_weights,
    )
    opportunity_probability = _positive_probability(opportunity_model, test_x)

    direction_train_mask = labels.opportunity[train_indices] == 1
    direction_indices = train_indices[direction_train_mask]
    direction_model = _candidate(candidate)
    direction_weights = _session_balanced_weights(samples, direction_indices)
    _fit(
        direction_model,
        matrix[direction_indices][:, feature_indices],
        labels.long_when_opportunity[direction_indices],
        direction_weights,
    )
    direction_probability = _positive_probability(direction_model, test_x)
    return {
        "opportunity": opportunity_probability,
        "direction": direction_probability,
        "opportunity_prior": float(
            np.average(labels.opportunity[train_indices], weights=opportunity_weights)
        ),
        "direction_prior": float(
            np.average(
                labels.long_when_opportunity[direction_indices],
                weights=direction_weights,
            )
        ),
    }


def _feature_indices(candidate: str, names: tuple[str, ...]) -> np.ndarray:
    source = (
        "pooled_logistic_structure_c0p02"
        if "structure" in candidate
        else "pooled_hgb_core"
    )
    return step20_feature_indices(source, names)


def _candidate(name: str):
    if name.startswith("conditional_logistic"):
        return Pipeline(
            (
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.02,
                        max_iter=2_000,
                        random_state=RANDOM_SEED,
                        solver="lbfgs",
                    ),
                ),
            )
        )
    if name == "conditional_hgb_core":
        return HistGradientBoostingClassifier(
            learning_rate=0.025,
            max_iter=180,
            max_leaf_nodes=7,
            min_samples_leaf=120,
            l2_regularization=12.0,
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown Step 23 candidate: {name}")


def _fit(model, x, y, weights) -> None:
    if isinstance(model, Pipeline):
        model.fit(x, y, model__sample_weight=weights)
    else:
        model.fit(x, y, sample_weight=weights)


def _positive_probability(model, values) -> np.ndarray:
    probabilities = model.predict_proba(values)
    return np.asarray(probabilities[:, list(model.classes_).index(1)], dtype=float)


def _head_report(*, fold_index, output, labels, test_indices):
    opportunity_actual = labels.opportunity[test_indices]
    direction_mask = opportunity_actual == 1
    direction_actual = labels.long_when_opportunity[test_indices][direction_mask]
    opportunity_probability = np.asarray(output["opportunity"])
    direction_probability = np.asarray(output["direction"])
    opportunity_prior = np.full(len(test_indices), output["opportunity_prior"])
    direction_prior = np.full(np.sum(direction_mask), output["direction_prior"])
    opportunity_metrics = binary_metrics(opportunity_actual, opportunity_probability)
    direction_metrics = binary_metrics(
        direction_actual, direction_probability[direction_mask]
    )
    opportunity_prior_metrics = binary_metrics(opportunity_actual, opportunity_prior)
    direction_prior_metrics = binary_metrics(direction_actual, direction_prior)
    return {
        "fold_index": fold_index,
        "opportunity": opportunity_metrics,
        "direction": direction_metrics,
        "opportunity_brier_skill_vs_prior": _brier_skill(
            opportunity_metrics, opportunity_prior_metrics
        ),
        "direction_brier_skill_vs_prior": _brier_skill(
            direction_metrics, direction_prior_metrics
        ),
        "unfiltered_chosen_direction_win_rate": _chosen_win_rate(
            opportunity_actual,
            labels.long_when_opportunity[test_indices],
            direction_probability,
        ),
    }


def _pooled_head_report(
    *, by_fold, folds, labels, fold_indices, calibrated=None
):
    opportunity_actual_parts = []
    direction_actual_parts = []
    opportunity_probability_parts = []
    direction_probability_parts = []
    opportunity_prior_parts = []
    direction_prior_parts = []
    win_numerators = []
    win_denominators = []
    for fold_index in fold_indices:
        indices = np.asarray(folds[fold_index].test_indices)
        actual_opportunity = labels.opportunity[indices]
        mask = actual_opportunity == 1
        output = by_fold[fold_index]
        opportunity_probability = (
            calibrated[fold_index]["opportunity"]
            if calibrated is not None
            else np.asarray(output["opportunity"])
        )
        direction_probability = (
            calibrated[fold_index]["direction"]
            if calibrated is not None
            else np.asarray(output["direction"])
        )
        opportunity_actual_parts.append(actual_opportunity)
        direction_actual_parts.append(labels.long_when_opportunity[indices][mask])
        opportunity_probability_parts.append(opportunity_probability)
        direction_probability_parts.append(direction_probability[mask])
        opportunity_prior_parts.append(
            np.full(len(indices), output["opportunity_prior"])
        )
        direction_prior_parts.append(
            np.full(np.sum(mask), output["direction_prior"])
        )
        wins = _chosen_wins(
            actual_opportunity,
            labels.long_when_opportunity[indices],
            direction_probability,
        )
        win_numerators.append(int(np.sum(wins)))
        win_denominators.append(len(wins))
    opportunity_actual = np.concatenate(opportunity_actual_parts)
    direction_actual = np.concatenate(direction_actual_parts)
    opportunity_probability = np.concatenate(opportunity_probability_parts)
    direction_probability = np.concatenate(direction_probability_parts)
    opportunity_metrics = binary_metrics(opportunity_actual, opportunity_probability)
    direction_metrics = binary_metrics(direction_actual, direction_probability)
    opportunity_prior_metrics = binary_metrics(
        opportunity_actual, np.concatenate(opportunity_prior_parts)
    )
    direction_prior_metrics = binary_metrics(
        direction_actual, np.concatenate(direction_prior_parts)
    )
    return {
        "opportunity": opportunity_metrics,
        "direction": direction_metrics,
        "opportunity_brier_skill_vs_prior": _brier_skill(
            opportunity_metrics, opportunity_prior_metrics
        ),
        "direction_brier_skill_vs_prior": _brier_skill(
            direction_metrics, direction_prior_metrics
        ),
        "unfiltered_chosen_direction_win_rate": sum(win_numerators)
        / sum(win_denominators),
    }


def _chosen_wins(opportunity_actual, long_actual, direction_probability):
    choose_long = np.asarray(direction_probability) >= 0.5
    correct_direction = choose_long == (np.asarray(long_actual) == 1)
    return (np.asarray(opportunity_actual) == 1) & correct_direction


def _chosen_win_rate(opportunity_actual, long_actual, direction_probability):
    wins = _chosen_wins(opportunity_actual, long_actual, direction_probability)
    return float(np.mean(wins))


def _select_calibration(*, probabilities, actual, prior):
    reports = []
    for method in CALIBRATION_METHODS:
        artifact = fit_binary_calibrator(
            method=method,
            probabilities=probabilities,
            actual=actual,
            prior=prior,
        )
        calibrated = apply_binary_calibrator(artifact, probabilities)
        blockers = (
            ["CALIBRATION_RANK_REVERSAL"]
            if artifact.method == "platt"
            and float(artifact.parameters["coefficient"]) <= 0
            else []
        )
        reports.append(
            {
                "method": method,
                "artifact": artifact,
                "metrics": binary_metrics(actual, calibrated),
                "selection_blockers": blockers,
            }
        )
    viable = [item for item in reports if not item["selection_blockers"]]
    selected = min(
        viable,
        key=lambda item: (
            item["metrics"]["brier"],
            item["metrics"]["log_loss"],
            item["method"],
        ),
    )
    comparison = [
        {
            "method": item["method"],
            "metrics": item["metrics"],
            "selection_viable": not item["selection_blockers"],
            "selection_blockers": item["selection_blockers"],
        }
        for item in reports
    ]
    return selected["artifact"], comparison


def _calibrated_scores(
    output,
    *,
    opportunity_calibration: BinaryCalibrationArtifact,
    direction_calibration: BinaryCalibrationArtifact,
):
    opportunity = apply_binary_calibrator(
        opportunity_calibration, np.asarray(output["opportunity"])
    )
    direction = apply_binary_calibrator(
        direction_calibration, np.asarray(output["direction"])
    )
    return {
        "opportunity": opportunity,
        "direction": direction,
        "LONG": opportunity * direction,
        "SHORT": opportunity * (1.0 - direction),
    }


def _model_blockers(report, folds):
    blockers = []
    if report["direction"]["roc_auc"] is None or report["direction"]["roc_auc"] <= MODEL_GATE["minimum_direction_auc"]:
        blockers.append("DIRECTION_AUC_GATE_FAILED")
    if report["direction_brier_skill_vs_prior"] <= MODEL_GATE["minimum_direction_brier_skill"]:
        blockers.append("DIRECTION_BRIER_SKILL_GATE_FAILED")
    if report["opportunity"]["roc_auc"] is None or report["opportunity"]["roc_auc"] <= MODEL_GATE["minimum_opportunity_auc"]:
        blockers.append("OPPORTUNITY_AUC_GATE_FAILED")
    if report["opportunity_brier_skill_vs_prior"] <= MODEL_GATE["minimum_opportunity_brier_skill"]:
        blockers.append("OPPORTUNITY_BRIER_SKILL_GATE_FAILED")
    if report["direction"]["probability_std"] < MODEL_GATE["minimum_direction_probability_std"]:
        blockers.append("DIRECTION_PROBABILITY_DISPERSION_TOO_LOW")
    if sum(float(item["direction"]["roc_auc"] or 0.0) > 0.5 for item in folds) < MODEL_GATE["minimum_folds_above_random_direction_auc"]:
        blockers.append("INSUFFICIENT_FOLDS_ABOVE_RANDOM_DIRECTION_AUC")
    return blockers


def _diagnostic_model_blockers(report):
    blockers = []
    if report["direction"]["roc_auc"] is None or report["direction"]["roc_auc"] <= 0.51:
        blockers.append("DIAGNOSTIC_DIRECTION_AUC_GATE_FAILED")
    if report["direction_brier_skill_vs_prior"] <= 0:
        blockers.append("DIAGNOSTIC_DIRECTION_BRIER_SKILL_GATE_FAILED")
    if report["opportunity"]["roc_auc"] is None or report["opportunity"]["roc_auc"] <= 0.50:
        blockers.append("DIAGNOSTIC_OPPORTUNITY_AUC_GATE_FAILED")
    if report["opportunity_brier_skill_vs_prior"] <= 0:
        blockers.append("DIAGNOSTIC_OPPORTUNITY_BRIER_SKILL_GATE_FAILED")
    return blockers


def _brier_skill(metrics, prior_metrics):
    return 1.0 - float(metrics["brier"]) / float(prior_metrics["brier"])


def _model_rank(item):
    report = item["selection"]
    return (
        not item["selection_viable"],
        -float(report["direction_brier_skill_vs_prior"]),
        -float(report["direction"]["roc_auc"] or 0.0),
        -float(report["opportunity_brier_skill_vs_prior"]),
        str(item["candidate"]),
    )
