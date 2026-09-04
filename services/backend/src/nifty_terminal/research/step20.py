"""Step 20 pooled directional meta-label research with coverage constraints.

Step 19 trained separate long and short expected-R models.  Their score ordering
was unstable between chronological folds and the best exploratory policy was too
sparse.  Step 20 treats LONG and SHORT as two alternatives for the same decision,
pools their target-first labels, and gives one compact model an explicit direction
interaction.  This doubles supervised support without mixing timestamps or using
future information.

All output remains historical research.  A successful historical run still needs
a frozen, genuinely future shadow period before any official signal is possible.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nifty_terminal.calendar.nse import IST
from nifty_terminal.context.features import ContextFeatureBuild
from nifty_terminal.ml.definitions import RANDOM_SEED
from nifty_terminal.ml.models import WalkForwardConfig
from nifty_terminal.ml.split import PurgedWalkForwardSplitter
from nifty_terminal.research.step18b import (
    BinaryCalibrationArtifact,
    TradePath,
    apply_binary_calibrator,
    binary_metrics,
    build_trade_paths,
    fit_binary_calibrator,
)
from nifty_terminal.research.step18f import (
    benchmark_suite,
    daily_uplift_bootstrap,
    fit_causal_baseline,
    predict_causal_baseline,
    session_bootstrap_values,
)


STEP20_VERSION = "pooled_directional_coverage_research.v1.1"
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
    "pooled_logistic_structure_c0p02",
    "pooled_logistic_core_c0p02",
    "pooled_hgb_core",
)
CALIBRATION_METHODS = ("identity", "platt")
ACTIVATION_PERCENTILES = (0.45, 0.55, 0.65)
DIRECTIONAL_MARGINS = (0.00, 0.025, 0.05)
MODEL_GATE = {
    "minimum_pooled_roc_auc": 0.515,
    "minimum_pooled_brier_skill": 0.0,
    "minimum_probability_std": 0.01,
    "minimum_fold_roc_auc": 0.48,
    "minimum_folds_above_random_auc": 2,
    "minimum_folds_positive_brier_skill": 2,
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
            "version": STEP20_VERSION,
            "walk_forward": WALK_FORWARD_CONFIG.to_contract(),
            "models": MODEL_CANDIDATES,
            "calibration": CALIBRATION_METHODS,
            "activation_percentiles": ACTIVATION_PERCENTILES,
            "directional_margins": DIRECTIONAL_MARGINS,
            "model_gate": MODEL_GATE,
            "selection_policy_gate": SELECTION_POLICY_GATE,
            "diagnostic_policy_gate": DIAGNOSTIC_POLICY_GATE,
            "trial_ledger": {
                "model_candidates": len(MODEL_CANDIDATES),
                "calibration_candidates": len(CALIBRATION_METHODS),
                "policy_candidates": len(ACTIVATION_PERCENTILES) * len(DIRECTIONAL_MARGINS),
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


@dataclass(frozen=True, slots=True)
class CoveragePolicy:
    activation_percentile: float
    directional_margin: float


def run_pooled_directional_research(
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
        raise ValueError("Step 20 requires at least 24,000 complete trade paths")
    folds = PurgedWalkForwardSplitter().split(samples, WALK_FORWARD_CONFIG)
    names = context.matrix.feature_names
    labels = {
        "LONG": np.asarray([long_paths[item.sample_id].success for item in samples], dtype=int),
        "SHORT": np.asarray([short_paths[item.sample_id].success for item in samples], dtype=int),
    }

    comparisons: list[dict[str, object]] = []
    raw_predictions: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for candidate in MODEL_CANDIDATES:
        by_fold: dict[int, dict[str, np.ndarray]] = {}
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
            actual = _stack_labels(labels, np.asarray(fold.test_indices))
            prior = np.full(len(actual), output["train_prior"])
            metrics = binary_metrics(actual, output["pooled"])
            prior_metrics = binary_metrics(actual, prior)
            fold_reports.append(
                {
                    "fold_index": fold_index,
                    "metrics": metrics,
                    "prior_metrics": prior_metrics,
                    "brier_skill_vs_train_prior": _brier_skill(metrics, prior_metrics),
                }
            )
        pooled_actual = np.concatenate(
            [_stack_labels(labels, np.asarray(folds[index].test_indices)) for index in MODEL_SELECTION_FOLDS]
        )
        pooled_probability = np.concatenate([by_fold[index]["pooled"] for index in MODEL_SELECTION_FOLDS])
        pooled_prior = np.concatenate(
            [
                np.full(len(by_fold[index]["pooled"]), by_fold[index]["train_prior"])
                for index in MODEL_SELECTION_FOLDS
            ]
        )
        metrics = binary_metrics(pooled_actual, pooled_probability)
        prior_metrics = binary_metrics(pooled_actual, pooled_prior)
        skill = _brier_skill(metrics, prior_metrics)
        blockers = _model_blockers(metrics, skill, fold_reports)
        comparisons.append(
            {
                "candidate": candidate,
                "feature_count": _candidate_feature_count(candidate, names),
                "direction_interaction": True,
                "selection_metrics": metrics,
                "selection_prior_metrics": prior_metrics,
                "selection_brier_skill_vs_prior": skill,
                "folds": fold_reports,
                "selection_viable": not blockers,
                "selection_blockers": blockers,
            }
        )
        raw_predictions[candidate] = by_fold

    viable_models = [item for item in comparisons if item["selection_viable"]]
    selected_model_report = min(viable_models or comparisons, key=_model_rank)
    selected_candidate = str(selected_model_report["candidate"])
    for fold_index in (CALIBRATION_FOLD, POLICY_SELECTION_FOLD, *HISTORICAL_DIAGNOSTIC_FOLDS):
        fold = folds[fold_index]
        raw_predictions[selected_candidate][fold_index] = _fit_predict(
            candidate=selected_candidate,
            samples=samples,
            matrix=matrix,
            names=names,
            labels=labels,
            train_indices=np.asarray(fold.train_indices),
            test_indices=np.asarray(fold.test_indices),
        )

    calibration_output = raw_predictions[selected_candidate][CALIBRATION_FOLD]
    calibration_indices = np.asarray(folds[CALIBRATION_FOLD].test_indices)
    calibration_actual = _stack_labels(labels, calibration_indices)
    calibrations = []
    for method in CALIBRATION_METHODS:
        artifact = fit_binary_calibrator(
            method=method,
            probabilities=calibration_output["pooled"],
            actual=calibration_actual,
            prior=float(calibration_output["train_prior"]),
        )
        calibrated = apply_binary_calibrator(artifact, calibration_output["pooled"])
        calibrations.append(
            {
                "method": method,
                "artifact": artifact,
                "metrics": binary_metrics(calibration_actual, calibrated),
                "selection_blockers": _calibration_blockers(artifact),
            }
        )
    viable_calibrations = [item for item in calibrations if not item["selection_blockers"]]
    selected_calibration_report = min(
        viable_calibrations,
        key=lambda item: (item["metrics"]["brier"], item["metrics"]["log_loss"], item["method"]),
    )
    calibration: BinaryCalibrationArtifact = selected_calibration_report["artifact"]

    calibrated_by_fold = {
        index: {
            "LONG": apply_binary_calibrator(calibration, output["LONG"]),
            "SHORT": apply_binary_calibrator(calibration, output["SHORT"]),
            "pooled": apply_binary_calibrator(calibration, output["pooled"]),
            "train_prior": output["train_prior"],
        }
        for index, output in raw_predictions[selected_candidate].items()
    }

    reference_scores = _maximum_scores(calibrated_by_fold[CALIBRATION_FOLD])
    selection_scores = _maximum_scores(calibrated_by_fold[POLICY_SELECTION_FOLD])
    selection_percentiles = _reference_percentiles(reference_scores, selection_scores)
    selection_indices = np.asarray(folds[POLICY_SELECTION_FOLD].test_indices)
    selection_samples = tuple(samples[index] for index in selection_indices)
    selection_benchmarks = _benchmarks_for_folds(
        fold_indices=(POLICY_SELECTION_FOLD,),
        folds=folds,
        samples=samples,
        matrix=matrix,
        names=names,
        labels=labels,
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
                long_probabilities=calibrated_by_fold[POLICY_SELECTION_FOLD]["LONG"],
                short_probabilities=calibrated_by_fold[POLICY_SELECTION_FOLD]["SHORT"],
                long_paths=long_paths,
                short_paths=short_paths,
                policy=policy,
                benchmarks=selection_benchmarks,
            )
            blockers = _policy_blockers(metrics, SELECTION_POLICY_GATE, require_benchmarks=False)
            if not selected_model_report["selection_viable"]:
                blockers = sorted(set(blockers + ["MODEL_SELECTION_GATE_FAILED"]))
            policies.append(
                {
                    "policy": _policy_contract(policy),
                    "metrics": metrics,
                    "selection_viable": not blockers,
                    "selection_blockers": blockers,
                }
            )
    viable_policies = [item for item in policies if item["selection_viable"]]
    exploratory_policy = min(policies, key=_policy_rank)
    selected_policy_report = min(viable_policies, key=_policy_rank) if viable_policies else None
    evaluation_policy_report = selected_policy_report or exploratory_policy
    evaluation_policy = CoveragePolicy(**evaluation_policy_report["policy"])

    diagnostic_indices = np.concatenate(
        [np.asarray(folds[index].test_indices) for index in HISTORICAL_DIAGNOSTIC_FOLDS]
    )
    diagnostic_samples = tuple(samples[int(index)] for index in diagnostic_indices)
    diagnostic_long = np.concatenate(
        [calibrated_by_fold[index]["LONG"] for index in HISTORICAL_DIAGNOSTIC_FOLDS]
    )
    diagnostic_short = np.concatenate(
        [calibrated_by_fold[index]["SHORT"] for index in HISTORICAL_DIAGNOSTIC_FOLDS]
    )
    diagnostic_reference = np.concatenate(
        (
            reference_scores,
            selection_scores,
        )
    )
    diagnostic_percentiles = _reference_percentiles(
        diagnostic_reference,
        np.maximum(diagnostic_long, diagnostic_short),
    )
    diagnostic_benchmarks = _benchmarks_for_folds(
        fold_indices=HISTORICAL_DIAGNOSTIC_FOLDS,
        folds=folds,
        samples=samples,
        matrix=matrix,
        names=names,
        labels=labels,
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
        diagnostic_blockers = sorted(set(diagnostic_blockers + ["NO_POLICY_PASSED_SELECTION_GATE"]))

    diagnostic_actual = np.concatenate(
        [_stack_labels(labels, np.asarray(folds[index].test_indices)) for index in HISTORICAL_DIAGNOSTIC_FOLDS]
    )
    diagnostic_probability = np.concatenate(
        [calibrated_by_fold[index]["pooled"] for index in HISTORICAL_DIAGNOSTIC_FOLDS]
    )
    diagnostic_prior = np.concatenate(
        [
            np.full(
                len(calibrated_by_fold[index]["pooled"]),
                calibrated_by_fold[index]["train_prior"],
            )
            for index in HISTORICAL_DIAGNOSTIC_FOLDS
        ]
    )
    diagnostic_model_metrics = binary_metrics(diagnostic_actual, diagnostic_probability)
    diagnostic_prior_metrics = binary_metrics(diagnostic_actual, diagnostic_prior)
    diagnostic_model_skill = _brier_skill(diagnostic_model_metrics, diagnostic_prior_metrics)
    diagnostic_model_blockers = []
    if diagnostic_model_metrics["roc_auc"] is None or diagnostic_model_metrics["roc_auc"] <= 0.51:
        diagnostic_model_blockers.append("DIAGNOSTIC_ROC_AUC_GATE_FAILED")
    if diagnostic_model_skill <= 0:
        diagnostic_model_blockers.append("DIAGNOSTIC_BRIER_SKILL_GATE_FAILED")
    if diagnostic_model_metrics["ece_10_bin"] > 0.05:
        diagnostic_model_blockers.append("DIAGNOSTIC_ECE_GATE_FAILED")

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
        "step20_version": STEP20_VERSION,
        "research_identity": RESEARCH_IDENTITY,
        "dataset_id": context.dataset.dataset_id,
        "context_bundle_sha256": context_bundle_sha256,
        "objective": {
            "model_output": "pooled probability that the direction-specific 1R target is hit before its 0.75R stop",
            "direction_choice": "higher calibrated LONG/SHORT probability",
            "activation": "causal percentile versus earlier score reference",
            "frequency_goal": "at least 3 non-overlapping trades per represented session",
            "horizon_minutes": 60,
            "entry": "next finalized 1m open",
            "same_minute_resolution": "STOP_FIRST_CONSERVATIVE",
            "slippage_points_one_way": 0.5,
        },
        "dataset": {
            "complete_trade_paths": len(samples),
            "directional_training_rows": len(samples) * 2,
            "feature_count": matrix.shape[1],
            "excluded_trade_paths": exclusions,
        },
        "chronology": {
            "walk_forward": WALK_FORWARD_CONFIG.to_contract(),
            "model_selection_folds": list(MODEL_SELECTION_FOLDS),
            "calibration_fold": CALIBRATION_FOLD,
            "policy_selection_fold": POLICY_SELECTION_FOLD,
            "historical_diagnostic_folds": list(HISTORICAL_DIAGNOSTIC_FOLDS),
            "future_labels_used_in_scores": False,
            "same_timestamp_long_short_rows_kept_in_same_fold": True,
            "diagnostic_thresholds_locked_before_diagnostic": True,
            "historical_period_previously_seen": True,
        },
        "anti_overfit_controls": {
            "pooled_directional_meta_label": True,
            "candidate_count": len(MODEL_CANDIDATES),
            "calibration_candidate_count": len(CALIBRATION_METHODS),
            "policy_candidate_count": len(ACTIVATION_PERCENTILES) * len(DIRECTIONAL_MARGINS),
            "total_declared_trials": len(MODEL_CANDIDATES) + len(CALIBRATION_METHODS) + len(ACTIVATION_PERCENTILES) * len(DIRECTIONAL_MARGINS),
            "compact_domain_features": True,
            "session_balanced_training_weights": True,
            "purge_and_embargo_bars": 12,
            "non_overlapping_positions": True,
            "trade_frequency_is_a_hard_gate": True,
            "win_rate_is_not_optimized_without_expectancy": True,
        },
        "model_comparison": comparisons,
        "selected_model": {
            "candidate": selected_candidate,
            "selection_viable": selected_model_report["selection_viable"],
            "selection_blockers": selected_model_report["selection_blockers"],
            "feature_count": selected_model_report["feature_count"],
        },
        "calibration": {
            "selected": calibration.to_contract(),
            "comparison": [
                {"method": item["method"], "metrics": item["metrics"]}
                | {
                    "selection_viable": not item["selection_blockers"],
                    "selection_blockers": item["selection_blockers"],
                }
                for item in calibrations
            ],
            "fit_fold": CALIBRATION_FOLD,
        },
        "policy_selection": {
            "candidate_count": len(policies),
            "passing_candidate_count": len(viable_policies),
            "selected": selected_policy_report,
            "best_exploratory_rejected": None if selected_policy_report else exploratory_policy,
            "gate": SELECTION_POLICY_GATE,
        },
        "historical_diagnostic": {
            "evaluated_policy_source": "SELECTED" if selected_policy_report else "BEST_REJECTED_EXPLORATORY_ONLY",
            "policy": _policy_contract(evaluation_policy),
            "metrics": diagnostic,
            "gate": DIAGNOSTIC_POLICY_GATE,
            "gate_passed": not diagnostic_blockers and selected_policy_report is not None,
            "blockers": diagnostic_blockers,
        },
        "diagnostic_model": {
            "metrics": diagnostic_model_metrics,
            "prior_metrics": diagnostic_prior_metrics,
            "brier_skill_vs_prior": diagnostic_model_skill,
            "blockers": diagnostic_model_blockers,
            "gate_passed": not diagnostic_model_blockers,
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


def _fit_predict(*, candidate, samples, matrix, names, labels, train_indices, test_indices):
    feature_indices = _feature_indices(candidate, names)
    train_x = _directional_design(matrix[train_indices][:, feature_indices])
    test_x = _directional_design(matrix[test_indices][:, feature_indices])
    train_y = _stack_labels(labels, train_indices)
    weights = _session_balanced_weights(samples, train_indices)
    pooled_weights = np.concatenate((weights, weights))
    model = _candidate(candidate)
    if isinstance(model, Pipeline):
        model.fit(train_x, train_y, model__sample_weight=pooled_weights)
    else:
        model.fit(train_x, train_y, sample_weight=pooled_weights)
    probability = model.predict_proba(test_x)[:, list(model.classes_).index(1)]
    count = len(test_indices)
    return {
        "LONG": np.asarray(probability[:count], dtype=float),
        "SHORT": np.asarray(probability[count:], dtype=float),
        "pooled": np.asarray(probability, dtype=float),
        "train_prior": float(np.average(train_y, weights=pooled_weights)),
    }


def _directional_design(values: np.ndarray) -> np.ndarray:
    """Add explicit side interactions while keeping both sides in one time fold."""
    values = np.asarray(values, dtype=float)
    raw = np.vstack((values, values))
    signs = np.concatenate((np.ones(len(values)), -np.ones(len(values))))
    return np.column_stack((raw, raw * signs[:, None], signs))


def _stack_labels(labels: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    return np.concatenate((labels["LONG"][indices], labels["SHORT"][indices]))


def _candidate(name: str):
    if name.startswith("pooled_logistic"):
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
    if name == "pooled_hgb_core":
        return HistGradientBoostingClassifier(
            learning_rate=0.025,
            max_iter=180,
            max_leaf_nodes=7,
            min_samples_leaf=120,
            l2_regularization=12.0,
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown Step 20 candidate: {name}")


def _feature_indices(candidate: str, names: tuple[str, ...]) -> np.ndarray:
    if "structure" in candidate:
        indices = [
            index
            for index, name in enumerate(names)
            if name.startswith("price_action__")
            or name.startswith("research_v3__")
            or name.startswith("cross__")
        ]
    else:
        higher_timeframe_suffixes = (
            "__atr_pct",
            "__rsi_14",
            "__bollinger_z_20",
            "__roc_12",
            "__distance_ema20_atr",
            "__range_atr",
            "__trend_ema20_above_ema50",
        )
        context_suffixes = (
            "__return_1",
            "__return_3",
            "__return_12",
            "__realized_vol_12",
            "__realized_vol_48",
            "__ema_8_21_atr",
            "__ema_20_50_atr",
            "__rsi_14",
            "__return_z_48",
        )
        indices = []
        for index, name in enumerate(names):
            include = (
                name.startswith("primary_5m__")
                or name.startswith("research_v3__")
                or name.startswith("price_action__")
                or name.startswith("cross__")
                or (
                    name.startswith(("context_15m__", "context_1h__"))
                    and name.endswith(higher_timeframe_suffixes)
                )
                or (
                    name.startswith("context_market__")
                    and name.endswith(context_suffixes)
                )
            )
            if include and not _is_sparse_candle_flag(name):
                indices.append(index)
    if not indices:
        raise ValueError(f"Step 20 candidate {candidate} selected no features")
    return np.asarray(indices, dtype=int)


def _is_sparse_candle_flag(name: str) -> bool:
    return name.endswith(
        (
            "__doji",
            "__hammer",
            "__shooting_star",
            "__bullish_engulfing",
            "__bearish_engulfing",
            "__inside_bar",
            "__outside_bar",
        )
    )


def _candidate_feature_count(candidate: str, names: tuple[str, ...]) -> int:
    # Directional design contains raw, side-interaction, and one side indicator.
    count = len(_feature_indices(candidate, names))
    return 2 * count + 1


def _session_balanced_weights(samples, indices: np.ndarray) -> np.ndarray:
    sessions = [_session_key(samples[int(index)].decision_time) for index in indices]
    counts = Counter(sessions)
    values = np.asarray([1.0 / counts[item] for item in sessions], dtype=float)
    return values * len(values) / np.sum(values)


def _brier_skill(metrics, prior_metrics) -> float:
    return 1.0 - float(metrics["brier"]) / float(prior_metrics["brier"])


def _calibration_blockers(artifact: BinaryCalibrationArtifact) -> list[str]:
    if artifact.method == "platt" and float(artifact.parameters["coefficient"]) <= 0:
        return ["CALIBRATION_RANK_REVERSAL"]
    return []


def _model_blockers(metrics, skill: float, folds) -> list[str]:
    blockers = []
    auc = metrics["roc_auc"]
    if auc is None or auc <= MODEL_GATE["minimum_pooled_roc_auc"]:
        blockers.append("POOLED_ROC_AUC_GATE_FAILED")
    if skill <= MODEL_GATE["minimum_pooled_brier_skill"]:
        blockers.append("POOLED_BRIER_SKILL_GATE_FAILED")
    if metrics["probability_std"] < MODEL_GATE["minimum_probability_std"]:
        blockers.append("PROBABILITY_DISPERSION_TOO_LOW")
    fold_aucs = [float(item["metrics"]["roc_auc"] or 0.0) for item in folds]
    if min(fold_aucs) < MODEL_GATE["minimum_fold_roc_auc"]:
        blockers.append("FOLD_ROC_AUC_INSTABILITY")
    if sum(item > 0.5 for item in fold_aucs) < MODEL_GATE["minimum_folds_above_random_auc"]:
        blockers.append("INSUFFICIENT_FOLDS_ABOVE_RANDOM_AUC")
    if sum(float(item["brier_skill_vs_train_prior"]) > 0 for item in folds) < MODEL_GATE["minimum_folds_positive_brier_skill"]:
        blockers.append("INSUFFICIENT_FOLDS_POSITIVE_BRIER_SKILL")
    return blockers


def _model_rank(item) -> tuple[object, ...]:
    metrics = item["selection_metrics"]
    return (
        not item["selection_viable"],
        -float(item["selection_brier_skill_vs_prior"]),
        -float(metrics["roc_auc"] or 0.0),
        float(metrics["brier"]),
        str(item["candidate"]),
    )


def _maximum_scores(output) -> np.ndarray:
    return np.maximum(output["LONG"], output["SHORT"])


def _reference_percentiles(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=float))
    if len(ordered) == 0:
        raise ValueError("Step 20 percentile reference cannot be empty")
    return np.searchsorted(ordered, np.asarray(values, dtype=float), side="right") / len(ordered)


def simulate_coverage_policy(
    *, samples, score_percentiles, long_probabilities, short_probabilities,
    long_paths, short_paths, policy: CoveragePolicy, benchmarks,
) -> dict[str, object]:
    trades: list[TradePath] = []
    waits = Counter[str]()
    active_until: datetime | None = None
    for index, sample in enumerate(samples):
        if active_until is not None and sample.decision_time < active_until:
            waits["ACTIVE_POSITION"] += 1
            continue
        if score_percentiles[index] < policy.activation_percentile:
            waits["SCORE_PERCENTILE_BELOW_THRESHOLD"] += 1
            continue
        long_probability = float(long_probabilities[index])
        short_probability = float(short_probabilities[index])
        if abs(long_probability - short_probability) < policy.directional_margin:
            waits["DIRECTIONAL_MARGIN_TOO_SMALL"] += 1
            continue
        path = (
            long_paths[sample.sample_id]
            if long_probability > short_probability
            else short_paths[sample.sample_id]
        )
        trades.append(path)
        active_until = path.exited_at
    return _replay(tuple(trades), len(samples), waits, policy, benchmarks)


def _replay(trades, decisions, waits, policy, benchmarks) -> dict[str, object]:
    r_values = np.asarray([item.r_multiple for item in trades], dtype=float)
    points = [item.net_points for item in trades]
    gains = [item for item in points if item > 0]
    losses = [item for item in points if item < 0]
    cumulative = peak = drawdown = 0.0
    for value in r_values:
        cumulative += float(value)
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    counts = Counter(_session_key(item.decision_time) for item in trades)
    daily = _daily_totals(trades)
    return {
        "policy": _policy_contract(policy),
        "evaluation_decisions": decisions,
        "trade_count": len(trades),
        "buy_count": sum(item.direction == "LONG" for item in trades),
        "sell_count": sum(item.direction == "SHORT" for item in trades),
        "coverage": len(trades) / decisions if decisions else 0.0,
        "session_count": len(counts),
        "trades_per_session": len(trades) / len(counts) if counts else 0.0,
        "maximum_session_trade_share": max(counts.values()) / len(trades) if trades else None,
        "target_hit_count": sum(item.exit_reason == "TARGET" for item in trades),
        "stop_hit_count": sum(item.exit_reason == "STOP" for item in trades),
        "expired_count": sum(item.exit_reason == "HORIZON" for item in trades),
        "win_rate": len(gains) / len(trades) if trades else None,
        "net_points": float(sum(points)),
        "average_points": float(np.mean(points)) if points else None,
        "average_r_multiple": float(np.mean(r_values)) if len(r_values) else None,
        "average_r_session_bootstrap_95": session_bootstrap_values(trades, r_values),
        "profit_factor": sum(gains) / abs(sum(losses)) if gains and losses else None,
        "maximum_drawdown_r": drawdown,
        "benchmark_daily_r_uplift_95": {
            name: daily_uplift_bootstrap(daily, report["daily_total_r"])
            for name, report in benchmarks.items()
        },
        "wait_counts": dict(waits),
        "hypothetical_index_points_only": True,
        "rupee_pnl_available": False,
    }


def _policy_blockers(metrics, gate, *, require_benchmarks: bool) -> list[str]:
    blockers = []
    if metrics["trade_count"] < gate["minimum_trades"]:
        blockers.append("TRADE_SUPPORT_TOO_LOW")
    if metrics["buy_count"] < gate["minimum_trades_per_direction"]:
        blockers.append("BUY_SUPPORT_TOO_LOW")
    if metrics["sell_count"] < gate["minimum_trades_per_direction"]:
        blockers.append("SELL_SUPPORT_TOO_LOW")
    if metrics["session_count"] < gate["minimum_sessions"]:
        blockers.append("SESSION_SUPPORT_TOO_LOW")
    if metrics["trades_per_session"] < gate["minimum_trades_per_session"]:
        blockers.append("TRADE_CADENCE_TOO_LOW")
    if metrics["win_rate"] is None or metrics["win_rate"] < gate["minimum_win_rate"]:
        blockers.append("WIN_RATE_GATE_FAILED")
    if metrics["profit_factor"] is None or metrics["profit_factor"] <= gate["minimum_profit_factor"]:
        blockers.append("PROFIT_FACTOR_GATE_FAILED")
    lower = metrics["average_r_session_bootstrap_95"]["lower"]
    if lower is None or lower <= gate["minimum_average_r_lower_95"]:
        blockers.append("EXPECTANCY_CONFIDENCE_GATE_FAILED")
    if metrics["maximum_drawdown_r"] > gate["maximum_drawdown_r"]:
        blockers.append("MAXIMUM_DRAWDOWN_GATE_FAILED")
    share = metrics["maximum_session_trade_share"]
    if share is None or share > gate["maximum_session_trade_share"]:
        blockers.append("SESSION_CONCENTRATION_TOO_HIGH")
    if require_benchmarks:
        for name in ("WAIT", "ALWAYS_LONG", "ALWAYS_SHORT", "TECHNICAL_TREND"):
            interval = metrics["benchmark_daily_r_uplift_95"].get(name)
            if interval is None or interval["lower"] <= gate["minimum_daily_r_uplift_lower_95"]:
                blockers.append(f"DAILY_R_UPLIFT_NOT_POSITIVE_VS_{name}")
    return sorted(set(blockers))


def _policy_rank(item) -> tuple[object, ...]:
    metrics = item["metrics"]
    lower = metrics["average_r_session_bootstrap_95"]["lower"]
    return (
        not item["selection_viable"],
        len(item["selection_blockers"]),
        -(lower if lower is not None else -1_000.0),
        -int(metrics["trade_count"]),
        -float(metrics["win_rate"] or 0.0),
    )


def _benchmarks_for_folds(
    *, fold_indices, folds, samples, matrix, names, labels, long_paths, short_paths,
):
    tests = np.concatenate([np.asarray(folds[index].test_indices) for index in fold_indices])
    long_baselines = []
    short_baselines = []
    for fold_index in fold_indices:
        fold = folds[fold_index]
        train = np.asarray(fold.train_indices)
        test = np.asarray(fold.test_indices)
        for direction, destination in (("LONG", long_baselines), ("SHORT", short_baselines)):
            state = fit_causal_baseline(
                samples=samples,
                matrix=matrix,
                actual=labels[direction].astype(float),
                train_indices=train,
                names=names,
            )
            destination.append(
                predict_causal_baseline(state, samples=samples, matrix=matrix, indices=test)
            )
    return benchmark_suite(
        samples=tuple(samples[int(index)] for index in tests),
        matrix=matrix[tests],
        names=names,
        long_paths=long_paths,
        short_paths=short_paths,
        long_baseline=np.concatenate(long_baselines),
        short_baseline=np.concatenate(short_baselines),
    )


def _daily_totals(paths) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for path in paths:
        totals[_session_key(path.decision_time)] += float(path.r_multiple)
    return dict(totals)


def _policy_contract(policy: CoveragePolicy) -> dict[str, float]:
    return {
        "activation_percentile": policy.activation_percentile,
        "directional_margin": policy.directional_margin,
    }


def _session_key(value: datetime) -> str:
    return str(value.astimezone(IST).date())
