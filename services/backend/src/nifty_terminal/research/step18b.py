"""Step 18B trade-aligned directional probability research and replay."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
import json
import math
import platform
import warnings

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

from nifty_terminal.calendar.nse import IST
from nifty_terminal.domain.candle import Candle
from nifty_terminal.features.research_v3 import (
    RESEARCH_FEATURE_SET_HASH,
    RESEARCH_FEATURE_VERSION,
    ResearchFeatureMatrix,
    build_research_feature_matrix,
)
from nifty_terminal.ml.definitions import RANDOM_SEED
from nifty_terminal.ml.models import DatasetBuildReport, TrainingSample
from nifty_terminal.ml.split import PurgedWalkForwardSplitter
from nifty_terminal.research.v2 import NESTED_WALK_FORWARD_CONFIG


STEP18B_VERSION = "trade_aligned_model_research.v1.1"
MODEL_SELECTION_FOLDS = (0, 1)
CALIBRATION_FIT_FOLD = 2
CALIBRATION_SELECTION_FOLD = 3
HISTORICAL_DIAGNOSTIC_FOLD = 4
TARGET_ATR = Decimal("1.0")
STOP_ATR = Decimal("0.75")
HORIZON_MINUTES = 60
ONE_WAY_SLIPPAGE_POINTS = Decimal("0.5")
CANDIDATE_NAMES = (
    "technical_logistic_l2",
    "stationary_logistic_l2_c0p01",
    "stationary_logistic_l2_c0p1",
    "stationary_logistic_l2_c1",
    "stationary_elasticnet_c0p1_l1r0p25",
    "stationary_elasticnet_c1_l1r0p5",
    "stationary_hgb_shallow",
    "stationary_hgb_medium",
)
CALIBRATION_METHODS = (
    "identity",
    "platt",
    "isotonic",
    "beta",
    "prior_shrinkage",
)
BINARY_GATE = {
    "minimum_roc_auc": 0.51,
    "minimum_probability_standard_deviation": 0.015,
    "minimum_brier_skill_vs_prior": 0.0,
    "minimum_positive_support": 200,
    "maximum_ece": 0.05,
    "minimum_lower_95_brier_skill": 0.0,
}
POLICY_GATE = {
    "minimum_selection_trades": 100,
    "minimum_selection_buys": 30,
    "minimum_selection_sells": 30,
    "minimum_selection_profit_factor": 1.05,
    "minimum_selection_lower_95_average_r": 0.0,
    "minimum_diagnostic_trades": 100,
    "minimum_diagnostic_buys": 30,
    "minimum_diagnostic_sells": 30,
    "minimum_diagnostic_profit_factor": 1.0,
    "minimum_diagnostic_lower_95_average_r": 0.0,
    "overlapping_positions": False,
    "same_minute_stop_and_target": "STOP_FIRST_CONSERVATIVE",
}
RESEARCH_IDENTITY = hashlib.sha256(
    json.dumps(
        {
            "version": STEP18B_VERSION,
            "features": RESEARCH_FEATURE_SET_HASH,
            "candidates": CANDIDATE_NAMES,
            "calibration": CALIBRATION_METHODS,
            "target_atr": str(TARGET_ATR),
            "stop_atr": str(STOP_ATR),
            "horizon": HORIZON_MINUTES,
            "slippage": str(ONE_WAY_SLIPPAGE_POINTS),
            "binary_gate": BINARY_GATE,
            "policy_gate": POLICY_GATE,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


@dataclass(frozen=True, slots=True)
class TradePath:
    sample_id: str
    decision_time: datetime
    direction: str
    success: int
    exit_reason: str
    entered_at: datetime
    exited_at: datetime
    entry: float
    stop: float
    target: float
    exit: float
    net_points: float
    r_multiple: float


@dataclass(frozen=True, slots=True)
class BinaryCalibrationArtifact:
    method: str
    parameters: dict[str, object]

    def to_contract(self) -> dict[str, object]:
        return {
            "method": self.method,
            "parameters": self.parameters,
            "safe_json_parameters_only": True,
        }


def run_trade_aligned_research(
    *,
    dataset: DatasetBuildReport,
    minute_candles: tuple[Candle, ...],
    primary_candles: tuple[Candle, ...],
    feature_matrix: ResearchFeatureMatrix | None = None,
) -> dict[str, object]:
    features = feature_matrix or build_research_feature_matrix(dataset, primary_candles)
    if features.sample_ids != tuple(item.sample_id for item in dataset.samples):
        raise ValueError("Injected research features are not aligned to dataset samples")
    samples, matrix, long_paths, short_paths, exclusions = build_trade_paths(
        dataset=dataset,
        features=features,
        minute_candles=minute_candles,
    )
    folds = PurgedWalkForwardSplitter().split(samples, NESTED_WALK_FORWARD_CONFIG)
    labels = {
        "LONG": np.asarray([long_paths[item.sample_id].success for item in samples]),
        "SHORT": np.asarray([short_paths[item.sample_id].success for item in samples]),
    }
    probabilities: dict[str, dict[str, dict[int, np.ndarray]]] = {
        direction: {} for direction in labels
    }
    comparisons: dict[str, list[dict[str, object]]] = {direction: [] for direction in labels}
    selected_candidates: dict[str, str] = {}

    for direction, outcome in labels.items():
        for candidate in CANDIDATE_NAMES:
            by_fold: dict[int, np.ndarray] = {}
            fold_reports = []
            fold_priors = []
            non_converged_folds = []
            for fold in folds:
                train_x = matrix[np.asarray(fold.train_indices)]
                test_x = matrix[np.asarray(fold.test_indices)]
                train_y = outcome[np.asarray(fold.train_indices)]
                test_y = outcome[np.asarray(fold.test_indices)]
                model, indices = _candidate(candidate, features.feature_names)
                converged = _fit_with_convergence_check(
                    model, train_x[:, indices], train_y
                )
                if not converged:
                    non_converged_folds.append(fold.fold_index)
                predicted = _positive_probability(model, test_x[:, indices])
                by_fold[fold.fold_index] = predicted
                prior = np.full(len(test_y), float(np.mean(train_y)))
                fold_priors.append((fold.fold_index, prior))
                fold_reports.append(
                    {
                        "fold_index": fold.fold_index,
                        "metrics": binary_metrics(test_y, predicted),
                        "prior_metrics": binary_metrics(test_y, prior),
                        "optimizer_converged": converged,
                    }
                )
            selection_probability = np.concatenate(
                [by_fold[index] for index in MODEL_SELECTION_FOLDS]
            )
            selection_actual = np.concatenate(
                [outcome[np.asarray(folds[index].test_indices)] for index in MODEL_SELECTION_FOLDS]
            )
            selection_prior = np.concatenate(
                [dict(fold_priors)[index] for index in MODEL_SELECTION_FOLDS]
            )
            metrics = binary_metrics(selection_actual, selection_probability)
            prior_metrics = binary_metrics(selection_actual, selection_prior)
            skill = 1.0 - metrics["brier"] / prior_metrics["brier"]
            blockers = _binary_selection_blockers(
                metrics=metrics,
                prior=prior_metrics,
                skill=skill,
            )
            if non_converged_folds:
                blockers.append("MODEL_OPTIMIZER_DID_NOT_CONVERGE")
            comparisons[direction].append(
                {
                    "name": candidate,
                    "selection_metrics": metrics,
                    "selection_prior_metrics": prior_metrics,
                    "selection_brier_skill_vs_prior": skill,
                    "selection_viable": not blockers,
                    "selection_blockers": blockers,
                    "non_converged_folds": non_converged_folds,
                    "folds": fold_reports,
                }
            )
            probabilities[direction][candidate] = by_fold
        viable = [item for item in comparisons[direction] if item["selection_viable"]]
        pool = viable or comparisons[direction]
        selected_candidates[direction] = str(min(pool, key=_candidate_rank)["name"])

    calibration_reports: dict[str, list[dict[str, object]]] = {}
    selected_calibrations: dict[str, BinaryCalibrationArtifact] = {}
    for direction, outcome in labels.items():
        candidate = selected_candidates[direction]
        by_fold = probabilities[direction][candidate]
        fit_indices = np.asarray(folds[CALIBRATION_FIT_FOLD].test_indices)
        selection_indices = np.asarray(folds[CALIBRATION_SELECTION_FOLD].test_indices)
        fit_actual = outcome[fit_indices]
        selection_actual = outcome[selection_indices]
        fit_train_actual = outcome[np.asarray(folds[CALIBRATION_FIT_FOLD].train_indices)]
        prior = float(np.mean(fit_train_actual))
        raw_selection_metrics = binary_metrics(
            selection_actual, by_fold[CALIBRATION_SELECTION_FOLD]
        )
        reports = []
        artifacts = {}
        for method in CALIBRATION_METHODS:
            artifact = fit_binary_calibrator(
                method=method,
                probabilities=by_fold[CALIBRATION_FIT_FOLD],
                actual=fit_actual,
                prior=prior,
            )
            artifacts[method] = artifact
            transformed = apply_binary_calibrator(
                artifact, by_fold[CALIBRATION_SELECTION_FOLD]
            )
            metrics = binary_metrics(selection_actual, transformed)
            blockers = []
            if metrics["brier"] > raw_selection_metrics["brier"]:
                blockers.append("CALIBRATION_BRIER_DEGRADATION")
            if metrics["log_loss"] > raw_selection_metrics["log_loss"]:
                blockers.append("CALIBRATION_LOG_LOSS_DEGRADATION")
            if metrics["probability_std"] < 0.01:
                blockers.append("CALIBRATION_PROBABILITY_COLLAPSE")
            reports.append(
                {
                    "method": method,
                    "metrics": metrics,
                    "selection_viable": not blockers,
                    "selection_blockers": blockers,
                    "artifact": artifact.to_contract(),
                }
            )
        viable = [item for item in reports if item["selection_viable"]]
        pool = viable or [item for item in reports if item["method"] == "identity"]
        selected = min(
            pool,
            key=lambda item: (
                item["metrics"]["brier"],
                item["metrics"]["log_loss"],
                item["metrics"]["ece_10_bin"],
                item["method"],
            ),
        )
        selected_calibrations[direction] = artifacts[str(selected["method"])]
        calibration_reports[direction] = reports

    probability_diagnostics = {}
    calibrated_by_direction: dict[str, dict[int, np.ndarray]] = {}
    for direction, outcome in labels.items():
        candidate = selected_candidates[direction]
        artifact = selected_calibrations[direction]
        calibrated_by_direction[direction] = {
            fold.fold_index: apply_binary_calibrator(
                artifact, probabilities[direction][candidate][fold.fold_index]
            )
            for fold in folds
        }
        fold = folds[HISTORICAL_DIAGNOSTIC_FOLD]
        indices = np.asarray(fold.test_indices)
        actual = outcome[indices]
        calibrated = calibrated_by_direction[direction][HISTORICAL_DIAGNOSTIC_FOLD]
        raw = probabilities[direction][candidate][HISTORICAL_DIAGNOSTIC_FOLD]
        prior = np.full(len(actual), float(np.mean(outcome[np.asarray(fold.train_indices)])))
        raw_metrics = binary_metrics(actual, raw)
        calibrated_metrics = binary_metrics(actual, calibrated)
        prior_metrics = binary_metrics(actual, prior)
        skill = 1.0 - calibrated_metrics["brier"] / prior_metrics["brier"]
        interval = block_bootstrap_brier_skill(
            samples=tuple(samples[index] for index in indices),
            actual=actual,
            probabilities=calibrated,
            prior=prior,
        )
        selected_comparison = next(
            item
            for item in comparisons[direction]
            if item["name"] == candidate
        )
        positive_skill_folds = sum(
            item["metrics"]["brier"] < item["prior_metrics"]["brier"]
            for item in selected_comparison["folds"]
        )
        auc_above_random_folds = sum(
            item["metrics"]["roc_auc"] is not None
            and item["metrics"]["roc_auc"] > 0.5
            for item in selected_comparison["folds"]
        )
        regimes = binary_regime_diagnostics(
            matrix=matrix[indices],
            feature_names=features.feature_names,
            actual=actual,
            probabilities=calibrated,
        )
        blockers = _binary_final_blockers(
            raw=raw_metrics,
            calibrated=calibrated_metrics,
            prior=prior_metrics,
            skill=skill,
            skill_interval=interval,
            positive_skill_folds=positive_skill_folds,
            auc_above_random_folds=auc_above_random_folds,
        )
        for regime in regimes:
            if (
                regime["sample_count"] >= 200
                and regime["metrics"]["ece_10_bin"] > BINARY_GATE["maximum_ece"]
            ):
                blockers.append(f"{regime['regime']}_ECE_GATE_FAILED")
        technical = probabilities[direction]["technical_logistic_l2"][
            HISTORICAL_DIAGNOSTIC_FOLD
        ]
        technical_metrics = binary_metrics(actual, technical)
        probability_diagnostics[direction] = {
            "selected_candidate": candidate,
            "selected_calibration": artifact.method,
            "raw_metrics": raw_metrics,
            "calibrated_metrics": calibrated_metrics,
            "prior_metrics": prior_metrics,
            "technical_baseline_metrics": technical_metrics,
            "brier_skill_vs_prior": skill,
            "session_block_bootstrap_brier_skill_95": interval,
            "fold_stability": {
                "positive_brier_skill_folds": positive_skill_folds,
                "auc_above_random_folds": auc_above_random_folds,
                "required_of_five": 3,
            },
            "regime_diagnostics": regimes,
            "gate_passed": not blockers,
            "blockers": blockers,
        }

    selection_fold = folds[CALIBRATION_SELECTION_FOLD]
    selection_indices = np.asarray(selection_fold.test_indices)
    policy_comparison = []
    for activation in (0.45, 0.475, 0.50, 0.525, 0.55, 0.575, 0.60, 0.625, 0.65):
        for margin in (0.02, 0.05, 0.08, 0.10, 0.15):
            replay = simulate_directional_policy(
                samples=tuple(samples[index] for index in selection_indices),
                long_probabilities=calibrated_by_direction["LONG"][CALIBRATION_SELECTION_FOLD],
                short_probabilities=calibrated_by_direction["SHORT"][CALIBRATION_SELECTION_FOLD],
                long_paths=long_paths,
                short_paths=short_paths,
                activation_probability=activation,
                directional_margin=margin,
            )
            blockers = _policy_blockers(replay, selection=True)
            policy_comparison.append(
                {
                    "activation_probability": activation,
                    "directional_margin": margin,
                    "metrics": replay,
                    "selection_viable": not blockers,
                    "selection_blockers": blockers,
                }
            )
    viable_policies = [item for item in policy_comparison if item["selection_viable"]]
    policy_pool = viable_policies or policy_comparison
    selected_policy = min(policy_pool, key=_policy_rank)

    diagnostic_fold = folds[HISTORICAL_DIAGNOSTIC_FOLD]
    diagnostic_indices = np.asarray(diagnostic_fold.test_indices)
    historical_replay = simulate_directional_policy(
        samples=tuple(samples[index] for index in diagnostic_indices),
        long_probabilities=calibrated_by_direction["LONG"][HISTORICAL_DIAGNOSTIC_FOLD],
        short_probabilities=calibrated_by_direction["SHORT"][HISTORICAL_DIAGNOSTIC_FOLD],
        long_paths=long_paths,
        short_paths=short_paths,
        activation_probability=float(selected_policy["activation_probability"]),
        directional_margin=float(selected_policy["directional_margin"]),
    )
    replay_blockers = _policy_blockers(historical_replay, selection=False)
    model_blockers = [
        f"{direction}_{item}"
        for direction, report in probability_diagnostics.items()
        for item in report["blockers"]
    ]
    blockers = model_blockers + replay_blockers
    if not viable_policies:
        blockers.append("NO_POLICY_PASSED_SELECTION_GATE")
    blockers.extend(
        (
            "HISTORICAL_PERIOD_USED_FOR_RESEARCH",
            "FORWARD_CONFIRMATION_NOT_COMPLETED",
            "HISTORICAL_NEWS_NOT_AVAILABLE",
            "CROSS_MARKET_HISTORY_NOT_AVAILABLE",
        )
    )

    return {
        "schema_version": 1,
        "step18b_version": STEP18B_VERSION,
        "research_identity": RESEARCH_IDENTITY,
        "dataset_id": dataset.dataset_id,
        "target_definition": {
            "model_outputs": [
                "P(LONG target before LONG stop within 60m)",
                "P(SHORT target before SHORT stop within 60m)",
            ],
            "entry": "next finalized 1m open after finalized 5m decision",
            "target_atr": float(TARGET_ATR),
            "stop_atr": float(STOP_ATR),
            "horizon_minutes": HORIZON_MINUTES,
            "same_minute_resolution": "STOP_FIRST_CONSERVATIVE",
            "one_way_slippage_points": float(ONE_WAY_SLIPPAGE_POINTS),
            "expired_is_model_success": False,
            "target_policy_alignment": True,
        },
        "dataset": {
            "eligible_trade_aligned_samples": len(samples),
            "excluded_trade_paths": exclusions,
            "long_target_first_support": int(np.sum(labels["LONG"])),
            "short_target_first_support": int(np.sum(labels["SHORT"])),
        },
        "feature_architecture": {
            "version": RESEARCH_FEATURE_VERSION,
            "feature_set_hash": RESEARCH_FEATURE_SET_HASH,
            "feature_count": len(features.feature_names),
            "feature_names": list(features.feature_names),
            "absolute_price_features_removed": True,
            "finalized_candles_only": True,
            "developing_candle_used": False,
            "nifty_spot_volume_used": False,
            "vwap_used": False,
            "patterns_are_numeric_causal_features_not_chart_images": True,
        },
        "chronology": {
            "walk_forward": NESTED_WALK_FORWARD_CONFIG.to_contract(),
            "model_selection_folds": list(MODEL_SELECTION_FOLDS),
            "calibration_fit_fold": CALIBRATION_FIT_FOLD,
            "calibration_selection_and_policy_fold": CALIBRATION_SELECTION_FOLD,
            "historical_diagnostic_fold": HISTORICAL_DIAGNOSTIC_FOLD,
            "historical_diagnostic_starts_at": diagnostic_fold.test_starts_at.isoformat(),
            "historical_diagnostic_ends_at": diagnostic_fold.test_ends_at.isoformat(),
        },
        "candidate_comparison": comparisons,
        "selected_candidates": selected_candidates,
        "calibration_comparison": calibration_reports,
        "selected_calibrations": {
            direction: artifact.method
            for direction, artifact in selected_calibrations.items()
        },
        "probability_diagnostics": probability_diagnostics,
        "policy_selection": {
            "candidate_count": len(policy_comparison),
            "passing_candidate_count": len(viable_policies),
            "selected_policy": {
                "activation_probability": selected_policy["activation_probability"],
                "directional_margin": selected_policy["directional_margin"],
            },
            "selected_policy_metrics": selected_policy["metrics"],
            "selected_policy_blockers": selected_policy["selection_blockers"],
        },
        "historical_simulated_live_replay": historical_replay,
        "research_gate": {
            "binary_probability_gate": BINARY_GATE,
            "policy_gate": POLICY_GATE,
            "passed_before_mandatory_forward_and_data_context_blockers": not (
                model_blockers + replay_blockers + (
                    ["NO_POLICY_PASSED_SELECTION_GATE"] if not viable_policies else []
                )
            ),
            "blockers": blockers,
        },
        "known_data_limitations": {
            "historical_news_used": False,
            "cross_market_context_used": False,
            "nifty_constituent_breadth_used": False,
            "india_vix_used": False,
            "nifty_futures_volume_or_open_interest_used": False,
            "reason": "No point-in-time canonical historical datasets are available yet",
        },
        "existing_step17_runtime_modified": False,
        "existing_step18_report_modified": False,
        "model_artifact_created": False,
        "approved_for_live_inference": False,
        "precise_probability_display_allowed": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
    }


def build_trade_paths(
    *,
    dataset: DatasetBuildReport,
    features: ResearchFeatureMatrix,
    minute_candles: tuple[Candle, ...],
) -> tuple[
    tuple[TrainingSample, ...],
    np.ndarray,
    dict[str, TradePath],
    dict[str, TradePath],
    dict[str, int],
]:
    if features.sample_ids != tuple(item.sample_id for item in dataset.samples):
        raise ValueError("Research features are not aligned to dataset samples")
    minute_by_open = {item.opens_at: item for item in minute_candles}
    label_by_id = {item.label_id: item for item in dataset.labels}
    retained_samples = []
    retained_rows = []
    long_paths = {}
    short_paths = {}
    exclusions = Counter[str]()
    for index, sample in enumerate(dataset.samples):
        label = label_by_id[sample.label_id]
        atr = label.atr_at_decision
        if atr is None or atr <= 0:
            exclusions["ATR_UNAVAILABLE"] += 1
            continue
        entry = minute_by_open.get(sample.decision_time)
        if entry is None:
            exclusions["ENTRY_MINUTE_MISSING"] += 1
            continue
        window = tuple(
            minute_by_open.get(sample.decision_time + timedelta(minutes=offset))
            for offset in range(HORIZON_MINUTES)
        )
        if any(item is None for item in window):
            exclusions["INCOMPLETE_MINUTE_WINDOW"] += 1
            continue
        complete_window = tuple(item for item in window if item is not None)
        long_path = _trade_path(sample, "LONG", entry, complete_window, atr)
        short_path = _trade_path(sample, "SHORT", entry, complete_window, atr)
        retained_samples.append(sample)
        retained_rows.append(features.rows[index])
        long_paths[sample.sample_id] = long_path
        short_paths[sample.sample_id] = short_path
    return (
        tuple(retained_samples),
        np.asarray(retained_rows, dtype=float),
        long_paths,
        short_paths,
        dict(exclusions),
    )


def _trade_path(
    sample: TrainingSample,
    direction: str,
    entry: Candle,
    window: tuple[Candle, ...],
    atr: Decimal,
) -> TradePath:
    sign = Decimal("1") if direction == "LONG" else Decimal("-1")
    stop = entry.open - sign * STOP_ATR * atr
    target = entry.open + sign * TARGET_ATR * atr
    exit_price = window[-1].close
    exited_at = window[-1].closes_at
    reason = "HORIZON"
    for candle in window:
        target_hit = candle.high >= target if direction == "LONG" else candle.low <= target
        stop_hit = candle.low <= stop if direction == "LONG" else candle.high >= stop
        if stop_hit:
            exit_price, exited_at, reason = stop, candle.closes_at, "STOP"
            break
        if target_hit:
            exit_price, exited_at, reason = target, candle.closes_at, "TARGET"
            break
    effective_entry = entry.open + sign * ONE_WAY_SLIPPAGE_POINTS
    effective_exit = exit_price - sign * ONE_WAY_SLIPPAGE_POINTS
    net = sign * (effective_exit - effective_entry)
    return TradePath(
        sample_id=sample.sample_id,
        decision_time=sample.decision_time,
        direction=direction,
        success=1 if reason == "TARGET" else 0,
        exit_reason=reason,
        entered_at=entry.opens_at,
        exited_at=exited_at,
        entry=float(entry.open),
        stop=float(stop),
        target=float(target),
        exit=float(exit_price),
        net_points=float(net),
        r_multiple=float(net / (STOP_ATR * atr)),
    )


def binary_metrics(actual: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    actual = np.asarray(actual, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    predicted = probabilities >= 0.5
    return {
        "sample_count": len(actual),
        "positive_support": int(np.sum(actual)),
        "positive_rate": float(np.mean(actual)),
        "accuracy_at_0p5": float(np.mean(predicted == actual)),
        "brier": float(np.mean((probabilities - actual) ** 2)),
        "log_loss": float(log_loss(actual, probabilities, labels=[0, 1])),
        "roc_auc": (
            float(roc_auc_score(actual, probabilities)) if len(set(actual)) == 2 else None
        ),
        "average_precision": (
            float(average_precision_score(actual, probabilities))
            if np.any(actual)
            else None
        ),
        "ece_10_bin": _binary_ece(actual, probabilities),
        "probability_mean": float(np.mean(probabilities)),
        "probability_std": float(np.std(probabilities)),
        "predicted_positive_share": float(np.mean(predicted)),
    }


def fit_binary_calibrator(
    *, method: str, probabilities: np.ndarray, actual: np.ndarray, prior: float
) -> BinaryCalibrationArtifact:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    actual = np.asarray(actual, dtype=int)
    if method == "identity":
        return BinaryCalibrationArtifact(method, {})
    if method == "platt":
        model = LogisticRegression(
            C=1.0, max_iter=2_000, random_state=RANDOM_SEED, solver="lbfgs"
        )
        model.fit(_logit(probabilities).reshape(-1, 1), actual)
        return BinaryCalibrationArtifact(
            method,
            {"coefficient": float(model.coef_[0, 0]), "intercept": float(model.intercept_[0])},
        )
    if method == "beta":
        model = LogisticRegression(
            C=1.0, max_iter=2_000, random_state=RANDOM_SEED, solver="lbfgs"
        )
        x = np.column_stack((np.log(probabilities), np.log1p(-probabilities)))
        model.fit(x, actual)
        return BinaryCalibrationArtifact(
            method,
            {"coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])},
        )
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(probabilities, actual)
        return BinaryCalibrationArtifact(
            method,
            {"x_thresholds": model.X_thresholds_.tolist(), "y_thresholds": model.y_thresholds_.tolist()},
        )
    if method == "prior_shrinkage":
        best_alpha = min(
            (index / 100 for index in range(101)),
            key=lambda alpha: float(
                np.mean((alpha * probabilities + (1.0 - alpha) * prior - actual) ** 2)
            ),
        )
        return BinaryCalibrationArtifact(
            method, {"alpha": best_alpha, "prior": prior}
        )
    raise ValueError(f"Unknown binary calibration method: {method}")


def apply_binary_calibrator(
    artifact: BinaryCalibrationArtifact, probabilities: np.ndarray
) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    if artifact.method == "identity":
        return probabilities.copy()
    if artifact.method == "platt":
        logits = (
            float(artifact.parameters["coefficient"]) * _logit(probabilities)
            + float(artifact.parameters["intercept"])
        )
        return _sigmoid(logits)
    if artifact.method == "beta":
        coefficients = np.asarray(artifact.parameters["coefficients"], dtype=float)
        x = np.column_stack((np.log(probabilities), np.log1p(-probabilities)))
        return _sigmoid(x @ coefficients + float(artifact.parameters["intercept"]))
    if artifact.method == "isotonic":
        return np.interp(
            probabilities,
            np.asarray(artifact.parameters["x_thresholds"], dtype=float),
            np.asarray(artifact.parameters["y_thresholds"], dtype=float),
        )
    if artifact.method == "prior_shrinkage":
        alpha = float(artifact.parameters["alpha"])
        return alpha * probabilities + (1.0 - alpha) * float(artifact.parameters["prior"])
    raise ValueError(f"Unknown binary calibration method: {artifact.method}")


def block_bootstrap_brier_skill(
    *,
    samples: tuple[TrainingSample, ...],
    actual: np.ndarray,
    probabilities: np.ndarray,
    prior: np.ndarray,
    iterations: int = 1_000,
) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[str(sample.decision_time.astimezone(IST).date())].append(index)
    names = tuple(sorted(groups))
    if len(names) < 2:
        return {"lower": -1.0, "median": -1.0, "upper": -1.0, "session_count": len(names)}
    rng = np.random.default_rng(RANDOM_SEED)
    skills = []
    actual = np.asarray(actual, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    prior = np.asarray(prior, dtype=float)
    for _ in range(iterations):
        selected_days = rng.choice(names, size=len(names), replace=True)
        indices = np.asarray([index for day in selected_days for index in groups[str(day)]])
        model_loss = float(np.mean((probabilities[indices] - actual[indices]) ** 2))
        prior_loss = float(np.mean((prior[indices] - actual[indices]) ** 2))
        skills.append(1.0 - model_loss / prior_loss if prior_loss > 0 else -1.0)
    lower, median, upper = np.quantile(skills, (0.025, 0.5, 0.975))
    return {
        "lower": float(lower),
        "median": float(median),
        "upper": float(upper),
        "session_count": len(names),
    }


def binary_regime_diagnostics(
    *,
    matrix: np.ndarray,
    feature_names: tuple[str, ...],
    actual: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, object]]:
    index = feature_names.index("enhanced__trend_alignment")
    alignment = matrix[:, index]
    definitions = (
        ("BEAR_ALIGNED", alignment <= -0.66),
        ("MIXED", (alignment > -0.66) & (alignment < 0.66)),
        ("BULL_ALIGNED", alignment >= 0.66),
    )
    reports = []
    for name, mask in definitions:
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            reports.append({"regime": name, "sample_count": 0, "metrics": None})
            continue
        reports.append(
            {
                "regime": name,
                "sample_count": len(indices),
                "metrics": binary_metrics(actual[indices], probabilities[indices]),
            }
        )
    return reports


def simulate_directional_policy(
    *,
    samples: tuple[TrainingSample, ...],
    long_probabilities: np.ndarray,
    short_probabilities: np.ndarray,
    long_paths: dict[str, TradePath],
    short_paths: dict[str, TradePath],
    activation_probability: float,
    directional_margin: float,
) -> dict[str, object]:
    trades = []
    waits = Counter[str]()
    active_until: datetime | None = None
    for index, sample in enumerate(samples):
        if active_until is not None and sample.decision_time < active_until:
            waits["ACTIVE_POSITION"] += 1
            continue
        long_probability = float(long_probabilities[index])
        short_probability = float(short_probabilities[index])
        best = max(long_probability, short_probability)
        if best < activation_probability:
            waits["PROBABILITY_BELOW_THRESHOLD"] += 1
            continue
        if abs(long_probability - short_probability) < directional_margin:
            waits["DIRECTIONAL_MARGIN_TOO_SMALL"] += 1
            continue
        path = (
            long_paths[sample.sample_id]
            if long_probability > short_probability
            else short_paths[sample.sample_id]
        )
        trades.append(path)
        active_until = path.exited_at
    return _replay_metrics(
        trades=tuple(trades),
        decision_count=len(samples),
        waits=waits,
        activation_probability=activation_probability,
        directional_margin=directional_margin,
    )


def _replay_metrics(
    *,
    trades: tuple[TradePath, ...],
    decision_count: int,
    waits: Counter[str],
    activation_probability: float,
    directional_margin: float,
) -> dict[str, object]:
    points = [item.net_points for item in trades]
    r_values = [item.r_multiple for item in trades]
    positives = [item for item in points if item > 0]
    negatives = [item for item in points if item < 0]
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in r_values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    interval = _trade_session_bootstrap(trades)
    return {
        "policy": {
            "activation_probability": activation_probability,
            "directional_margin": directional_margin,
            "target_atr": float(TARGET_ATR),
            "stop_atr": float(STOP_ATR),
            "horizon_minutes": HORIZON_MINUTES,
            "one_way_slippage_points": float(ONE_WAY_SLIPPAGE_POINTS),
            "overlapping_positions": False,
        },
        "evaluation_decisions": decision_count,
        "trade_count": len(trades),
        "buy_count": sum(item.direction == "LONG" for item in trades),
        "sell_count": sum(item.direction == "SHORT" for item in trades),
        "coverage": len(trades) / decision_count if decision_count else 0.0,
        "target_hit_count": sum(item.exit_reason == "TARGET" for item in trades),
        "stop_hit_count": sum(item.exit_reason == "STOP" for item in trades),
        "expired_count": sum(item.exit_reason == "HORIZON" for item in trades),
        "win_rate": len(positives) / len(trades) if trades else None,
        "net_points": float(sum(points)),
        "average_points": float(np.mean(points)) if points else None,
        "average_r_multiple": float(np.mean(r_values)) if r_values else None,
        "average_r_session_bootstrap_95": interval,
        "profit_factor": (
            sum(positives) / abs(sum(negatives)) if positives and negatives else None
        ),
        "maximum_drawdown_r": drawdown,
        "wait_counts": dict(waits),
        "hypothetical_index_points_only": True,
        "rupee_pnl_available": False,
    }


def _trade_session_bootstrap(
    trades: tuple[TradePath, ...], iterations: int = 1_000
) -> dict[str, float | int | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        groups[str(trade.decision_time.astimezone(IST).date())].append(trade.r_multiple)
    names = tuple(sorted(groups))
    if len(names) < 2:
        return {"lower": None, "median": None, "upper": None, "session_count": len(names)}
    rng = np.random.default_rng(RANDOM_SEED)
    averages = []
    for _ in range(iterations):
        selected = rng.choice(names, size=len(names), replace=True)
        values = [value for day in selected for value in groups[str(day)]]
        averages.append(float(np.mean(values)))
    lower, median, upper = np.quantile(averages, (0.025, 0.5, 0.975))
    return {
        "lower": float(lower),
        "median": float(median),
        "upper": float(upper),
        "session_count": len(names),
    }


def _candidate(
    name: str, feature_names: tuple[str, ...]
) -> tuple[object, np.ndarray]:
    all_indices = np.arange(len(feature_names))
    if name == "technical_logistic_l2":
        wanted = {
            "primary_5m__return_1",
            "primary_5m__roc_5",
            "primary_5m__rsi_14",
            "primary_5m__distance_ema20_atr",
            "primary_5m__atr_pct",
            "context_15m__roc_5",
            "context_1h__roc_5",
            "enhanced__trend_alignment",
            "enhanced__momentum_alignment",
            "research_v3__macd_histogram_atr",
            "research_v3__adx_14",
        }
        indices = np.asarray([index for index, item in enumerate(feature_names) if item in wanted])
        return _logistic(c=0.1), indices
    if name == "stationary_logistic_l2_c0p01":
        return _logistic(c=0.01), all_indices
    if name == "stationary_logistic_l2_c0p1":
        return _logistic(c=0.1), all_indices
    if name == "stationary_logistic_l2_c1":
        return _logistic(c=1.0), all_indices
    if name == "stationary_elasticnet_c0p1_l1r0p25":
        return _elastic(c=0.1, ratio=0.25), all_indices
    if name == "stationary_elasticnet_c1_l1r0p5":
        return _elastic(c=1.0, ratio=0.5), all_indices
    if name == "stationary_hgb_shallow":
        return HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=160,
            max_leaf_nodes=7,
            min_samples_leaf=50,
            l2_regularization=2.0,
            random_state=RANDOM_SEED,
        ), all_indices
    if name == "stationary_hgb_medium":
        return HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=3.0,
            random_state=RANDOM_SEED,
        ), all_indices
    raise ValueError(f"Unknown Step 18B candidate: {name}")


def _logistic(*, c: float) -> Pipeline:
    return Pipeline(
        (("scale", StandardScaler()), ("model", LogisticRegression(C=c, max_iter=2_000, random_state=RANDOM_SEED, solver="lbfgs")))
    )


def _elastic(*, c: float, ratio: float) -> Pipeline:
    return Pipeline(
        (("scale", StandardScaler()), ("model", LogisticRegression(C=c, l1_ratio=ratio, max_iter=10_000, tol=1e-3, random_state=RANDOM_SEED, solver="saga")))
    )


def _fit_with_convergence_check(
    model: object, matrix: np.ndarray, labels: np.ndarray
) -> bool:
    """Fit once and make non-convergence an explicit research result."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(matrix, labels)
    return not any(issubclass(item.category, ConvergenceWarning) for item in caught)


def _positive_probability(model: object, matrix: np.ndarray) -> np.ndarray:
    classes = tuple(int(item) for item in model.classes_)
    return model.predict_proba(matrix)[:, classes.index(1)]


def _binary_selection_blockers(
    *, metrics: dict[str, object], prior: dict[str, object], skill: float
) -> list[str]:
    blockers = []
    if metrics["positive_support"] < BINARY_GATE["minimum_positive_support"]:
        blockers.append("POSITIVE_SUPPORT_TOO_LOW")
    if metrics["roc_auc"] is None or metrics["roc_auc"] <= BINARY_GATE["minimum_roc_auc"]:
        blockers.append("ROC_AUC_GATE_FAILED")
    if skill <= BINARY_GATE["minimum_brier_skill_vs_prior"]:
        blockers.append("BRIER_SKILL_GATE_FAILED")
    if metrics["log_loss"] >= prior["log_loss"]:
        blockers.append("LOG_LOSS_GATE_FAILED")
    if metrics["probability_std"] < BINARY_GATE["minimum_probability_standard_deviation"]:
        blockers.append("PROBABILITY_DISPERSION_TOO_LOW")
    return blockers


def _binary_final_blockers(
    *,
    raw: dict[str, object],
    calibrated: dict[str, object],
    prior: dict[str, object],
    skill: float,
    skill_interval: dict[str, float],
    positive_skill_folds: int,
    auc_above_random_folds: int,
) -> list[str]:
    blockers = _binary_selection_blockers(metrics=calibrated, prior=prior, skill=skill)
    if calibrated["ece_10_bin"] > BINARY_GATE["maximum_ece"]:
        blockers.append("ECE_GATE_FAILED")
    if skill_interval["lower"] <= BINARY_GATE["minimum_lower_95_brier_skill"]:
        blockers.append("BRIER_SKILL_LOWER_CONFIDENCE_BOUND_NOT_POSITIVE")
    if positive_skill_folds < 3:
        blockers.append("INSUFFICIENT_POSITIVE_BRIER_SKILL_FOLDS")
    if auc_above_random_folds < 3:
        blockers.append("INSUFFICIENT_AUC_ABOVE_RANDOM_FOLDS")
    if calibrated["brier"] > raw["brier"] or calibrated["log_loss"] > raw["log_loss"]:
        blockers.append("CALIBRATION_DEGRADES_RAW_PROPER_SCORES")
    return blockers


def _policy_blockers(metrics: dict[str, object], *, selection: bool) -> list[str]:
    prefix = "selection" if selection else "diagnostic"
    blockers = []
    if metrics["trade_count"] < POLICY_GATE[f"minimum_{prefix}_trades"]:
        blockers.append(f"{prefix.upper()}_TRADE_SUPPORT_TOO_LOW")
    if metrics["buy_count"] < POLICY_GATE[f"minimum_{prefix}_buys"]:
        blockers.append(f"{prefix.upper()}_BUY_SUPPORT_TOO_LOW")
    if metrics["sell_count"] < POLICY_GATE[f"minimum_{prefix}_sells"]:
        blockers.append(f"{prefix.upper()}_SELL_SUPPORT_TOO_LOW")
    profit_factor = metrics["profit_factor"]
    if profit_factor is None or profit_factor <= POLICY_GATE[f"minimum_{prefix}_profit_factor"]:
        blockers.append(f"{prefix.upper()}_PROFIT_FACTOR_GATE_FAILED")
    lower = metrics["average_r_session_bootstrap_95"]["lower"]
    if lower is None or lower <= POLICY_GATE[f"minimum_{prefix}_lower_95_average_r"]:
        blockers.append(f"{prefix.upper()}_EXPECTANCY_CONFIDENCE_GATE_FAILED")
    return blockers


def _candidate_rank(item: dict[str, object]) -> tuple[object, ...]:
    metrics = item["selection_metrics"]
    return (
        not bool(item["selection_viable"]),
        metrics["brier"],
        metrics["log_loss"],
        -float(metrics["roc_auc"] or 0.0),
        item["name"],
    )


def _policy_rank(item: dict[str, object]) -> tuple[object, ...]:
    metrics = item["metrics"]
    lower = metrics["average_r_session_bootstrap_95"]["lower"]
    return (
        not bool(item["selection_viable"]),
        -(lower if lower is not None else -1_000.0),
        -(metrics["average_r_multiple"] if metrics["average_r_multiple"] is not None else -1_000.0),
        -(metrics["profit_factor"] if metrics["profit_factor"] is not None else 0.0),
        -min(metrics["buy_count"], metrics["sell_count"]),
        item["activation_probability"],
        item["directional_margin"],
    )


def _binary_ece(actual: np.ndarray, probabilities: np.ndarray) -> float:
    error = 0.0
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == 9 else probabilities < upper
        )
        if np.any(mask):
            error += float(np.mean(mask)) * abs(
                float(np.mean(actual[mask])) - float(np.mean(probabilities[mask]))
            )
    return error


def _logit(probabilities: np.ndarray) -> np.ndarray:
    return np.log(probabilities / (1.0 - probabilities))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))
