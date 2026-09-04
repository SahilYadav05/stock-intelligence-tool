"""Locked-candidate calibration research, shadow artifact, and signal backtest."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
import json

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nifty_terminal.calibration.temperature import apply_temperature, fit_temperature
from nifty_terminal.domain.candle import Candle
from nifty_terminal.features.definitions import FEATURE_SET_HASH, FEATURE_VERSION
from nifty_terminal.ml.definitions import CLASS_ORDER, RANDOM_SEED
from nifty_terminal.ml.metrics import calculate_metrics, prior_probabilities
from nifty_terminal.ml.models import DatasetBuildReport, MetricSummary, TrainingSample
from nifty_terminal.ml.split import PurgedWalkForwardSplitter
from nifty_terminal.research.v2 import NESTED_WALK_FORWARD_CONFIG


STEP16_VERSION = "locked_shadow_research.v1"
LOCKED_ATR_MULTIPLIER = Decimal("1.5")
LOCKED_CANDIDATE = "multinomial_logistic_unweighted"
CALIBRATION_METHODS = ("identity", "temperature", "prior_shrinkage", "vector_scaling")
CALIBRATION_FIT_FOLD = 2
CALIBRATION_SELECTION_FOLD = 3
HISTORICAL_BACKTEST_FOLD = 4
POLICY_DEFINITION = {
    "activation_probability": 0.60,
    "minimum_directional_margin": 0.15,
    "maximum_neither_probability": 0.45,
    "stop_atr": 0.75,
    "target_atr": 1.0,
    "maximum_holding_minutes": 60,
    "one_way_slippage_points": 0.5,
    "same_minute_stop_and_target": "STOP_FIRST_CONSERVATIVE",
    "overlapping_positions": False,
    "instrument_note": "NIFTY spot is not directly tradable; results are points/R only",
}


@dataclass(frozen=True, slots=True)
class CalibrationArtifactV2:
    method: str
    parameters: dict[str, object]

    def to_contract(self) -> dict[str, object]:
        return {
            "method": self.method,
            "parameters": self.parameters,
            "safe_json_parameters_only": True,
        }


@dataclass(frozen=True, slots=True)
class CalibrationMethodEvaluation:
    artifact: CalibrationArtifactV2
    selection_metrics: MetricSummary
    selection_brier_skill: float

    def to_contract(self) -> dict[str, object]:
        return {
            "artifact": self.artifact.to_contract(),
            "selection_metrics": self.selection_metrics.to_contract(),
            "selection_brier_skill": self.selection_brier_skill,
        }


def run_locked_research(
    *,
    dataset: DatasetBuildReport,
    minute_candles: tuple[Candle, ...],
) -> dict[str, object]:
    samples = dataset.samples
    folds = PurgedWalkForwardSplitter().split(samples, NESTED_WALK_FORWARD_CONFIG)
    probabilities = {
        fold.fold_index: _fold_probabilities(samples, fold.train_indices, fold.test_indices)
        for fold in folds
    }
    actual = {
        fold.fold_index: tuple(samples[index].outcome.value for index in fold.test_indices)
        for fold in folds
    }

    fit_fold = folds[CALIBRATION_FIT_FOLD]
    fit_training_actual = tuple(
        samples[index].outcome.value for index in fit_fold.train_indices
    )
    fit_prior = _prior_vector(fit_training_actual)
    method_evaluations = []
    for method in CALIBRATION_METHODS:
        artifact = fit_calibrator(
            method=method,
            probabilities=probabilities[CALIBRATION_FIT_FOLD],
            actual=actual[CALIBRATION_FIT_FOLD],
            prior=fit_prior,
        )
        selected_probabilities = apply_calibrator(
            artifact,
            probabilities[CALIBRATION_SELECTION_FOLD],
        )
        selection_metrics = calculate_metrics(
            actual[CALIBRATION_SELECTION_FOLD],
            selected_probabilities,
        )
        selection_fold = folds[CALIBRATION_SELECTION_FOLD]
        selection_prior = prior_probabilities(
            tuple(samples[index].outcome.value for index in selection_fold.train_indices),
            len(selection_fold.test_indices),
        )
        prior_metrics = calculate_metrics(
            actual[CALIBRATION_SELECTION_FOLD],
            selection_prior,
        )
        method_evaluations.append(
            CalibrationMethodEvaluation(
                artifact=artifact,
                selection_metrics=selection_metrics,
                selection_brier_skill=(
                    1.0
                    - selection_metrics.multiclass_brier
                    / prior_metrics.multiclass_brier
                ),
            )
        )
    selected = min(
        method_evaluations,
        key=lambda item: (
            item.selection_metrics.multiclass_brier,
            item.selection_metrics.log_loss,
            item.selection_metrics.raw_ece_10_bin,
            item.artifact.method,
        ),
    )

    final_fold = folds[HISTORICAL_BACKTEST_FOLD]
    final_raw = probabilities[HISTORICAL_BACKTEST_FOLD]
    final_calibrated = apply_calibrator(selected.artifact, final_raw)
    final_actual = actual[HISTORICAL_BACKTEST_FOLD]
    final_prior = prior_probabilities(
        tuple(samples[index].outcome.value for index in final_fold.train_indices),
        len(final_fold.test_indices),
    )
    raw_metrics = calculate_metrics(final_actual, final_raw)
    calibrated_metrics = calculate_metrics(final_actual, final_calibrated)
    prior_metrics = calculate_metrics(final_actual, final_prior)
    brier_skill = 1.0 - calibrated_metrics.multiclass_brier / prior_metrics.multiclass_brier
    calibration_blockers = _calibration_blockers(
        raw=raw_metrics,
        calibrated=calibrated_metrics,
        prior=prior_metrics,
        brier_skill=brier_skill,
    )
    final_samples = tuple(samples[index] for index in final_fold.test_indices)
    backtest = simulate_signals(
        samples=final_samples,
        probabilities=final_calibrated,
        dataset=dataset,
        minute_candles=minute_candles,
    )

    combined_probabilities = np.vstack(
        [probabilities[CALIBRATION_FIT_FOLD], probabilities[CALIBRATION_SELECTION_FOLD]]
    )
    combined_actual = (
        actual[CALIBRATION_FIT_FOLD] + actual[CALIBRATION_SELECTION_FOLD]
    )
    shadow_prior = _prior_vector(combined_actual)
    shadow_calibrator = fit_calibrator(
        method=selected.artifact.method,
        probabilities=combined_probabilities,
        actual=combined_actual,
        prior=shadow_prior,
    )
    shadow_artifact = build_shadow_artifact(
        dataset=dataset,
        calibrator=shadow_calibrator,
    )
    return {
        "schema_version": 1,
        "step16_version": STEP16_VERSION,
        "dataset_id": dataset.dataset_id,
        "locked_specification": {
            "atr_multiplier": format(LOCKED_ATR_MULTIPLIER, "f"),
            "horizon_minutes": 60,
            "candidate": LOCKED_CANDIDATE,
            "locked_from_step15_screening": True,
            "step15_ranking_correction": (
                "Target definitions are ranked by Brier skill versus each target's "
                "own historical-prior baseline, not by incomparable absolute Brier scores"
            ),
        },
        "nested_timeline": {
            "calibration_fit_fold": CALIBRATION_FIT_FOLD,
            "calibration_fit_starts_at": fit_fold.test_starts_at.isoformat(),
            "calibration_fit_ends_at": fit_fold.test_ends_at.isoformat(),
            "calibration_selection_fold": CALIBRATION_SELECTION_FOLD,
            "calibration_selection_starts_at": folds[3].test_starts_at.isoformat(),
            "calibration_selection_ends_at": folds[3].test_ends_at.isoformat(),
            "historical_backtest_fold": HISTORICAL_BACKTEST_FOLD,
            "historical_backtest_starts_at": final_fold.test_starts_at.isoformat(),
            "historical_backtest_ends_at": final_fold.test_ends_at.isoformat(),
        },
        "calibration_method_comparison": [
            item.to_contract() for item in method_evaluations
        ],
        "selected_calibration_method": selected.artifact.method,
        "historical_backtest_probability_metrics": {
            "raw": raw_metrics.to_contract(),
            "calibrated": calibrated_metrics.to_contract(),
            "prior": prior_metrics.to_contract(),
            "brier_skill_vs_prior": brier_skill,
            "blockers": calibration_blockers,
        },
        "signal_backtest": backtest,
        "shadow_artifact": shadow_artifact,
        "historical_period_previously_used_for_target_screening": True,
        "forward_confirmation_required": True,
        "minimum_forward_confirmation": {
            "starts_after": final_fold.test_ends_at.isoformat(),
            "minimum_eligible_predictions": 2_000,
            "minimum_distinct_sessions": 60,
            "selection_or_threshold_changes_restart_confirmation": True,
        },
        "approved_for_live_inference": False,
        "precise_probability_display_allowed": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
        "news_used": False,
        "nifty_spot_volume_used": False,
    }


def fit_calibrator(
    *,
    method: str,
    probabilities: np.ndarray,
    actual: tuple[str, ...],
    prior: np.ndarray,
) -> CalibrationArtifactV2:
    indices = np.asarray([CLASS_ORDER.index(name) for name in actual], dtype=int)
    if method == "identity":
        return CalibrationArtifactV2(method, {})
    if method == "temperature":
        return CalibrationArtifactV2(
            method,
            {"temperature": fit_temperature(probabilities, indices)},
        )
    if method == "prior_shrinkage":
        best_alpha = min(
            (index / 100 for index in range(101)),
            key=lambda alpha: _negative_log_likelihood(
                alpha * probabilities + (1.0 - alpha) * prior,
                indices,
            ),
        )
        return CalibrationArtifactV2(
            method,
            {"alpha": best_alpha, "prior": prior.tolist()},
        )
    if method == "vector_scaling":
        features = np.log(np.clip(probabilities, 1e-12, 1.0))
        model = LogisticRegression(
            C=0.1,
            max_iter=2_000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )
        model.fit(features, np.asarray(actual))
        return CalibrationArtifactV2(
            method,
            {
                "classes": [str(item) for item in model.classes_],
                "coefficients": model.coef_.tolist(),
                "intercepts": model.intercept_.tolist(),
            },
        )
    raise ValueError(f"Unknown calibration method: {method}")


def apply_calibrator(
    artifact: CalibrationArtifactV2,
    probabilities: np.ndarray,
) -> np.ndarray:
    if artifact.method == "identity":
        return probabilities.copy()
    if artifact.method == "temperature":
        return apply_temperature(
            probabilities,
            float(artifact.parameters["temperature"]),
        )
    if artifact.method == "prior_shrinkage":
        alpha = float(artifact.parameters["alpha"])
        prior = np.asarray(artifact.parameters["prior"], dtype=float)
        return alpha * probabilities + (1.0 - alpha) * prior
    if artifact.method == "vector_scaling":
        coefficients = np.asarray(artifact.parameters["coefficients"], dtype=float)
        intercepts = np.asarray(artifact.parameters["intercepts"], dtype=float)
        classes = tuple(str(item) for item in artifact.parameters["classes"])
        features = np.log(np.clip(probabilities, 1e-12, 1.0))
        logits = features @ coefficients.T + intercepts
        logits -= logits.max(axis=1, keepdims=True)
        transformed = np.exp(logits)
        transformed /= transformed.sum(axis=1, keepdims=True)
        return transformed[:, [classes.index(name) for name in CLASS_ORDER]]
    raise ValueError(f"Unknown calibration method: {artifact.method}")


def build_shadow_artifact(
    *,
    dataset: DatasetBuildReport,
    calibrator: CalibrationArtifactV2,
) -> dict[str, object]:
    samples = dataset.samples
    pipeline = _model()
    x = np.asarray([item.feature_values for item in samples], dtype=float)
    y = np.asarray([item.outcome.value for item in samples])
    pipeline.fit(x, y)
    scaler: StandardScaler = pipeline.named_steps["scale"]
    model: LogisticRegression = pipeline.named_steps["model"]
    artifact = {
        "schema_version": 1,
        "artifact_version": "nifty_shadow_logistic_1p5.v1",
        "dataset_id": dataset.dataset_id,
        "candidate": LOCKED_CANDIDATE,
        "atr_multiplier": format(LOCKED_ATR_MULTIPLIER, "f"),
        "horizon_minutes": 60,
        "trained_through": samples[-1].decision_time.isoformat(),
        "labels_known_through": samples[-1].label_window_end.isoformat(),
        "feature_names": list(samples[0].feature_names),
        "feature_version": FEATURE_VERSION,
        "feature_set_hash": FEATURE_SET_HASH,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "classes": [str(item) for item in model.classes_],
        "coefficients": model.coef_.tolist(),
        "intercepts": model.intercept_.tolist(),
        "calibration": calibrator.to_contract(),
        "runtime": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "serialization": "SAFE_JSON_PARAMETERS_ONLY",
        "shadow_only": True,
        "approved_for_live_inference": False,
    }
    checksum = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**artifact, "sha256": checksum}


def simulate_signals(
    *,
    samples: tuple[TrainingSample, ...],
    probabilities: np.ndarray,
    dataset: DatasetBuildReport,
    minute_candles: tuple[Candle, ...],
) -> dict[str, object]:
    if len(samples) != len(probabilities):
        raise ValueError("Signal backtest requires aligned samples and probabilities")
    minute_by_open = {item.opens_at: item for item in minute_candles}
    label_by_id = {item.label_id: item for item in dataset.labels}
    trades = []
    waits = Counter[str]()
    active_until: datetime | None = None
    for row_index, sample in enumerate(samples):
        if active_until is not None and sample.decision_time < active_until:
            waits["ACTIVE_POSITION"] += 1
            continue
        row = {name: probabilities[row_index, index] for index, name in enumerate(CLASS_ORDER)}
        direction = _direction(row)
        if direction is None:
            waits["PROBABILITY_POLICY_WAIT"] += 1
            continue
        entry_candle = minute_by_open.get(sample.decision_time)
        if entry_candle is None:
            waits["ENTRY_MINUTE_MISSING"] += 1
            continue
        label = label_by_id[sample.label_id]
        atr = label.atr_at_decision
        if atr is None or atr <= 0:
            waits["ATR_UNAVAILABLE"] += 1
            continue
        trade = _simulate_trade(
            sample=sample,
            direction=direction,
            probability=max(row["UP"], row["DOWN"]),
            entry=entry_candle,
            atr=atr,
            minute_by_open=minute_by_open,
        )
        if trade is None:
            waits["INCOMPLETE_MINUTE_WINDOW"] += 1
            continue
        trades.append(trade)
        active_until = datetime.fromisoformat(str(trade["exited_at"]))

    net = [float(item["net_points"]) for item in trades]
    positive = [item for item in net if item > 0]
    negative = [item for item in net if item < 0]
    cumulative = 0.0
    peak = 0.0
    maximum_drawdown = 0.0
    maximum_consecutive_losses = 0
    consecutive_losses = 0
    for value in net:
        cumulative += value
        peak = max(peak, cumulative)
        maximum_drawdown = max(maximum_drawdown, peak - cumulative)
        consecutive_losses = consecutive_losses + 1 if value < 0 else 0
        maximum_consecutive_losses = max(maximum_consecutive_losses, consecutive_losses)
    return {
        "policy": POLICY_DEFINITION,
        "evaluation_decisions": len(samples),
        "trade_count": len(trades),
        "buy_count": sum(item["direction"] == "BUY" for item in trades),
        "sell_count": sum(item["direction"] == "SELL" for item in trades),
        "wait_counts": dict(waits),
        "target_hit_count": sum(item["exit_reason"] == "TARGET" for item in trades),
        "stop_hit_count": sum(item["exit_reason"] == "STOP" for item in trades),
        "expired_count": sum(item["exit_reason"] == "HORIZON" for item in trades),
        "win_rate": len(positive) / len(net) if net else None,
        "net_points": sum(net),
        "average_points": sum(net) / len(net) if net else None,
        "average_r_multiple": (
            sum(float(item["r_multiple"]) for item in trades) / len(trades)
            if trades
            else None
        ),
        "profit_factor": (
            sum(positive) / abs(sum(negative))
            if positive and negative
            else None
        ),
        "maximum_drawdown_points": maximum_drawdown,
        "maximum_consecutive_losses": maximum_consecutive_losses,
        "hypothetical_index_points_only": True,
        "rupee_pnl_available": False,
        "trades": trades,
    }


def _simulate_trade(
    *,
    sample: TrainingSample,
    direction: str,
    probability: float,
    entry: Candle,
    atr: Decimal,
    minute_by_open: dict[datetime, Candle],
) -> dict[str, object] | None:
    sign = Decimal("1") if direction == "BUY" else Decimal("-1")
    stop = entry.open - sign * Decimal("0.75") * atr
    target = entry.open + sign * atr
    window = []
    for offset in range(60):
        candle = minute_by_open.get(sample.decision_time + timedelta(minutes=offset))
        if candle is None:
            return None
        window.append(candle)
    exit_price = window[-1].close
    exit_reason = "HORIZON"
    exited_at = window[-1].closes_at
    for candle in window:
        target_hit = candle.high >= target if direction == "BUY" else candle.low <= target
        stop_hit = candle.low <= stop if direction == "BUY" else candle.high >= stop
        if target_hit and stop_hit:
            exit_price, exit_reason, exited_at = stop, "STOP", candle.closes_at
            break
        if stop_hit:
            exit_price, exit_reason, exited_at = stop, "STOP", candle.closes_at
            break
        if target_hit:
            exit_price, exit_reason, exited_at = target, "TARGET", candle.closes_at
            break
    slippage = Decimal("0.5")
    effective_entry = entry.open + sign * slippage
    effective_exit = exit_price - sign * slippage
    net_points = sign * (effective_exit - effective_entry)
    return {
        "sample_id": sample.sample_id,
        "decision_time": sample.decision_time.isoformat(),
        "entered_at": entry.opens_at.isoformat(),
        "exited_at": exited_at.isoformat(),
        "direction": direction,
        "probability": probability,
        "entry": format(entry.open, "f"),
        "stop": format(stop, "f"),
        "target": format(target, "f"),
        "exit": format(exit_price, "f"),
        "exit_reason": exit_reason,
        "net_points": float(net_points),
        "r_multiple": float(net_points / (Decimal("0.75") * atr)),
    }


def _direction(probabilities: dict[str, float]) -> str | None:
    up, down, neither = (
        probabilities["UP"],
        probabilities["DOWN"],
        probabilities["NEITHER"],
    )
    if neither > 0.45:
        return None
    if up >= 0.60 and up - down >= 0.15:
        return "BUY"
    if down >= 0.60 and down - up >= 0.15:
        return "SELL"
    return None


def _fold_probabilities(
    samples: tuple[TrainingSample, ...],
    train_indices: tuple[int, ...],
    test_indices: tuple[int, ...],
) -> np.ndarray:
    model = _model()
    model.fit(
        np.asarray([samples[index].feature_values for index in train_indices], dtype=float),
        np.asarray([samples[index].outcome.value for index in train_indices]),
    )
    raw = model.predict_proba(
        np.asarray([samples[index].feature_values for index in test_indices], dtype=float)
    )
    classes = tuple(str(item) for item in model.classes_)
    return raw[:, [classes.index(name) for name in CLASS_ORDER]]


def _model() -> Pipeline:
    return Pipeline(
        steps=(
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight=None,
                    max_iter=2_000,
                    random_state=RANDOM_SEED,
                    solver="lbfgs",
                ),
            ),
        )
    )


def _prior_vector(actual: tuple[str, ...]) -> np.ndarray:
    return prior_probabilities(actual, 1)[0]


def _negative_log_likelihood(probabilities: np.ndarray, indices: np.ndarray) -> float:
    selected = probabilities[np.arange(len(indices)), indices]
    return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def _calibration_blockers(
    *,
    raw: MetricSummary,
    calibrated: MetricSummary,
    prior: MetricSummary,
    brier_skill: float,
) -> list[str]:
    blockers = []
    if brier_skill <= 0:
        blockers.append("NO_POSITIVE_BRIER_SKILL")
    if calibrated.log_loss >= prior.log_loss:
        blockers.append("LOG_LOSS_DOES_NOT_BEAT_PRIOR")
    if calibrated.raw_ece_10_bin > 0.05:
        blockers.append("ECE_GATE_FAILED")
    if calibrated.balanced_accuracy <= 1.0 / 3.0:
        blockers.append("BALANCED_ACCURACY_GATE_FAILED")
    if (
        calibrated.multiclass_brier > raw.multiclass_brier
        or calibrated.log_loss > raw.log_loss
    ):
        blockers.append("CALIBRATION_DEGRADES_RAW_PROPER_SCORES")
    blockers.append("TARGET_SELECTED_USING_THIS_HISTORICAL_PERIOD")
    blockers.append("FORWARD_CONFIRMATION_NOT_COMPLETED")
    return blockers
