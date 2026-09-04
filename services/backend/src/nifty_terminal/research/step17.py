"""Chronological signal-policy research for the locked Step 16 shadow model."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import math
from statistics import stdev

import numpy as np

from nifty_terminal.domain.candle import Candle
from nifty_terminal.ml.metrics import prior_probabilities
from nifty_terminal.ml.models import DatasetBuildReport, TrainingSample
from nifty_terminal.ml.split import PurgedWalkForwardSplitter
from nifty_terminal.research.step16 import (
    CALIBRATION_FIT_FOLD,
    CALIBRATION_SELECTION_FOLD,
    HISTORICAL_BACKTEST_FOLD,
    LOCKED_ATR_MULTIPLIER,
    LOCKED_CANDIDATE,
    _fold_probabilities,
    _simulate_trade,
    apply_calibrator,
    fit_calibrator,
)
from nifty_terminal.research.v2 import NESTED_WALK_FORWARD_CONFIG


STEP17_VERSION = "shadow_policy_research.v1"
MINIMUM_SELECTION_TRADES = 30
MINIMUM_SELECTION_TRADES_PER_DIRECTION = 5
MINIMUM_EVALUATION_TRADES = 20
MINIMUM_EVALUATION_TRADES_PER_DIRECTION = 3


@dataclass(frozen=True, slots=True)
class ShadowPolicyThresholds:
    score_source: str
    activation_score: float
    minimum_class_margin: float
    maximum_neither_score: float
    minimum_prior_lift: float

    def to_contract(self) -> dict[str, object]:
        return {
            "score_source": self.score_source,
            "activation_score": self.activation_score,
            "minimum_class_margin": self.minimum_class_margin,
            "maximum_neither_score": self.maximum_neither_score,
            "minimum_prior_lift": self.minimum_prior_lift,
        }


def run_policy_research(
    *,
    dataset: DatasetBuildReport,
    minute_candles: tuple[Candle, ...],
    calibration_method: str,
) -> dict[str, object]:
    samples = dataset.samples
    folds = PurgedWalkForwardSplitter().split(samples, NESTED_WALK_FORWARD_CONFIG)
    needed = (CALIBRATION_FIT_FOLD, CALIBRATION_SELECTION_FOLD, HISTORICAL_BACKTEST_FOLD)
    raw = {
        index: _fold_probabilities(
            samples,
            folds[index].train_indices,
            folds[index].test_indices,
        )
        for index in needed
    }
    actual = {
        index: tuple(samples[item].outcome.value for item in folds[index].test_indices)
        for index in needed
    }
    calibration_prior = prior_probabilities(
        tuple(samples[item].outcome.value for item in folds[CALIBRATION_FIT_FOLD].train_indices),
        1,
    )[0]
    calibrator = fit_calibrator(
        method=calibration_method,
        probabilities=raw[CALIBRATION_FIT_FOLD],
        actual=actual[CALIBRATION_FIT_FOLD],
        prior=calibration_prior,
    )
    calibrated = {
        index: apply_calibrator(calibrator, raw[index])
        for index in (CALIBRATION_SELECTION_FOLD, HISTORICAL_BACKTEST_FOLD)
    }

    selection_fold = folds[CALIBRATION_SELECTION_FOLD]
    selection_samples = tuple(samples[index] for index in selection_fold.test_indices)
    selection_prior = prior_probabilities(
        tuple(samples[index].outcome.value for index in selection_fold.train_indices),
        1,
    )[0]
    candidates = []
    for thresholds in _candidate_grid():
        source = (
            raw[CALIBRATION_SELECTION_FOLD]
            if thresholds.score_source == "RAW_MODEL_SCORE"
            else calibrated[CALIBRATION_SELECTION_FOLD]
        )
        result = simulate_policy(
            samples=selection_samples,
            scores=source,
            prior=selection_prior,
            thresholds=thresholds,
            dataset=dataset,
            minute_candles=minute_candles,
            include_trades=False,
        )
        blockers = _selection_blockers(result)
        candidates.append(
            {
                "thresholds": thresholds,
                "metrics": result,
                "blockers": blockers,
                "passed": not blockers,
            }
        )
    passing = [item for item in candidates if item["passed"]]
    ranked_pool = passing or candidates
    leader = max(ranked_pool, key=_candidate_rank)
    selected_thresholds: ShadowPolicyThresholds = leader["thresholds"]

    final_fold = folds[HISTORICAL_BACKTEST_FOLD]
    final_samples = tuple(samples[index] for index in final_fold.test_indices)
    final_prior = prior_probabilities(
        tuple(samples[index].outcome.value for index in final_fold.train_indices),
        1,
    )[0]
    final_source = (
        raw[HISTORICAL_BACKTEST_FOLD]
        if selected_thresholds.score_source == "RAW_MODEL_SCORE"
        else calibrated[HISTORICAL_BACKTEST_FOLD]
    )
    evaluation = simulate_policy(
        samples=final_samples,
        scores=final_source,
        prior=final_prior,
        thresholds=selected_thresholds,
        dataset=dataset,
        minute_candles=minute_candles,
        include_trades=True,
    )
    evaluation_blockers = _evaluation_blockers(evaluation)
    historical_policy_gate_passed = not leader["blockers"] and not evaluation_blockers

    deployment_prior = prior_probabilities(
        tuple(item.outcome.value for item in samples),
        1,
    )[0]
    policy_artifact = build_policy_artifact(
        dataset_id=dataset.dataset_id,
        thresholds=selected_thresholds,
        deployment_prior=deployment_prior,
        historical_policy_gate_passed=historical_policy_gate_passed,
        blockers=(
            tuple(leader["blockers"])
            + tuple(evaluation_blockers)
            + (
                "TARGET_AND_POLICY_SELECTED_ON_HISTORICAL_DATA",
                "FORWARD_CONFIRMATION_NOT_COMPLETED",
            )
        ),
    )
    compact_candidates = sorted(candidates, key=_candidate_rank, reverse=True)[:20]
    return {
        "schema_version": 1,
        "step17_version": STEP17_VERSION,
        "dataset_id": dataset.dataset_id,
        "locked_model": {
            "candidate": LOCKED_CANDIDATE,
            "atr_multiplier": format(LOCKED_ATR_MULTIPLIER, "f"),
            "horizon_minutes": 60,
            "calibration_method": calibration_method,
        },
        "chronology": {
            "calibration_fit_fold": CALIBRATION_FIT_FOLD,
            "policy_selection_fold": CALIBRATION_SELECTION_FOLD,
            "policy_selection_starts_at": selection_fold.test_starts_at.isoformat(),
            "policy_selection_ends_at": selection_fold.test_ends_at.isoformat(),
            "historical_evaluation_fold": HISTORICAL_BACKTEST_FOLD,
            "historical_evaluation_starts_at": final_fold.test_starts_at.isoformat(),
            "historical_evaluation_ends_at": final_fold.test_ends_at.isoformat(),
        },
        "candidate_count": len(candidates),
        "passing_selection_candidate_count": len(passing),
        "top_selection_candidates": [
            {
                "thresholds": item["thresholds"].to_contract(),
                "metrics": item["metrics"],
                "blockers": item["blockers"],
                "passed": item["passed"],
            }
            for item in compact_candidates
        ],
        "selected_policy": selected_thresholds.to_contract(),
        "selection_metrics": leader["metrics"],
        "selection_blockers": leader["blockers"],
        "historical_evaluation": evaluation,
        "historical_evaluation_blockers": evaluation_blockers,
        "historical_policy_gate_passed": historical_policy_gate_passed,
        "policy_artifact": policy_artifact,
        "forward_confirmation_required": True,
        "approved_for_live_inference": False,
        "precise_probability_display_allowed": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
        "news_used": False,
        "nifty_spot_volume_used": False,
    }


def simulate_policy(
    *,
    samples: tuple[TrainingSample, ...],
    scores: np.ndarray,
    prior: np.ndarray,
    thresholds: ShadowPolicyThresholds,
    dataset: DatasetBuildReport,
    minute_candles: tuple[Candle, ...],
    include_trades: bool,
) -> dict[str, object]:
    minute_by_open = {item.opens_at: item for item in minute_candles}
    labels = {item.label_id: item for item in dataset.labels}
    trades = []
    waits: dict[str, int] = {}
    active_until: datetime | None = None
    for row_index, sample in enumerate(samples):
        if active_until is not None and sample.decision_time < active_until:
            waits["ACTIVE_POSITION"] = waits.get("ACTIVE_POSITION", 0) + 1
            continue
        direction = policy_direction(scores[row_index], prior, thresholds)
        if direction is None:
            waits["POLICY_WAIT"] = waits.get("POLICY_WAIT", 0) + 1
            continue
        entry = minute_by_open.get(sample.decision_time)
        label = labels.get(sample.label_id)
        if entry is None or label is None or label.atr_at_decision is None:
            waits["REPLAY_INPUT_UNAVAILABLE"] = waits.get("REPLAY_INPUT_UNAVAILABLE", 0) + 1
            continue
        trade = _simulate_trade(
            sample=sample,
            direction=direction,
            probability=float(max(scores[row_index, 0], scores[row_index, 2])),
            entry=entry,
            atr=label.atr_at_decision,
            minute_by_open=minute_by_open,
        )
        if trade is None:
            waits["INCOMPLETE_MINUTE_WINDOW"] = waits.get("INCOMPLETE_MINUTE_WINDOW", 0) + 1
            continue
        trades.append(trade)
        active_until = datetime.fromisoformat(str(trade["exited_at"]))
    return _trade_metrics(
        trades=trades,
        evaluation_decisions=len(samples),
        waits=waits,
        include_trades=include_trades,
    )


def policy_direction(
    scores: np.ndarray,
    prior: np.ndarray,
    thresholds: ShadowPolicyThresholds,
) -> str | None:
    down, neither, up = (float(item) for item in scores)
    if neither > thresholds.maximum_neither_score:
        return None
    if up >= down:
        direction, directional, opposing, prior_directional = "BUY", up, down, float(prior[2])
    else:
        direction, directional, opposing, prior_directional = "SELL", down, up, float(prior[0])
    if directional < thresholds.activation_score:
        return None
    if directional - max(opposing, neither) < thresholds.minimum_class_margin:
        return None
    if directional - prior_directional < thresholds.minimum_prior_lift:
        return None
    return direction


def build_policy_artifact(
    *,
    dataset_id: str,
    thresholds: ShadowPolicyThresholds,
    deployment_prior: np.ndarray,
    historical_policy_gate_passed: bool,
    blockers: tuple[str, ...],
) -> dict[str, object]:
    artifact = {
        "schema_version": 1,
        "artifact_version": "nifty_shadow_policy.v1",
        "dataset_id": dataset_id,
        "thresholds": thresholds.to_contract(),
        "deployment_prior": {
            "DOWN": float(deployment_prior[0]),
            "NEITHER": float(deployment_prior[1]),
            "UP": float(deployment_prior[2]),
        },
        "historical_policy_gate_passed": historical_policy_gate_passed,
        "shadow_candidate_directions_enabled": historical_policy_gate_passed,
        "runtime_mode": (
            "SHADOW_CANDIDATES" if historical_policy_gate_passed else "WAIT_ONLY"
        ),
        "blockers": list(dict.fromkeys(blockers)),
        "shadow_only": True,
        "approved_for_live_inference": False,
        "precise_probability_display_allowed": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
    }
    checksum = hashlib.sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**artifact, "sha256": checksum}


def _candidate_grid() -> tuple[ShadowPolicyThresholds, ...]:
    return tuple(
        ShadowPolicyThresholds(source, activation, margin, maximum_neither, prior_lift)
        for source in ("RAW_MODEL_SCORE", "CALIBRATED_PROBABILITY")
        for activation in (0.38, 0.40, 0.425, 0.45, 0.475, 0.50)
        for margin in (0.02, 0.04, 0.06, 0.08)
        for maximum_neither in (0.35, 0.40, 0.45)
        for prior_lift in (0.00, 0.02, 0.04)
    )


def _trade_metrics(
    *,
    trades: list[dict[str, object]],
    evaluation_decisions: int,
    waits: dict[str, int],
    include_trades: bool,
) -> dict[str, object]:
    points = [float(item["net_points"]) for item in trades]
    multiples = [float(item["r_multiple"]) for item in trades]
    positive = [item for item in points if item > 0]
    negative = [item for item in points if item < 0]
    average_r = sum(multiples) / len(multiples) if multiples else None
    standard_error = (
        stdev(multiples) / math.sqrt(len(multiples)) if len(multiples) > 1 else None
    )
    lower_confidence = (
        average_r - 1.96 * standard_error
        if average_r is not None and standard_error is not None
        else None
    )
    cumulative_r = 0.0
    peak_r = 0.0
    maximum_drawdown_r = 0.0
    for value in multiples:
        cumulative_r += value
        peak_r = max(peak_r, cumulative_r)
        maximum_drawdown_r = max(maximum_drawdown_r, peak_r - cumulative_r)
    result = {
        "evaluation_decisions": evaluation_decisions,
        "trade_count": len(trades),
        "coverage": len(trades) / evaluation_decisions if evaluation_decisions else 0.0,
        "buy_count": sum(item["direction"] == "BUY" for item in trades),
        "sell_count": sum(item["direction"] == "SELL" for item in trades),
        "wait_counts": waits,
        "target_hit_count": sum(item["exit_reason"] == "TARGET" for item in trades),
        "stop_hit_count": sum(item["exit_reason"] == "STOP" for item in trades),
        "expired_count": sum(item["exit_reason"] == "HORIZON" for item in trades),
        "win_rate": len(positive) / len(points) if points else None,
        "net_points": sum(points),
        "average_points": sum(points) / len(points) if points else None,
        "average_r_multiple": average_r,
        "lower_95_average_r_multiple": lower_confidence,
        "profit_factor": (
            sum(positive) / abs(sum(negative)) if positive and negative else None
        ),
        "maximum_drawdown_r": maximum_drawdown_r,
        "hypothetical_index_points_only": True,
        "rupee_pnl_available": False,
    }
    if include_trades:
        result["trades"] = trades
    return result


def _selection_blockers(metrics: dict[str, object]) -> list[str]:
    blockers = []
    if int(metrics["trade_count"]) < MINIMUM_SELECTION_TRADES:
        blockers.append("SELECTION_TRADE_COUNT_TOO_LOW")
    if int(metrics["buy_count"]) < MINIMUM_SELECTION_TRADES_PER_DIRECTION:
        blockers.append("SELECTION_BUY_SUPPORT_TOO_LOW")
    if int(metrics["sell_count"]) < MINIMUM_SELECTION_TRADES_PER_DIRECTION:
        blockers.append("SELECTION_SELL_SUPPORT_TOO_LOW")
    if metrics["average_r_multiple"] is None or float(metrics["average_r_multiple"]) <= 0:
        blockers.append("SELECTION_AVERAGE_R_NOT_POSITIVE")
    if metrics["profit_factor"] is None or float(metrics["profit_factor"]) <= 1.0:
        blockers.append("SELECTION_PROFIT_FACTOR_NOT_ABOVE_ONE")
    if (
        metrics["lower_95_average_r_multiple"] is None
        or float(metrics["lower_95_average_r_multiple"]) <= -0.10
    ):
        blockers.append("SELECTION_DOWNSIDE_CONFIDENCE_GATE_FAILED")
    return blockers


def _evaluation_blockers(metrics: dict[str, object]) -> list[str]:
    blockers = []
    if int(metrics["trade_count"]) < MINIMUM_EVALUATION_TRADES:
        blockers.append("EVALUATION_TRADE_COUNT_TOO_LOW")
    if int(metrics["buy_count"]) < MINIMUM_EVALUATION_TRADES_PER_DIRECTION:
        blockers.append("EVALUATION_BUY_SUPPORT_TOO_LOW")
    if int(metrics["sell_count"]) < MINIMUM_EVALUATION_TRADES_PER_DIRECTION:
        blockers.append("EVALUATION_SELL_SUPPORT_TOO_LOW")
    if metrics["average_r_multiple"] is None or float(metrics["average_r_multiple"]) <= 0:
        blockers.append("EVALUATION_AVERAGE_R_NOT_POSITIVE")
    if metrics["profit_factor"] is None or float(metrics["profit_factor"]) <= 1.0:
        blockers.append("EVALUATION_PROFIT_FACTOR_NOT_ABOVE_ONE")
    return blockers


def _candidate_rank(item: dict[str, object]) -> tuple[float, float, int, float]:
    metrics: dict[str, object] = item["metrics"]
    return (
        float(metrics["lower_95_average_r_multiple"] or -999.0),
        float(metrics["profit_factor"] or 0.0),
        int(metrics["trade_count"]),
        float(metrics["average_r_multiple"] or -999.0),
    )
