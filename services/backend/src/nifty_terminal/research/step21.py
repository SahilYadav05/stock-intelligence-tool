"""Step 21 event-based price-action research.

Step 20 showed weak probability ranking but excessive five-minute-row activation.
This stage predeclares four causal market-structure events and asks a compact model
to rank only those events.  It does not tune setup definitions on the diagnostic
folds and never creates a live artifact.
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
from nifty_terminal.research.step18f import daily_uplift_bootstrap, session_bootstrap_values
from nifty_terminal.research.step20 import _benchmarks_for_folds


STEP21_VERSION = "event_price_action_research.v1"
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
SETUP_ORDER = (
    "LIQUIDITY_SWEEP_REVERSAL",
    "CONFIRMED_STRUCTURE_BREAK",
    "COMPRESSION_EXPANSION",
    "TREND_EMA_RECLAIM",
)
MODEL_CANDIDATES = ("event_logistic_compact", "event_hgb_compact")
ACTIVATION_PERCENTILES = (0.35, 0.55, 0.75)
MAX_TRADES_PER_SESSION = 5
MODEL_GATE = {
    "minimum_pooled_roc_auc": 0.515,
    "minimum_pooled_brier_skill": 0.0,
    "minimum_probability_std": 0.01,
    "minimum_fold_roc_auc": 0.48,
    "minimum_folds_above_random_auc": 2,
    "minimum_folds_positive_brier_skill": 2,
    "minimum_test_events_per_fold": 50,
}
SELECTION_POLICY_GATE = {
    "minimum_trades": 60,
    "minimum_trades_per_direction": 15,
    "minimum_sessions": 20,
    "minimum_trades_per_session": 1.5,
    "minimum_win_rate": 0.50,
    "minimum_profit_factor": 1.10,
    "minimum_average_r_lower_95": 0.0,
    "maximum_drawdown_r": 10.0,
    "maximum_session_trade_share": 0.10,
}
DIAGNOSTIC_POLICY_GATE = {
    "minimum_trades": 120,
    "minimum_trades_per_direction": 30,
    "minimum_sessions": 45,
    "minimum_trades_per_session": 1.5,
    "minimum_win_rate": 0.50,
    "minimum_profit_factor": 1.05,
    "minimum_average_r_lower_95": 0.0,
    "maximum_drawdown_r": 15.0,
    "maximum_session_trade_share": 0.06,
    "minimum_daily_r_uplift_lower_95": 0.0,
}

EVENT_DEFINITION = {
    "setups": {
        "LIQUIDITY_SWEEP_REVERSAL": "confirmed swing sweep plus directional close",
        "CONFIRMED_STRUCTURE_BREAK": "first close beyond confirmed swing with directional body",
        "COMPRESSION_EXPANSION": "three-bar compression below 0.75 followed by expansion above 1.05",
        "TREND_EMA_RECLAIM": "EMA20 distance crosses zero in confirmed structure direction",
    },
    "priority": SETUP_ORDER,
    "conflicting_direction_policy": "WAIT",
    "maximum_trades_per_session": MAX_TRADES_PER_SESSION,
    "feature_time": "finalized decision candle only; previous row must be same session",
}
RESEARCH_IDENTITY = hashlib.sha256(
    json.dumps(
        {
            "version": STEP21_VERSION,
            "walk_forward": WALK_FORWARD_CONFIG.to_contract(),
            "event_definition": EVENT_DEFINITION,
            "models": MODEL_CANDIDATES,
            "activation_percentiles": ACTIVATION_PERCENTILES,
            "model_gate": MODEL_GATE,
            "selection_policy_gate": SELECTION_POLICY_GATE,
            "diagnostic_policy_gate": DIAGNOSTIC_POLICY_GATE,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


@dataclass(frozen=True, slots=True)
class EventCandidate:
    sample_index: int
    direction: str
    setup: str


@dataclass(frozen=True, slots=True)
class EventPolicy:
    activation_percentile: float
    maximum_trades_per_session: int = MAX_TRADES_PER_SESSION


def build_event_candidates(samples, matrix: np.ndarray, names: tuple[str, ...]):
    required = (
        "price_action__break_of_structure_up",
        "price_action__break_of_structure_down",
        "price_action__swing_low_liquidity_sweep",
        "price_action__swing_high_liquidity_sweep",
        "price_action__range_compression_3_to_12",
        "price_action__close_location",
        "price_action__body_efficiency",
        "price_action__structure_score",
        "primary_5m__distance_ema20_atr",
    )
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError("Step 21 event features missing: " + ", ".join(missing))
    columns = {name: names.index(name) for name in required}
    events: list[EventCandidate] = []
    diagnostics = Counter[str]()
    previous = None
    previous_session = None
    for index, sample in enumerate(samples):
        row = matrix[index]
        session = _session_key(sample.decision_time)
        same_session = previous is not None and previous_session == session
        candidates: list[tuple[str, str]] = []
        close_location = float(row[columns["price_action__close_location"]])
        body = float(row[columns["price_action__body_efficiency"]])
        structure = float(row[columns["price_action__structure_score"]])
        compression = float(row[columns["price_action__range_compression_3_to_12"]])
        ema_distance = float(row[columns["primary_5m__distance_ema20_atr"]])
        if float(row[columns["price_action__swing_low_liquidity_sweep"]]) >= 0.5 and close_location >= 0.60 and body >= 0.10:
            candidates.append(("LONG", "LIQUIDITY_SWEEP_REVERSAL"))
        if float(row[columns["price_action__swing_high_liquidity_sweep"]]) >= 0.5 and close_location <= 0.40 and body <= -0.10:
            candidates.append(("SHORT", "LIQUIDITY_SWEEP_REVERSAL"))
        if same_session:
            previous_bos_up = float(previous[columns["price_action__break_of_structure_up"]])
            previous_bos_down = float(previous[columns["price_action__break_of_structure_down"]])
            if float(row[columns["price_action__break_of_structure_up"]]) >= 0.5 and previous_bos_up < 0.5 and close_location >= 0.60 and body >= 0.15:
                candidates.append(("LONG", "CONFIRMED_STRUCTURE_BREAK"))
            if float(row[columns["price_action__break_of_structure_down"]]) >= 0.5 and previous_bos_down < 0.5 and close_location <= 0.40 and body <= -0.15:
                candidates.append(("SHORT", "CONFIRMED_STRUCTURE_BREAK"))
            previous_compression = float(previous[columns["price_action__range_compression_3_to_12"]])
            if previous_compression <= 0.75 and compression >= 1.05:
                if close_location >= 0.70 and body >= 0.40:
                    candidates.append(("LONG", "COMPRESSION_EXPANSION"))
                if close_location <= 0.30 and body <= -0.40:
                    candidates.append(("SHORT", "COMPRESSION_EXPANSION"))
            previous_ema = float(previous[columns["primary_5m__distance_ema20_atr"]])
            if previous_ema <= -0.15 and ema_distance >= 0 and structure >= 0.5 and close_location >= 0.60 and body >= 0.15:
                candidates.append(("LONG", "TREND_EMA_RECLAIM"))
            if previous_ema >= 0.15 and ema_distance <= 0 and structure <= -0.5 and close_location <= 0.40 and body <= -0.15:
                candidates.append(("SHORT", "TREND_EMA_RECLAIM"))
        directions = {item[0] for item in candidates}
        if len(directions) > 1:
            diagnostics["CONFLICTING_DIRECTION_WAIT"] += 1
        elif candidates:
            direction = candidates[0][0]
            setup = next(name for name in SETUP_ORDER if (direction, name) in candidates)
            events.append(EventCandidate(index, direction, setup))
            diagnostics[f"{direction}:{setup}"] += 1
        else:
            diagnostics["NO_EVENT"] += 1
        previous = row
        previous_session = session
    return tuple(events), dict(sorted(diagnostics.items()))


def run_event_price_action_research(*, context: ContextFeatureBuild, minute_candles, context_bundle_sha256: str):
    samples, matrix, long_paths, short_paths, exclusions = build_trade_paths(
        dataset=context.dataset, features=context.matrix, minute_candles=minute_candles
    )
    if len(samples) < 24_000:
        raise ValueError("Step 21 requires at least 24,000 complete trade paths")
    matrix = np.asarray(matrix, dtype=float)
    names = context.matrix.feature_names
    events, event_diagnostics = build_event_candidates(samples, matrix, names)
    if len(events) < 1_000:
        raise ValueError("Step 21 requires at least 1,000 causal price-action events")
    folds = PurgedWalkForwardSplitter().split(samples, WALK_FORWARD_CONFIG)
    labels = {
        "LONG": np.asarray([long_paths[item.sample_id].success for item in samples], dtype=int),
        "SHORT": np.asarray([short_paths[item.sample_id].success for item in samples], dtype=int),
    }
    comparisons = []
    predictions: dict[str, dict[int, dict[str, object]]] = {}
    for candidate in MODEL_CANDIDATES:
        by_fold = {}
        fold_reports = []
        for fold_index in MODEL_SELECTION_FOLDS:
            output = _fit_predict(candidate, events, folds[fold_index], samples, matrix, names, labels)
            by_fold[fold_index] = output
            metrics = binary_metrics(output["actual"], output["probability"])
            prior_metrics = binary_metrics(output["actual"], np.full(len(output["actual"]), output["train_prior"]))
            fold_reports.append({
                "fold_index": fold_index,
                "event_count": len(output["events"]),
                "metrics": metrics,
                "prior_metrics": prior_metrics,
                "brier_skill_vs_train_prior": _brier_skill(metrics, prior_metrics),
            })
        actual = np.concatenate([by_fold[index]["actual"] for index in MODEL_SELECTION_FOLDS])
        probability = np.concatenate([by_fold[index]["probability"] for index in MODEL_SELECTION_FOLDS])
        prior = np.concatenate([np.full(len(by_fold[index]["actual"]), by_fold[index]["train_prior"]) for index in MODEL_SELECTION_FOLDS])
        metrics = binary_metrics(actual, probability)
        prior_metrics = binary_metrics(actual, prior)
        skill = _brier_skill(metrics, prior_metrics)
        blockers = _model_blockers(metrics, skill, fold_reports)
        comparisons.append({
            "candidate": candidate,
            "feature_count": _candidate_feature_count(candidate, names),
            "selection_metrics": metrics,
            "selection_brier_skill_vs_prior": skill,
            "folds": fold_reports,
            "selection_viable": not blockers,
            "selection_blockers": blockers,
        })
        predictions[candidate] = by_fold
    viable = [item for item in comparisons if item["selection_viable"]]
    selected = min(viable or comparisons, key=_model_rank)
    selected_name = selected["candidate"]
    for fold_index in (CALIBRATION_FOLD, POLICY_SELECTION_FOLD, *HISTORICAL_DIAGNOSTIC_FOLDS):
        predictions[selected_name][fold_index] = _fit_predict(
            selected_name, events, folds[fold_index], samples, matrix, names, labels
        )
    calibration_output = predictions[selected_name][CALIBRATION_FOLD]
    calibrations = []
    for method in ("identity", "platt"):
        artifact = fit_binary_calibrator(
            method=method,
            probabilities=calibration_output["probability"],
            actual=calibration_output["actual"],
            prior=float(calibration_output["train_prior"]),
        )
        blockers = _calibration_blockers(artifact)
        calibrated = apply_binary_calibrator(artifact, calibration_output["probability"])
        calibrations.append({"method": method, "artifact": artifact, "metrics": binary_metrics(calibration_output["actual"], calibrated), "blockers": blockers})
    calibration_report = min(
        [item for item in calibrations if not item["blockers"]],
        key=lambda item: (item["metrics"]["brier"], item["metrics"]["log_loss"], item["method"]),
    )
    calibration = calibration_report["artifact"]
    calibrated = {
        fold_index: apply_binary_calibrator(calibration, output["probability"])
        for fold_index, output in predictions[selected_name].items()
    }
    reference_scores = calibrated[CALIBRATION_FOLD]
    selection_output = predictions[selected_name][POLICY_SELECTION_FOLD]
    selection_percentiles = _reference_percentiles(reference_scores, calibrated[POLICY_SELECTION_FOLD])
    selection_benchmarks = _benchmarks_for_folds(
        fold_indices=(POLICY_SELECTION_FOLD,), folds=folds, samples=samples, matrix=matrix,
        names=names, labels=labels, long_paths=long_paths, short_paths=short_paths,
    )
    policies = []
    for threshold in ACTIVATION_PERCENTILES:
        policy = EventPolicy(threshold)
        metrics = _simulate(
            events=selection_output["events"], score_percentiles=selection_percentiles,
            samples=samples, long_paths=long_paths, short_paths=short_paths,
            policy=policy, benchmarks=selection_benchmarks,
            evaluation_decisions=len(folds[POLICY_SELECTION_FOLD].test_indices),
        )
        blockers = _policy_blockers(metrics, SELECTION_POLICY_GATE, require_benchmarks=False)
        if not selected["selection_viable"]:
            blockers = sorted(set(blockers + ["MODEL_SELECTION_GATE_FAILED"]))
        policies.append({"policy": _policy_contract(policy), "metrics": metrics, "selection_viable": not blockers, "selection_blockers": blockers})
    viable_policies = [item for item in policies if item["selection_viable"]]
    exploratory = min(policies, key=_policy_rank)
    selected_policy = min(viable_policies, key=_policy_rank) if viable_policies else None
    evaluated_policy_report = selected_policy or exploratory
    evaluated_policy = EventPolicy(**evaluated_policy_report["policy"])
    diagnostic_events = tuple(
        event
        for fold_index in HISTORICAL_DIAGNOSTIC_FOLDS
        for event in predictions[selected_name][fold_index]["events"]
    )
    diagnostic_probability = np.concatenate([calibrated[index] for index in HISTORICAL_DIAGNOSTIC_FOLDS])
    diagnostic_actual = np.concatenate([predictions[selected_name][index]["actual"] for index in HISTORICAL_DIAGNOSTIC_FOLDS])
    diagnostic_prior = np.concatenate([
        np.full(len(predictions[selected_name][index]["actual"]), predictions[selected_name][index]["train_prior"])
        for index in HISTORICAL_DIAGNOSTIC_FOLDS
    ])
    diagnostic_reference = np.concatenate((reference_scores, calibrated[POLICY_SELECTION_FOLD]))
    diagnostic_percentiles = _reference_percentiles(diagnostic_reference, diagnostic_probability)
    diagnostic_benchmarks = _benchmarks_for_folds(
        fold_indices=HISTORICAL_DIAGNOSTIC_FOLDS, folds=folds, samples=samples, matrix=matrix,
        names=names, labels=labels, long_paths=long_paths, short_paths=short_paths,
    )
    diagnostic = _simulate(
        events=diagnostic_events, score_percentiles=diagnostic_percentiles,
        samples=samples, long_paths=long_paths, short_paths=short_paths,
        policy=evaluated_policy, benchmarks=diagnostic_benchmarks,
        evaluation_decisions=sum(len(folds[index].test_indices) for index in HISTORICAL_DIAGNOSTIC_FOLDS),
    )
    diagnostic_blockers = _policy_blockers(diagnostic, DIAGNOSTIC_POLICY_GATE, require_benchmarks=True)
    if selected_policy is None:
        diagnostic_blockers = sorted(set(diagnostic_blockers + ["NO_POLICY_PASSED_SELECTION_GATE"]))
    diagnostic_model_metrics = binary_metrics(diagnostic_actual, diagnostic_probability)
    diagnostic_prior_metrics = binary_metrics(diagnostic_actual, diagnostic_prior)
    diagnostic_skill = _brier_skill(diagnostic_model_metrics, diagnostic_prior_metrics)
    diagnostic_model_blockers = []
    if diagnostic_model_metrics["roc_auc"] is None or diagnostic_model_metrics["roc_auc"] <= 0.51:
        diagnostic_model_blockers.append("DIAGNOSTIC_ROC_AUC_GATE_FAILED")
    if diagnostic_skill <= 0:
        diagnostic_model_blockers.append("DIAGNOSTIC_BRIER_SKILL_GATE_FAILED")
    historical_blockers = sorted(set(
        ([] if selected["selection_viable"] else ["MODEL_SELECTION_GATE_FAILED"])
        + ([] if selected_policy else ["NO_POLICY_PASSED_SELECTION_GATE"])
        + diagnostic_model_blockers + diagnostic_blockers
    ))
    return {
        "schema_version": 1,
        "step21_version": STEP21_VERSION,
        "research_identity": RESEARCH_IDENTITY,
        "dataset_id": context.dataset.dataset_id,
        "context_bundle_sha256": context_bundle_sha256,
        "objective": {
            "model_output": "probability that a predeclared directional price-action event reaches 1R before a 0.75R stop",
            "event_definition": EVENT_DEFINITION,
            "entry": "next finalized 1m open",
            "horizon_minutes": 60,
            "same_minute_resolution": "STOP_FIRST_CONSERVATIVE",
            "slippage_points_one_way": 0.5,
        },
        "dataset": {"complete_trade_paths": len(samples), "event_count": len(events), "event_diagnostics": event_diagnostics, "excluded_trade_paths": exclusions},
        "chronology": {
            "walk_forward": WALK_FORWARD_CONFIG.to_contract(),
            "model_selection_folds": list(MODEL_SELECTION_FOLDS),
            "calibration_fold": CALIBRATION_FOLD,
            "policy_selection_fold": POLICY_SELECTION_FOLD,
            "historical_diagnostic_folds": list(HISTORICAL_DIAGNOSTIC_FOLDS),
            "diagnostic_thresholds_locked_before_diagnostic": True,
            "future_labels_used_in_events": False,
            "historical_period_previously_seen": True,
        },
        "anti_overfit_controls": {
            "predeclared_event_setups": list(SETUP_ORDER),
            "model_candidates": len(MODEL_CANDIDATES),
            "policy_candidates": len(ACTIVATION_PERCENTILES),
            "purge_and_embargo_bars": 12,
            "non_overlapping_positions": True,
            "hard_daily_trade_cap": MAX_TRADES_PER_SESSION,
            "session_balanced_training_weights": True,
        },
        "model_comparison": comparisons,
        "selected_model": {"candidate": selected_name, "selection_viable": selected["selection_viable"], "selection_blockers": selected["selection_blockers"], "feature_count": selected["feature_count"]},
        "calibration": {"selected": calibration.to_contract(), "comparison": [{"method": item["method"], "metrics": item["metrics"], "selection_blockers": item["blockers"]} for item in calibrations]},
        "policy_selection": {"candidate_count": len(policies), "passing_candidate_count": len(viable_policies), "selected": selected_policy, "best_exploratory_rejected": None if selected_policy else exploratory, "gate": SELECTION_POLICY_GATE},
        "historical_diagnostic": {"evaluated_policy_source": "SELECTED" if selected_policy else "BEST_REJECTED_EXPLORATORY_ONLY", "policy": _policy_contract(evaluated_policy), "metrics": diagnostic, "gate": DIAGNOSTIC_POLICY_GATE, "gate_passed": not diagnostic_blockers and selected_policy is not None, "blockers": diagnostic_blockers},
        "diagnostic_model": {"metrics": diagnostic_model_metrics, "prior_metrics": diagnostic_prior_metrics, "brier_skill_vs_prior": diagnostic_skill, "blockers": diagnostic_model_blockers, "gate_passed": not diagnostic_model_blockers},
        "research_gate": {"passed_before_mandatory_forward_blockers": not historical_blockers, "historical_blockers": historical_blockers, "blockers": sorted(set(historical_blockers + ["HISTORICAL_PERIOD_USED_FOR_MODEL_DEVELOPMENT", "FORWARD_CONFIRMATION_NOT_COMPLETED"]))},
        "model_artifact_created": False,
        "approved_for_live_inference": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
    }


def _fit_predict(candidate, events, fold, samples, matrix, names, labels):
    train_set = set(fold.train_indices)
    test_set = set(fold.test_indices)
    train_events = tuple(item for item in events if item.sample_index in train_set)
    test_events = tuple(item for item in events if item.sample_index in test_set)
    if len(train_events) < 250 or len(test_events) < 25:
        raise ValueError("Insufficient causal events in a Step 21 fold")
    feature_indices = _feature_indices(candidate, names)
    train_x = _event_design(matrix, train_events, feature_indices)
    test_x = _event_design(matrix, test_events, feature_indices)
    train_y = np.asarray([labels[item.direction][item.sample_index] for item in train_events], dtype=int)
    test_y = np.asarray([labels[item.direction][item.sample_index] for item in test_events], dtype=int)
    weights = _event_session_weights(samples, train_events)
    model = _candidate(candidate)
    if isinstance(model, Pipeline):
        model.fit(train_x, train_y, model__sample_weight=weights)
    else:
        model.fit(train_x, train_y, sample_weight=weights)
    probability = model.predict_proba(test_x)[:, list(model.classes_).index(1)]
    return {"events": test_events, "actual": test_y, "probability": np.asarray(probability, dtype=float), "train_prior": float(np.average(train_y, weights=weights))}


def _candidate(name):
    if name == "event_logistic_compact":
        return Pipeline((("scale", StandardScaler()), ("model", LogisticRegression(C=0.03, max_iter=2_000, random_state=RANDOM_SEED, solver="lbfgs"))))
    if name == "event_hgb_compact":
        return HistGradientBoostingClassifier(learning_rate=0.025, max_iter=160, max_leaf_nodes=7, min_samples_leaf=60, l2_regularization=15.0, random_state=RANDOM_SEED)
    raise ValueError(f"Unknown Step 21 candidate: {name}")


def _feature_indices(candidate: str, names: tuple[str, ...]) -> np.ndarray:
    primary_suffixes = ("__return_1", "__return_5", "__range_pct", "__body_pct", "__upper_wick_pct", "__lower_wick_pct", "__atr_pct", "__rsi_14", "__bollinger_z_20", "__distance_ema20_atr", "__range_atr", "__minute_of_session", "__day_of_week")
    higher_suffixes = ("__atr_pct", "__rsi_14", "__bollinger_z_20", "__distance_ema20_atr", "__trend_ema20_above_ema50")
    market_suffixes = ("__return_1", "__return_3", "__realized_vol_12", "__ema_8_21_atr", "__rsi_14", "__return_z_48")
    indices = [
        index for index, name in enumerate(names)
        if name.startswith("price_action__")
        or name.startswith("research_v3__")
        or name.startswith("cross__")
        or (name.startswith("primary_5m__") and name.endswith(primary_suffixes))
        or (name.startswith(("context_15m__", "context_1h__")) and name.endswith(higher_suffixes))
        or (name.startswith("context_market__") and name.endswith(market_suffixes))
    ]
    if not indices:
        raise ValueError(f"Step 21 candidate {candidate} selected no features")
    return np.asarray(indices, dtype=int)


def _event_design(matrix, events, feature_indices):
    values = np.asarray([matrix[item.sample_index, feature_indices] for item in events], dtype=float)
    signs = np.asarray([1.0 if item.direction == "LONG" else -1.0 for item in events])
    setups = np.asarray([[1.0 if item.setup == setup else 0.0 for setup in SETUP_ORDER] for item in events])
    return np.column_stack((values, values * signs[:, None], signs, setups))


def _event_session_weights(samples, events):
    sessions = [_session_key(samples[item.sample_index].decision_time) for item in events]
    counts = Counter(sessions)
    values = np.asarray([1.0 / counts[item] for item in sessions], dtype=float)
    return values * len(values) / np.sum(values)


def _candidate_feature_count(candidate, names):
    return 2 * len(_feature_indices(candidate, names)) + 1 + len(SETUP_ORDER)


def _simulate(*, events, score_percentiles, samples, long_paths, short_paths, policy, benchmarks, evaluation_decisions):
    trades: list[TradePath] = []
    waits = Counter[str]()
    daily_counts = Counter[str]()
    active_until: datetime | None = None
    for event, percentile in zip(events, score_percentiles):
        sample = samples[event.sample_index]
        session = _session_key(sample.decision_time)
        if active_until is not None and sample.decision_time < active_until:
            waits["ACTIVE_POSITION"] += 1
            continue
        if percentile < policy.activation_percentile:
            waits["SCORE_PERCENTILE_BELOW_THRESHOLD"] += 1
            continue
        if daily_counts[session] >= policy.maximum_trades_per_session:
            waits["SESSION_TRADE_CAP"] += 1
            continue
        path = long_paths[sample.sample_id] if event.direction == "LONG" else short_paths[sample.sample_id]
        trades.append(path)
        daily_counts[session] += 1
        active_until = path.exited_at
    return _replay(tuple(trades), len(events), evaluation_decisions, waits, policy, benchmarks)


def _replay(trades, event_count, decisions, waits, policy, benchmarks):
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
    daily = defaultdict(float)
    for path in trades:
        daily[_session_key(path.decision_time)] += float(path.r_multiple)
    return {
        "policy": _policy_contract(policy), "evaluation_decisions": decisions,
        "eligible_event_count": event_count, "trade_count": len(trades),
        "buy_count": sum(item.direction == "LONG" for item in trades),
        "sell_count": sum(item.direction == "SHORT" for item in trades),
        "event_activation_rate": len(trades) / event_count if event_count else 0.0,
        "decision_coverage": len(trades) / decisions if decisions else 0.0,
        "session_count": len(counts), "trades_per_session": len(trades) / len(counts) if counts else 0.0,
        "maximum_session_trade_share": max(counts.values()) / len(trades) if trades else None,
        "target_hit_count": sum(item.exit_reason == "TARGET" for item in trades),
        "stop_hit_count": sum(item.exit_reason == "STOP" for item in trades),
        "expired_count": sum(item.exit_reason == "HORIZON" for item in trades),
        "win_rate": len(gains) / len(trades) if trades else None,
        "net_points": float(sum(points)), "average_points": float(np.mean(points)) if points else None,
        "average_r_multiple": float(np.mean(r_values)) if len(r_values) else None,
        "average_r_session_bootstrap_95": session_bootstrap_values(trades, r_values),
        "profit_factor": sum(gains) / abs(sum(losses)) if gains and losses else None,
        "maximum_drawdown_r": drawdown,
        "benchmark_daily_r_uplift_95": {name: daily_uplift_bootstrap(dict(daily), report["daily_total_r"]) for name, report in benchmarks.items()},
        "wait_counts": dict(waits), "hypothetical_index_points_only": True, "rupee_pnl_available": False,
    }


def _policy_blockers(metrics, gate, *, require_benchmarks):
    blockers = []
    if metrics["trade_count"] < gate["minimum_trades"]: blockers.append("TRADE_SUPPORT_TOO_LOW")
    if metrics["buy_count"] < gate["minimum_trades_per_direction"]: blockers.append("BUY_SUPPORT_TOO_LOW")
    if metrics["sell_count"] < gate["minimum_trades_per_direction"]: blockers.append("SELL_SUPPORT_TOO_LOW")
    if metrics["session_count"] < gate["minimum_sessions"]: blockers.append("SESSION_SUPPORT_TOO_LOW")
    if metrics["trades_per_session"] < gate["minimum_trades_per_session"]: blockers.append("TRADE_CADENCE_TOO_LOW")
    if metrics["win_rate"] is None or metrics["win_rate"] < gate["minimum_win_rate"]: blockers.append("WIN_RATE_GATE_FAILED")
    if metrics["profit_factor"] is None or metrics["profit_factor"] <= gate["minimum_profit_factor"]: blockers.append("PROFIT_FACTOR_GATE_FAILED")
    lower = metrics["average_r_session_bootstrap_95"]["lower"]
    if lower is None or lower <= gate["minimum_average_r_lower_95"]: blockers.append("EXPECTANCY_CONFIDENCE_GATE_FAILED")
    if metrics["maximum_drawdown_r"] > gate["maximum_drawdown_r"]: blockers.append("MAXIMUM_DRAWDOWN_GATE_FAILED")
    if metrics["maximum_session_trade_share"] is None or metrics["maximum_session_trade_share"] > gate["maximum_session_trade_share"]: blockers.append("SESSION_CONCENTRATION_TOO_HIGH")
    if require_benchmarks:
        for name in ("WAIT", "ALWAYS_LONG", "ALWAYS_SHORT", "TECHNICAL_TREND"):
            interval = metrics["benchmark_daily_r_uplift_95"].get(name)
            if interval is None or interval["lower"] <= gate["minimum_daily_r_uplift_lower_95"]:
                blockers.append(f"DAILY_R_UPLIFT_NOT_POSITIVE_VS_{name}")
    return sorted(set(blockers))


def _model_blockers(metrics, skill, folds):
    blockers = []
    auc = metrics["roc_auc"]
    if auc is None or auc <= MODEL_GATE["minimum_pooled_roc_auc"]: blockers.append("POOLED_ROC_AUC_GATE_FAILED")
    if skill <= MODEL_GATE["minimum_pooled_brier_skill"]: blockers.append("POOLED_BRIER_SKILL_GATE_FAILED")
    if metrics["probability_std"] < MODEL_GATE["minimum_probability_std"]: blockers.append("PROBABILITY_DISPERSION_TOO_LOW")
    fold_aucs = [float(item["metrics"]["roc_auc"] or 0.0) for item in folds]
    if min(fold_aucs) < MODEL_GATE["minimum_fold_roc_auc"]: blockers.append("FOLD_ROC_AUC_INSTABILITY")
    if sum(item > 0.5 for item in fold_aucs) < MODEL_GATE["minimum_folds_above_random_auc"]: blockers.append("INSUFFICIENT_FOLDS_ABOVE_RANDOM_AUC")
    if sum(float(item["brier_skill_vs_train_prior"]) > 0 for item in folds) < MODEL_GATE["minimum_folds_positive_brier_skill"]: blockers.append("INSUFFICIENT_FOLDS_POSITIVE_BRIER_SKILL")
    if min(item["event_count"] for item in folds) < MODEL_GATE["minimum_test_events_per_fold"]: blockers.append("FOLD_EVENT_SUPPORT_TOO_LOW")
    return blockers


def _calibration_blockers(artifact: BinaryCalibrationArtifact):
    return ["CALIBRATION_RANK_REVERSAL"] if artifact.method == "platt" and float(artifact.parameters["coefficient"]) <= 0 else []


def _brier_skill(metrics, prior_metrics):
    return 1.0 - float(metrics["brier"]) / float(prior_metrics["brier"])


def _reference_percentiles(reference, values):
    ordered = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ordered, np.asarray(values, dtype=float), side="right") / len(ordered)


def _model_rank(item):
    return (not item["selection_viable"], -float(item["selection_brier_skill_vs_prior"]), -float(item["selection_metrics"]["roc_auc"] or 0.0), item["candidate"])


def _policy_rank(item):
    lower = item["metrics"]["average_r_session_bootstrap_95"]["lower"]
    return (not item["selection_viable"], len(item["selection_blockers"]), -(lower if lower is not None else -1_000.0), -item["metrics"]["trade_count"])


def _policy_contract(policy):
    return {"activation_percentile": policy.activation_percentile, "maximum_trades_per_session": policy.maximum_trades_per_session}


def _session_key(value: datetime) -> str:
    return str(value.astimezone(IST).date())
