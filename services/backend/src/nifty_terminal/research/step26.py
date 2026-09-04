"""Step 26: live-plan-aligned price-action meta-label research.

Direction and levels come from the production price-action engine.  A compact
model may only accept or reject that deterministic setup.  It cannot reverse
the setup direction.  Historical execution uses the shared conservative replay
module, including the displayed entry zone, structure stop, T1/T2/T3, explicit
partial exits, one-way slippage and stop-first intraminute ambiguity handling.

The 2024-2026 history has already influenced development, so this experiment
can reject a design but can never approve it for live inference.  A passing
historical screen must be frozen and confirmed on genuinely future data.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
import hashlib
import json
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nifty_terminal.calendar.nse import IST
from nifty_terminal.context.features import ContextFeatureBuild
from nifty_terminal.domain.candle import Candle, Timeframe
from nifty_terminal.domain.enums import ConnectionState
from nifty_terminal.ml.definitions import RANDOM_SEED
from nifty_terminal.ml.models import WalkForwardConfig
from nifty_terminal.ml.split import PurgedWalkForwardSplitter
from nifty_terminal.price_action.engine import PRICE_ACTION_VERSION, PriceActionEngine
from nifty_terminal.price_action.models import SetupState
from nifty_terminal.price_action.replay import (
    EXECUTION_POLICY_CANDIDATES,
    PriceActionExecutionPolicy,
    PriceActionPathResult,
    replay_price_action_plan,
)
from nifty_terminal.research.step18b import (
    BinaryCalibrationArtifact,
    apply_binary_calibrator,
    binary_metrics,
    build_trade_paths,
    fit_binary_calibrator,
)
from nifty_terminal.research.step18f import (
    daily_uplift_bootstrap,
)
from nifty_terminal.research.step25 import feature_family_indices


STEP26_VERSION = "live_plan_meta_label_research.v1"
FEATURE_FAMILY = "STRUCTURE_LEVELS_COMPACT"
WALK_FORWARD_CONFIG = WalkForwardConfig(
    n_splits=7,
    minimum_train_samples=10_000,
    test_samples=2_000,
    purge_bars=12,
    embargo_bars=12,
    minimum_train_class_samples=25,
)
EXIT_SELECTION_FOLDS = (0, 1, 2)
MODEL_SELECTION_FOLDS = (0, 1, 2)
CALIBRATION_FOLD = 3
POLICY_SELECTION_FOLD = 4
HISTORICAL_DIAGNOSTIC_FOLDS = (5, 6)
MODEL_CANDIDATES = ("meta_logistic_c0p02", "meta_hgb_shallow")
CALIBRATION_METHODS = ("identity", "platt")
ACTIVATION_PERCENTILES = (0.50, 0.65, 0.80)
MAXIMUM_TRADES_PER_SESSION = 5
MODEL_GATE = {
    "minimum_auc": 0.52,
    "minimum_brier_skill": 0.0,
    "minimum_probability_std": 0.01,
    "minimum_folds_above_random": 2,
    "minimum_worst_fold_auc": 0.47,
}
EXIT_GATE = {
    "minimum_trades": 120,
    "minimum_sessions": 25,
    "minimum_profit_factor": 1.05,
    "minimum_average_r_lower_95": 0.0,
}
SELECTION_POLICY_GATE = {
    "minimum_trades": 60,
    "minimum_sessions": 20,
    "minimum_trades_per_session": 1.0,
    "minimum_win_rate": 0.52,
    "minimum_profit_factor": 1.15,
    "minimum_average_r_lower_95": 0.0,
    "maximum_drawdown_r": 12.0,
}
DIAGNOSTIC_POLICY_GATE = {
    "minimum_trades": 120,
    "minimum_sessions": 40,
    "minimum_trades_per_session": 1.0,
    "minimum_win_rate": 0.55,
    "minimum_profit_factor": 1.25,
    "minimum_average_r_lower_95": 0.0,
    "maximum_drawdown_r": 15.0,
    "minimum_daily_r_uplift_lower_95": 0.0,
}

RESEARCH_IDENTITY = hashlib.sha256(
    json.dumps(
        {
            "version": STEP26_VERSION,
            "price_action_version": PRICE_ACTION_VERSION,
            "feature_family": FEATURE_FAMILY,
            "walk_forward": WALK_FORWARD_CONFIG.to_contract(),
            "exit_policies": [item.to_contract() for item in EXECUTION_POLICY_CANDIDATES],
            "models": MODEL_CANDIDATES,
            "calibration": CALIBRATION_METHODS,
            "activation_percentiles": ACTIVATION_PERCENTILES,
            "maximum_trades_per_session": MAXIMUM_TRADES_PER_SESSION,
            "model_gate": MODEL_GATE,
            "exit_gate": EXIT_GATE,
            "selection_policy_gate": SELECTION_POLICY_GATE,
            "diagnostic_policy_gate": DIAGNOSTIC_POLICY_GATE,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


@dataclass(frozen=True, slots=True)
class LivePlanCandidate:
    sample_index: int
    sample_id: str
    decision_time: datetime
    direction: str
    confluence_score: int
    evidence_grade: str
    volatility_regime: str
    risk_atr: float
    paths: tuple[tuple[str, PriceActionPathResult], ...]

    def path(self, policy_name: str) -> PriceActionPathResult:
        return dict(self.paths)[policy_name]


@dataclass(frozen=True, slots=True)
class MetaPolicy:
    activation_percentile: float


def run_live_plan_meta_research(
    *,
    context: ContextFeatureBuild,
    minute_candles: tuple[Candle, ...],
    primary_candles: tuple[Candle, ...],
    context_15m_candles: tuple[Candle, ...],
    context_1h_candles: tuple[Candle, ...],
    context_bundle_sha256: str,
) -> dict[str, object]:
    samples, matrix, _, _, path_exclusions = build_trade_paths(
        dataset=context.dataset,
        features=context.matrix,
        minute_candles=minute_candles,
    )
    folds = PurgedWalkForwardSplitter().split(samples, WALK_FORWARD_CONFIG)
    candidates, candidate_exclusions = build_live_plan_candidates(
        samples=samples,
        minute_candles=minute_candles,
        primary_candles=primary_candles,
        context_15m_candles=context_15m_candles,
        context_1h_candles=context_1h_candles,
        label_by_id={item.label_id: item for item in context.dataset.labels},
    )
    if len(candidates) < 400:
        raise ValueError(
            f"Step 26 requires at least 400 entered price-action candidates; got {len(candidates)}"
        )
    candidate_by_sample = {item.sample_index: item for item in candidates}
    names = context.matrix.feature_names
    feature_indices = feature_family_indices(FEATURE_FAMILY, names)

    exit_selection_indices = _test_indices(folds, EXIT_SELECTION_FOLDS)
    exit_candidates = tuple(
        candidate_by_sample[index]
        for index in exit_selection_indices
        if index in candidate_by_sample
    )
    exit_reports = []
    for policy in EXECUTION_POLICY_CANDIDATES:
        metrics = replay_candidates(
            candidates=exit_candidates,
            policy_name=policy.name,
            scores=None,
            meta_policy=None,
        )
        blockers = _exit_blockers(metrics)
        exit_reports.append(
            {
                "policy": policy.to_contract(),
                "metrics": metrics,
                "selection_viable": not blockers,
                "selection_blockers": blockers,
            }
        )
    viable_exits = [item for item in exit_reports if item["selection_viable"]]
    selected_exit_report = min(viable_exits or exit_reports, key=_exit_rank)
    selected_exit_name = str(selected_exit_report["policy"]["name"])

    raw_predictions: dict[str, dict[int, dict[str, object]]] = {}
    model_reports = []
    for model_name in MODEL_CANDIDATES:
        by_fold = {}
        fold_reports = []
        for fold_index in MODEL_SELECTION_FOLDS:
            output = _fit_predict_meta(
                model_name=model_name,
                fold=folds[fold_index],
                candidates=candidates,
                matrix=matrix,
                feature_indices=feature_indices,
                exit_policy_name=selected_exit_name,
            )
            by_fold[fold_index] = output
            metrics = binary_metrics(output["actual"], output["probability"])
            prior_metrics = binary_metrics(output["actual"], output["prior"])
            fold_reports.append(
                {
                    "fold_index": fold_index,
                    "metrics": metrics,
                    "brier_skill_vs_prior": _brier_skill(metrics, prior_metrics),
                }
            )
        actual = np.concatenate([by_fold[index]["actual"] for index in MODEL_SELECTION_FOLDS])
        probability = np.concatenate(
            [by_fold[index]["probability"] for index in MODEL_SELECTION_FOLDS]
        )
        prior = np.concatenate([by_fold[index]["prior"] for index in MODEL_SELECTION_FOLDS])
        metrics = binary_metrics(actual, probability)
        prior_metrics = binary_metrics(actual, prior)
        skill = _brier_skill(metrics, prior_metrics)
        blockers = _model_blockers(metrics, skill, fold_reports)
        model_reports.append(
            {
                "candidate": model_name,
                "feature_count": 2 * len(feature_indices) + 1,
                "selection_metrics": metrics,
                "selection_brier_skill_vs_prior": skill,
                "folds": fold_reports,
                "selection_viable": not blockers,
                "selection_blockers": blockers,
            }
        )
        raw_predictions[model_name] = by_fold
    viable_models = [item for item in model_reports if item["selection_viable"]]
    selected_model_report = min(viable_models or model_reports, key=_model_rank)
    selected_model = str(selected_model_report["candidate"])
    for fold_index in (CALIBRATION_FOLD, POLICY_SELECTION_FOLD, *HISTORICAL_DIAGNOSTIC_FOLDS):
        raw_predictions[selected_model][fold_index] = _fit_predict_meta(
            model_name=selected_model,
            fold=folds[fold_index],
            candidates=candidates,
            matrix=matrix,
            feature_indices=feature_indices,
            exit_policy_name=selected_exit_name,
        )

    calibration_output = raw_predictions[selected_model][CALIBRATION_FOLD]
    calibrations = []
    for method in CALIBRATION_METHODS:
        artifact = fit_binary_calibrator(
            method=method,
            probabilities=calibration_output["probability"],
            actual=calibration_output["actual"],
            prior=float(calibration_output["train_prior"]),
        )
        calibrated = apply_binary_calibrator(artifact, calibration_output["probability"])
        blockers = (
            ["CALIBRATION_RANK_REVERSAL"]
            if artifact.method == "platt" and float(artifact.parameters["coefficient"]) <= 0
            else []
        )
        calibrations.append(
            {
                "method": method,
                "artifact": artifact,
                "metrics": binary_metrics(calibration_output["actual"], calibrated),
                "selection_blockers": blockers,
            }
        )
    viable_calibrations = [item for item in calibrations if not item["selection_blockers"]]
    selected_calibration = min(
        viable_calibrations,
        key=lambda item: (item["metrics"]["brier"], item["metrics"]["log_loss"]),
    )
    calibration: BinaryCalibrationArtifact = selected_calibration["artifact"]

    calibrated_by_fold = {
        fold_index: apply_binary_calibrator(calibration, output["probability"])
        for fold_index, output in raw_predictions[selected_model].items()
    }
    reference_scores = calibrated_by_fold[CALIBRATION_FOLD]
    policy_candidates = _fold_candidates(
        candidates, raw_predictions[selected_model][POLICY_SELECTION_FOLD]["test_sample_indices"]
    )
    policy_percentiles = _reference_percentiles(
        reference_scores, calibrated_by_fold[POLICY_SELECTION_FOLD]
    )
    policy_reports = []
    for activation in ACTIVATION_PERCENTILES:
        policy = MetaPolicy(activation)
        metrics = replay_candidates(
            candidates=policy_candidates,
            policy_name=selected_exit_name,
            scores=policy_percentiles,
            meta_policy=policy,
        )
        baseline = replay_candidates(
            candidates=policy_candidates,
            policy_name=selected_exit_name,
            scores=None,
            meta_policy=None,
        )
        metrics["baseline_daily_r_uplift_95"] = daily_uplift_bootstrap(
            metrics["daily_total_r"], baseline["daily_total_r"]
        )
        blockers = _policy_blockers(metrics, SELECTION_POLICY_GATE, require_uplift=False)
        if not selected_model_report["selection_viable"]:
            blockers.append("MODEL_SELECTION_GATE_FAILED")
        if not selected_exit_report["selection_viable"]:
            blockers.append("EXIT_POLICY_SELECTION_GATE_FAILED")
        policy_reports.append(
            {
                "policy": {"activation_percentile": activation},
                "metrics": metrics,
                "selection_viable": not blockers,
                "selection_blockers": sorted(set(blockers)),
            }
        )
    viable_policies = [item for item in policy_reports if item["selection_viable"]]
    selected_policy_report = min(viable_policies, key=_policy_rank) if viable_policies else None
    exploratory_policy_report = min(policy_reports, key=_policy_rank)
    evaluation_policy_report = selected_policy_report or exploratory_policy_report
    evaluation_policy = MetaPolicy(**evaluation_policy_report["policy"])

    diagnostic_outputs = [
        raw_predictions[selected_model][fold_index]
        for fold_index in HISTORICAL_DIAGNOSTIC_FOLDS
    ]
    diagnostic_candidates = tuple(
        item
        for output in diagnostic_outputs
        for item in _fold_candidates(candidates, output["test_sample_indices"])
    )
    diagnostic_probability = np.concatenate(
        [calibrated_by_fold[index] for index in HISTORICAL_DIAGNOSTIC_FOLDS]
    )
    diagnostic_reference = np.concatenate(
        (reference_scores, calibrated_by_fold[POLICY_SELECTION_FOLD])
    )
    diagnostic_percentiles = _reference_percentiles(
        diagnostic_reference, diagnostic_probability
    )
    diagnostic = replay_candidates(
        candidates=diagnostic_candidates,
        policy_name=selected_exit_name,
        scores=diagnostic_percentiles,
        meta_policy=evaluation_policy,
    )
    diagnostic_baseline = replay_candidates(
        candidates=diagnostic_candidates,
        policy_name=selected_exit_name,
        scores=None,
        meta_policy=None,
    )
    diagnostic["baseline_daily_r_uplift_95"] = daily_uplift_bootstrap(
        diagnostic["daily_total_r"], diagnostic_baseline["daily_total_r"]
    )
    diagnostic_blockers = _policy_blockers(
        diagnostic, DIAGNOSTIC_POLICY_GATE, require_uplift=True
    )
    if selected_policy_report is None:
        diagnostic_blockers.append("NO_POLICY_PASSED_SELECTION_GATE")

    diagnostic_actual = np.concatenate([item["actual"] for item in diagnostic_outputs])
    diagnostic_prior = np.concatenate([item["prior"] for item in diagnostic_outputs])
    diagnostic_model_metrics = binary_metrics(diagnostic_actual, diagnostic_probability)
    diagnostic_prior_metrics = binary_metrics(diagnostic_actual, diagnostic_prior)
    diagnostic_model_skill = _brier_skill(diagnostic_model_metrics, diagnostic_prior_metrics)
    diagnostic_model_blockers = []
    if diagnostic_model_metrics["roc_auc"] is None or diagnostic_model_metrics["roc_auc"] <= 0.51:
        diagnostic_model_blockers.append("DIAGNOSTIC_AUC_GATE_FAILED")
    if diagnostic_model_skill <= 0:
        diagnostic_model_blockers.append("DIAGNOSTIC_BRIER_SKILL_GATE_FAILED")

    historical_blockers = sorted(
        set(
            ([] if selected_exit_report["selection_viable"] else ["EXIT_POLICY_SELECTION_GATE_FAILED"])
            + ([] if selected_model_report["selection_viable"] else ["MODEL_SELECTION_GATE_FAILED"])
            + ([] if selected_policy_report is not None else ["NO_POLICY_PASSED_SELECTION_GATE"])
            + diagnostic_model_blockers
            + diagnostic_blockers
        )
    )
    release_blockers = sorted(
        set(
            historical_blockers
            + [
                "HISTORICAL_PERIOD_PREVIOUSLY_SEEN",
                "EXECUTABLE_FUTURES_HISTORY_NOT_AVAILABLE",
                "DERIVATIVES_FORWARD_DATA_NOT_READY",
                "FORWARD_CONFIRMATION_NOT_COMPLETED",
            ]
        )
    )
    return {
        "schema_version": 1,
        "step26_version": STEP26_VERSION,
        "research_identity": RESEARCH_IDENTITY,
        "dataset_id": context.dataset.dataset_id,
        "context_bundle_sha256": context_bundle_sha256,
        "objective": {
            "direction": "production price-action engine; meta-model cannot reverse it",
            "label": "net trade P&L after explicit partial exits and slippage is positive",
            "entry": "displayed trigger and entry zone; gaps outside zone are not chased",
            "stop_and_targets": "production structure stop with T1=1.25R, T2=2R, T3=3R",
            "horizon_minutes": 60,
            "same_minute_resolution": "STOP_FIRST_CONSERVATIVE",
            "maximum_trades_per_session": MAXIMUM_TRADES_PER_SESSION,
        },
        "dataset": {
            "complete_trade_paths": len(samples),
            "entered_price_action_candidates": len(candidates),
            "feature_family": FEATURE_FAMILY,
            "base_feature_count": len(feature_indices),
            "directional_design_feature_count": 2 * len(feature_indices) + 1,
            "trade_path_exclusions": path_exclusions,
            "candidate_exclusions": candidate_exclusions,
        },
        "chronology": {
            "walk_forward": WALK_FORWARD_CONFIG.to_contract(),
            "exit_selection_folds": list(EXIT_SELECTION_FOLDS),
            "model_selection_folds": list(MODEL_SELECTION_FOLDS),
            "calibration_fold": CALIBRATION_FOLD,
            "policy_selection_fold": POLICY_SELECTION_FOLD,
            "historical_diagnostic_folds": list(HISTORICAL_DIAGNOSTIC_FOLDS),
            "future_data_used_in_features": False,
            "diagnostic_thresholds_locked_before_diagnostic": True,
            "historical_period_previously_seen": True,
        },
        "execution_policy_selection": {
            "candidates": exit_reports,
            "selected": selected_exit_report,
            "gate": EXIT_GATE,
        },
        "model_comparison": model_reports,
        "selected_model": {
            "candidate": selected_model,
            "selection_viable": selected_model_report["selection_viable"],
            "selection_blockers": selected_model_report["selection_blockers"],
        },
        "calibration": {
            "selected": calibration.to_contract(),
            "comparison": [
                {
                    "method": item["method"],
                    "metrics": item["metrics"],
                    "selection_blockers": item["selection_blockers"],
                }
                for item in calibrations
            ],
        },
        "policy_selection": {
            "candidates": policy_reports,
            "selected": selected_policy_report,
            "best_exploratory_rejected": (
                None if selected_policy_report is not None else exploratory_policy_report
            ),
            "gate": SELECTION_POLICY_GATE,
        },
        "historical_diagnostic": {
            "evaluated_policy_source": (
                "SELECTED" if selected_policy_report is not None else "BEST_REJECTED_EXPLORATORY_ONLY"
            ),
            "policy": evaluation_policy_report["policy"],
            "metrics": diagnostic,
            "unfiltered_price_action_baseline": diagnostic_baseline,
            "gate": DIAGNOSTIC_POLICY_GATE,
            "gate_passed": not diagnostic_blockers and selected_policy_report is not None,
            "blockers": sorted(set(diagnostic_blockers)),
        },
        "diagnostic_model": {
            "metrics": diagnostic_model_metrics,
            "prior_metrics": diagnostic_prior_metrics,
            "brier_skill_vs_prior": diagnostic_model_skill,
            "gate_passed": not diagnostic_model_blockers,
            "blockers": diagnostic_model_blockers,
        },
        "research_gate": {
            "historical_screen_passed": not historical_blockers,
            "approved_for_live_inference": False,
            "blockers": release_blockers,
        },
        "model_artifact_created": False,
        "approved_for_live_inference": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
    }


def build_live_plan_candidates(
    *,
    samples,
    minute_candles: tuple[Candle, ...],
    primary_candles: tuple[Candle, ...],
    context_15m_candles: tuple[Candle, ...],
    context_1h_candles: tuple[Candle, ...],
    label_by_id,
) -> tuple[tuple[LivePlanCandidate, ...], dict[str, int]]:
    minute_by_open = {item.opens_at: item for item in minute_candles}
    primary_index = {item.candle_id: index for index, item in enumerate(primary_candles)}
    context_15_closes = [item.closes_at for item in context_15m_candles]
    context_1h_closes = [item.closes_at for item in context_1h_candles]
    engine = PriceActionEngine()
    candidates = []
    exclusions = Counter[str]()
    for sample_index, sample in enumerate(samples):
        index = primary_index.get(sample.primary_candle_id)
        if index is None:
            exclusions["PRIMARY_CANDLE_MISSING"] += 1
            continue
        end_15 = bisect_right(context_15_closes, sample.decision_time)
        end_1h = bisect_right(context_1h_closes, sample.decision_time)
        series = (
            *primary_candles[max(0, index - 749) : index + 1],
            *context_15m_candles[max(0, end_15 - 250) : end_15],
            *context_1h_candles[max(0, end_1h - 120) : end_1h],
        )
        snapshot = SimpleNamespace(
            data_status=ConnectionState.LIVE,
            snapshot_id=f"historical:{sample.sample_id}",
            candle_revision_checksum=sample.input_revision_checksum,
            instrument_id=sample.instrument_id,
            decision_time=sample.decision_time,
        )
        analysis = engine.analyze(
            SimpleNamespace(snapshot=snapshot, finalized_candles=series),
            generated_at=sample.decision_time,
        )
        if analysis.setup not in {SetupState.BUY_TRIGGER, SetupState.SELL_TRIGGER}:
            exclusions[f"SETUP_{analysis.setup.value}"] += 1
            continue
        if analysis.trade_plan is None:
            exclusions["TRADE_PLAN_UNAVAILABLE"] += 1
            continue
        window = tuple(
            minute_by_open.get(sample.decision_time + timedelta(minutes=offset))
            for offset in range(60)
        )
        if any(item is None for item in window):
            exclusions["INCOMPLETE_MINUTE_WINDOW"] += 1
            continue
        complete_window = tuple(item for item in window if item is not None)
        paths = tuple(
            (
                policy.name,
                replay_price_action_plan(
                    plan=analysis.trade_plan,
                    minute_candles=complete_window,
                    policy=policy,
                ),
            )
            for policy in EXECUTION_POLICY_CANDIDATES
        )
        if not paths[0][1].entered:
            exclusions[paths[0][1].status] += 1
            continue
        label = label_by_id[sample.label_id]
        atr = label.atr_at_decision
        if atr is None or atr <= 0:
            exclusions["ATR_UNAVAILABLE"] += 1
            continue
        candidates.append(
            LivePlanCandidate(
                sample_index=sample_index,
                sample_id=sample.sample_id,
                decision_time=sample.decision_time,
                direction=analysis.trade_plan.direction,
                confluence_score=analysis.confluence_score,
                evidence_grade=analysis.evidence_grade,
                volatility_regime=analysis.volatility_regime,
                risk_atr=float(analysis.trade_plan.risk_points / atr),
                paths=paths,
            )
        )
    return tuple(candidates), dict(exclusions)


def replay_candidates(
    *, candidates, policy_name: str, scores, meta_policy: MetaPolicy | None
) -> dict[str, object]:
    selected = []
    waits = Counter[str]()
    active_until = None
    daily_count = Counter[str]()
    if scores is not None and len(scores) != len(candidates):
        raise ValueError("Scores must align with price-action candidates")
    for index, candidate in enumerate(candidates):
        session = _session_key(candidate.decision_time)
        if active_until is not None and candidate.decision_time < active_until:
            waits["ACTIVE_POSITION"] += 1
            continue
        if daily_count[session] >= MAXIMUM_TRADES_PER_SESSION:
            waits["DAILY_TRADE_CAP"] += 1
            continue
        if meta_policy is not None and float(scores[index]) < meta_policy.activation_percentile:
            waits["META_SCORE_BELOW_THRESHOLD"] += 1
            continue
        path = candidate.path(policy_name)
        selected.append((candidate, path))
        daily_count[session] += 1
        active_until = path.exited_at
    return _replay_metrics(tuple(selected), len(candidates), waits, policy_name)


def _fit_predict_meta(
    *, model_name, fold, candidates, matrix, feature_indices, exit_policy_name
):
    train_set = set(fold.train_indices)
    test_set = set(fold.test_indices)
    train_candidates = tuple(item for item in candidates if item.sample_index in train_set)
    test_candidates = tuple(item for item in candidates if item.sample_index in test_set)
    if len(train_candidates) < 100 or len(test_candidates) < 20:
        raise ValueError("Step 26 fold has insufficient price-action candidate support")
    train_x = _meta_design(train_candidates, matrix, feature_indices)
    test_x = _meta_design(test_candidates, matrix, feature_indices)
    train_y = np.asarray(
        [item.path(exit_policy_name).profitable for item in train_candidates], dtype=int
    )
    test_y = np.asarray(
        [item.path(exit_policy_name).profitable for item in test_candidates], dtype=int
    )
    if len(set(train_y)) < 2 or len(set(test_y)) < 2:
        raise ValueError("Step 26 fold needs profitable and losing candidates")
    weights = _candidate_session_weights(train_candidates)
    model = _model(model_name)
    if isinstance(model, Pipeline):
        model.fit(train_x, train_y, model__sample_weight=weights)
    else:
        model.fit(train_x, train_y, sample_weight=weights)
    probability = model.predict_proba(test_x)[:, list(model.classes_).index(1)]
    prior = float(np.average(train_y, weights=weights))
    return {
        "actual": test_y,
        "probability": np.asarray(probability, dtype=float),
        "prior": np.full(len(test_y), prior),
        "train_prior": prior,
        "test_sample_indices": np.asarray(
            [item.sample_index for item in test_candidates], dtype=int
        ),
    }


def _meta_design(candidates, matrix, feature_indices) -> np.ndarray:
    rows = matrix[np.asarray([item.sample_index for item in candidates])][:, feature_indices]
    signs = np.asarray(
        [1.0 if item.direction == "BUY" else -1.0 for item in candidates]
    )
    return np.column_stack((rows, rows * signs[:, None], signs))


def _model(name: str):
    if name == "meta_logistic_c0p02":
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
    if name == "meta_hgb_shallow":
        return HistGradientBoostingClassifier(
            learning_rate=0.025,
            max_iter=140,
            max_leaf_nodes=7,
            min_samples_leaf=80,
            l2_regularization=12.0,
            random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown Step 26 model: {name}")


def _replay_metrics(selected, candidate_count, waits, policy_name):
    paths = tuple(item[1] for item in selected)
    r_values = np.asarray([float(item.r_multiple) for item in paths], dtype=float)
    points = np.asarray([float(item.net_points) for item in paths], dtype=float)
    gains = points[points > 0]
    losses = points[points < 0]
    cumulative = peak = drawdown = 0.0
    for value in r_values:
        cumulative += float(value)
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    daily = defaultdict(float)
    for candidate, path in selected:
        daily[_session_key(candidate.decision_time)] += float(path.r_multiple)
    sessions = Counter(_session_key(item.decision_time) for item, _ in selected)
    win_count = int(np.sum(points > 0))
    return {
        "execution_policy": policy_name,
        "candidate_count": candidate_count,
        "trade_count": len(paths),
        "buy_count": sum(item.direction == "BUY" for item in paths),
        "sell_count": sum(item.direction == "SELL" for item in paths),
        "session_count": len(sessions),
        "trades_per_session": len(paths) / len(sessions) if sessions else 0.0,
        "coverage": len(paths) / candidate_count if candidate_count else 0.0,
        "win_count": win_count,
        "win_rate": win_count / len(paths) if paths else None,
        "win_rate_wilson_95": _wilson_interval(win_count, len(paths)),
        "profit_factor": (
            float(np.sum(gains) / abs(np.sum(losses)))
            if len(gains) and len(losses)
            else None
        ),
        "net_points": float(np.sum(points)) if len(points) else 0.0,
        "average_points": float(np.mean(points)) if len(points) else None,
        "average_r_multiple": float(np.mean(r_values)) if len(r_values) else None,
        "average_r_session_bootstrap_95": _session_bootstrap(selected, r_values),
        "maximum_drawdown_r": drawdown,
        "target1_or_better_count": sum(item.maximum_target_reached >= 1 for item in paths),
        "target2_or_better_count": sum(item.maximum_target_reached >= 2 for item in paths),
        "target3_count": sum(item.maximum_target_reached >= 3 for item in paths),
        "stopped_count": sum(item.stop_hit for item in paths),
        "expired_count": sum(item.status == "EXPIRED" for item in paths),
        "wait_counts": dict(waits),
        "daily_total_r": dict(daily),
        "maximum_trades_per_session": MAXIMUM_TRADES_PER_SESSION,
        "hypothetical_spot_proxy_points_only": True,
        "rupee_pnl_available": False,
    }


def _wilson_interval(successes: int, count: int) -> dict[str, float | None]:
    if count == 0:
        return {"lower": None, "upper": None}
    z = 1.959963984540054
    probability = successes / count
    denominator = 1.0 + z * z / count
    centre = (probability + z * z / (2 * count)) / denominator
    half = z * np.sqrt(
        probability * (1 - probability) / count + z * z / (4 * count * count)
    ) / denominator
    return {"lower": float(centre - half), "upper": float(centre + half)}


def _session_bootstrap(selected, values, iterations: int = 2_000):
    groups = defaultdict(list)
    for (candidate, _), value in zip(selected, np.asarray(values, dtype=float)):
        groups[_session_key(candidate.decision_time)].append(float(value))
    sessions = tuple(sorted(groups))
    if len(sessions) < 2:
        return {
            "lower": None,
            "median": None,
            "upper": None,
            "session_count": len(sessions),
        }
    rng = np.random.default_rng(RANDOM_SEED)
    draws = []
    for _ in range(iterations):
        chosen = rng.choice(sessions, len(sessions), replace=True)
        cohort = [value for session in chosen for value in groups[str(session)]]
        draws.append(float(np.mean(cohort)))
    lower, median, upper = np.quantile(draws, (0.025, 0.5, 0.975))
    return {
        "lower": float(lower),
        "median": float(median),
        "upper": float(upper),
        "session_count": len(sessions),
    }


def _candidate_session_weights(candidates) -> np.ndarray:
    sessions = [_session_key(item.decision_time) for item in candidates]
    counts = Counter(sessions)
    weights = np.asarray([1.0 / counts[item] for item in sessions], dtype=float)
    return weights * len(weights) / np.sum(weights)


def _fold_candidates(candidates, sample_indices) -> tuple[LivePlanCandidate, ...]:
    lookup = {item.sample_index: item for item in candidates}
    return tuple(lookup[int(index)] for index in sample_indices)


def _test_indices(folds, fold_indices) -> tuple[int, ...]:
    return tuple(
        int(index)
        for fold_index in fold_indices
        for index in folds[fold_index].test_indices
    )


def _reference_percentiles(reference, values) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=float))
    return np.searchsorted(ordered, np.asarray(values, dtype=float), side="right") / len(ordered)


def _brier_skill(metrics, prior_metrics) -> float:
    return 1.0 - float(metrics["brier"]) / float(prior_metrics["brier"])


def _model_blockers(metrics, skill, folds):
    blockers = []
    if metrics["roc_auc"] is None or metrics["roc_auc"] <= MODEL_GATE["minimum_auc"]:
        blockers.append("META_MODEL_AUC_GATE_FAILED")
    if skill <= MODEL_GATE["minimum_brier_skill"]:
        blockers.append("META_MODEL_BRIER_SKILL_GATE_FAILED")
    if metrics["probability_std"] < MODEL_GATE["minimum_probability_std"]:
        blockers.append("META_MODEL_PROBABILITY_DISPERSION_TOO_LOW")
    fold_aucs = [float(item["metrics"]["roc_auc"] or 0.0) for item in folds]
    if sum(value > 0.5 for value in fold_aucs) < MODEL_GATE["minimum_folds_above_random"]:
        blockers.append("INSUFFICIENT_FOLDS_ABOVE_RANDOM")
    if min(fold_aucs) < MODEL_GATE["minimum_worst_fold_auc"]:
        blockers.append("META_MODEL_FOLD_INSTABILITY")
    return blockers


def _exit_blockers(metrics):
    blockers = []
    if metrics["trade_count"] < EXIT_GATE["minimum_trades"]:
        blockers.append("EXIT_TRADE_SUPPORT_TOO_LOW")
    if metrics["session_count"] < EXIT_GATE["minimum_sessions"]:
        blockers.append("EXIT_SESSION_SUPPORT_TOO_LOW")
    if metrics["profit_factor"] is None or metrics["profit_factor"] <= EXIT_GATE["minimum_profit_factor"]:
        blockers.append("EXIT_PROFIT_FACTOR_GATE_FAILED")
    lower = metrics["average_r_session_bootstrap_95"]["lower"]
    if lower is None or lower <= EXIT_GATE["minimum_average_r_lower_95"]:
        blockers.append("EXIT_EXPECTANCY_CONFIDENCE_GATE_FAILED")
    return blockers


def _policy_blockers(metrics, gate, *, require_uplift):
    blockers = []
    if metrics["trade_count"] < gate["minimum_trades"]:
        blockers.append("TRADE_SUPPORT_TOO_LOW")
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
    if require_uplift:
        uplift = metrics["baseline_daily_r_uplift_95"]["lower"]
        if uplift is None or uplift <= gate["minimum_daily_r_uplift_lower_95"]:
            blockers.append("DAILY_R_UPLIFT_NOT_POSITIVE_VS_UNFILTERED_PRICE_ACTION")
    return blockers


def _model_rank(item):
    return (
        not item["selection_viable"],
        -float(item["selection_brier_skill_vs_prior"]),
        -float(item["selection_metrics"]["roc_auc"] or 0.0),
        str(item["candidate"]),
    )


def _exit_rank(item):
    metrics = item["metrics"]
    lower = metrics["average_r_session_bootstrap_95"]["lower"]
    return (
        not item["selection_viable"],
        -(lower if lower is not None else -1_000.0),
        -float(metrics["profit_factor"] or 0.0),
        str(item["policy"]["name"]),
    )


def _policy_rank(item):
    metrics = item["metrics"]
    lower = metrics["average_r_session_bootstrap_95"]["lower"]
    return (
        not item["selection_viable"],
        len(item["selection_blockers"]),
        -(lower if lower is not None else -1_000.0),
        -int(metrics["trade_count"]),
    )


def _session_key(value: datetime) -> str:
    return str(value.astimezone(IST).date())
