"""Step 18E leakage-safe ranked utility cohort research.

Step 18D established small but statistically positive expected-R ranking skill.  Its
per-observation conformal lower bound was intentionally conservative, but is the
wrong activation object for bounded stop/target outcomes: almost every individual
lower bound lies near the stop.  This module keeps the same execution target and
models utility as a ranking problem.  Uncertainty is assessed on the selected
cohort of trades with a session-block bootstrap.
"""

from __future__ import annotations

from bisect import bisect_right, insort
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

import numpy as np

from nifty_terminal.calendar.nse import IST
from nifty_terminal.context.features import ContextFeatureBuild
from nifty_terminal.ml.definitions import RANDOM_SEED
from nifty_terminal.ml.split import PurgedWalkForwardSplitter
from nifty_terminal.research.step18b import TradePath, build_trade_paths
from nifty_terminal.research.step18d import (
    _candidate,
    _feature_indices,
    regression_metrics,
    session_bootstrap_mse_skill,
)
from nifty_terminal.research.v2 import NESTED_WALK_FORWARD_CONFIG


STEP18E_VERSION = "ranked_utility_cohort_research.v1"
MODEL_SELECTION_FOLDS = (0, 1)
SCORE_REFERENCE_FOLD = 2
POLICY_SELECTION_FOLD = 3
HISTORICAL_DIAGNOSTIC_FOLD = 4
ARCHITECTURES = ("NIFTY_ONLY", "NIFTY_BANK", "NIFTY_VIX", "ALL_CONTEXT")
MODEL_CANDIDATES = (
    "ridge_regularized_compact",
    "hgb_squared_shallow_compact",
    "extra_trees_regularized_compact",
)
PERCENTILE_THRESHOLDS = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
DIRECTION_MARGINS = (0.0, 0.05, 0.10)
MODEL_GATE = {
    "minimum_selection_mse_skill": 0.0,
    "minimum_selection_rank_correlation": 0.02,
    "minimum_diagnostic_mse_skill_lower_95": 0.0,
    "minimum_diagnostic_rank_correlation": 0.02,
}
POLICY_GATE = {
    "minimum_trades": 100,
    "minimum_buys": 30,
    "minimum_sells": 30,
    "minimum_profit_factor": 1.05,
    "minimum_average_r_lower_95": 0.0,
    "maximum_drawdown_r": 20.0,
}
RESEARCH_IDENTITY = hashlib.sha256(json.dumps(
    {
        "version": STEP18E_VERSION,
        "architectures": ARCHITECTURES,
        "models": MODEL_CANDIDATES,
        "percentiles": PERCENTILE_THRESHOLDS,
        "margins": DIRECTION_MARGINS,
        "model_gate": MODEL_GATE,
        "policy_gate": POLICY_GATE,
        "chronology": {
            "model_selection": MODEL_SELECTION_FOLDS,
            "score_reference": SCORE_REFERENCE_FOLD,
            "policy_selection": POLICY_SELECTION_FOLD,
            "diagnostic": HISTORICAL_DIAGNOSTIC_FOLD,
        },
    },
    sort_keys=True,
    separators=(",", ":"),
).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class RankedUtilityPolicy:
    minimum_score_percentile: float
    minimum_direction_percentile_margin: float


def run_ranked_utility_research(
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
    if len(samples) < 15_000:
        raise ValueError("Step 18E requires at least 15,000 complete trade paths")
    folds = PurgedWalkForwardSplitter().split(samples, NESTED_WALK_FORWARD_CONFIG)
    targets = {
        "LONG": np.asarray([long_paths[item.sample_id].r_multiple for item in samples]),
        "SHORT": np.asarray([short_paths[item.sample_id].r_multiple for item in samples]),
    }
    base_count = int(context.diagnostics["base_feature_count"])
    feature_names = context.matrix.feature_names
    architecture_indices = {
        name: architecture_feature_indices(name, feature_names, base_count)
        for name in ARCHITECTURES
    }

    comparisons: dict[str, list[dict[str, object]]] = {}
    selected: dict[str, dict[str, object]] = {}
    selected_predictions: dict[str, dict[int, np.ndarray]] = {}
    for direction, actual in targets.items():
        reports = []
        for architecture in ARCHITECTURES:
            architecture_base = architecture_indices[architecture]
            for candidate in MODEL_CANDIDATES:
                indices = _compact_intersection(
                    candidate, feature_names, architecture_base
                )
                by_fold: dict[int, np.ndarray] = {}
                fold_reports = []
                for fold_index in MODEL_SELECTION_FOLDS:
                    fold = folds[fold_index]
                    predicted = _fit_predict(
                        matrix, actual, fold.train_indices, fold.test_indices,
                        indices, candidate,
                    )
                    by_fold[fold_index] = predicted
                    test = np.asarray(fold.test_indices)
                    train = np.asarray(fold.train_indices)
                    baseline = np.full(len(test), float(np.mean(actual[train])))
                    fold_reports.append({
                        "fold_index": fold_index,
                        "metrics": regression_metrics(actual[test], predicted, baseline),
                    })
                selection_actual = np.concatenate([
                    actual[np.asarray(folds[index].test_indices)]
                    for index in MODEL_SELECTION_FOLDS
                ])
                selection_predicted = np.concatenate([
                    by_fold[index] for index in MODEL_SELECTION_FOLDS
                ])
                selection_baseline = np.concatenate([
                    np.full(
                        len(folds[index].test_indices),
                        float(np.mean(actual[np.asarray(folds[index].train_indices)])),
                    )
                    for index in MODEL_SELECTION_FOLDS
                ])
                metrics = regression_metrics(
                    selection_actual, selection_predicted, selection_baseline
                )
                blockers = _model_selection_blockers(metrics, fold_reports)
                reports.append({
                    "architecture": architecture,
                    "candidate": candidate,
                    "feature_count": len(indices),
                    "selection_metrics": metrics,
                    "folds": fold_reports,
                    "selection_viable": not blockers,
                    "selection_blockers": blockers,
                })
        viable = [item for item in reports if item["selection_viable"]]
        chosen = min(viable or reports, key=_model_rank)
        comparisons[direction] = reports
        selected[direction] = {
            "architecture": chosen["architecture"],
            "candidate": chosen["candidate"],
            "feature_count": chosen["feature_count"],
            "selection_viable": chosen["selection_viable"],
            "selection_blockers": chosen["selection_blockers"],
        }
        indices = _compact_intersection(
            str(chosen["candidate"]), feature_names,
            architecture_indices[str(chosen["architecture"])],
        )
        selected_predictions[direction] = {}
        for fold_index in (
            SCORE_REFERENCE_FOLD, POLICY_SELECTION_FOLD, HISTORICAL_DIAGNOSTIC_FOLD
        ):
            fold = folds[fold_index]
            selected_predictions[direction][fold_index] = _fit_predict(
                matrix, actual, fold.train_indices, fold.test_indices,
                indices, str(chosen["candidate"]),
            )

    percentiles: dict[str, dict[int, np.ndarray]] = {"LONG": {}, "SHORT": {}}
    for direction in ("LONG", "SHORT"):
        reference = selected_predictions[direction][SCORE_REFERENCE_FOLD]
        selection_scores = selected_predictions[direction][POLICY_SELECTION_FOLD]
        diagnostic_scores = selected_predictions[direction][HISTORICAL_DIAGNOSTIC_FOLD]
        percentiles[direction][POLICY_SELECTION_FOLD] = causal_score_percentiles(
            reference, selection_scores
        )
        percentiles[direction][HISTORICAL_DIAGNOSTIC_FOLD] = causal_score_percentiles(
            np.concatenate((reference, selection_scores)), diagnostic_scores
        )

    selection_indices = np.asarray(folds[POLICY_SELECTION_FOLD].test_indices)
    selection_samples = tuple(samples[index] for index in selection_indices)
    policy_reports = []
    for threshold in PERCENTILE_THRESHOLDS:
        for margin in DIRECTION_MARGINS:
            policy = RankedUtilityPolicy(threshold, margin)
            metrics = simulate_ranked_policy(
                samples=selection_samples,
                long_percentiles=percentiles["LONG"][POLICY_SELECTION_FOLD],
                short_percentiles=percentiles["SHORT"][POLICY_SELECTION_FOLD],
                long_paths=long_paths,
                short_paths=short_paths,
                policy=policy,
            )
            blockers = _policy_blockers(metrics)
            policy_reports.append({
                "policy": policy_contract(policy),
                "metrics": metrics,
                "selection_viable": not blockers,
                "selection_blockers": blockers,
            })
    viable_policies = [item for item in policy_reports if item["selection_viable"]]
    chosen_policy_report = min(viable_policies or policy_reports, key=_policy_rank)
    chosen_policy = RankedUtilityPolicy(**chosen_policy_report["policy"])

    diagnostic_indices = np.asarray(folds[HISTORICAL_DIAGNOSTIC_FOLD].test_indices)
    diagnostic_samples = tuple(samples[index] for index in diagnostic_indices)
    replay = simulate_ranked_policy(
        samples=diagnostic_samples,
        long_percentiles=percentiles["LONG"][HISTORICAL_DIAGNOSTIC_FOLD],
        short_percentiles=percentiles["SHORT"][HISTORICAL_DIAGNOSTIC_FOLD],
        long_paths=long_paths,
        short_paths=short_paths,
        policy=chosen_policy,
    )

    diagnostics = {}
    model_blockers = []
    for direction, actual in targets.items():
        fold = folds[HISTORICAL_DIAGNOSTIC_FOLD]
        test = np.asarray(fold.test_indices)
        train = np.asarray(fold.train_indices)
        predicted = selected_predictions[direction][HISTORICAL_DIAGNOSTIC_FOLD]
        baseline = np.full(len(test), float(np.mean(actual[train])))
        metrics = regression_metrics(actual[test], predicted, baseline)
        interval = session_bootstrap_mse_skill(
            samples=diagnostic_samples,
            actual=actual[test], predicted=predicted, baseline=baseline,
        )
        blockers = []
        if interval["lower"] <= MODEL_GATE["minimum_diagnostic_mse_skill_lower_95"]:
            blockers.append("MSE_SKILL_LOWER_95_NOT_POSITIVE")
        if metrics["rank_correlation"] <= MODEL_GATE["minimum_diagnostic_rank_correlation"]:
            blockers.append("RANK_CORRELATION_TOO_LOW")
        diagnostics[direction] = {
            "selected": selected[direction],
            "point_metrics": metrics,
            "session_block_bootstrap_mse_skill_95": interval,
            "causal_score_deciles": score_decile_table(
                percentiles[direction][HISTORICAL_DIAGNOSTIC_FOLD], actual[test]
            ),
            "gate_passed": not blockers,
            "blockers": blockers,
        }
        model_blockers.extend(f"{direction}_{item}" for item in blockers)

    replay_blockers = _policy_blockers(replay)
    selection_blockers = [] if viable_policies else ["NO_POLICY_PASSED_SELECTION_GATE"]
    historical_blockers = sorted(set(model_blockers + replay_blockers + selection_blockers))
    passed_before_forward = not historical_blockers
    all_blockers = historical_blockers + [
        "HISTORICAL_PERIOD_USED_FOR_MODEL_DEVELOPMENT",
        "FORWARD_CONFIRMATION_NOT_COMPLETED",
    ]
    return {
        "schema_version": 1,
        "step18e_version": STEP18E_VERSION,
        "research_identity": RESEARCH_IDENTITY,
        "dataset_id": context.dataset.dataset_id,
        "context_bundle_sha256": context_bundle_sha256,
        "objective": {
            "model_outputs": "expected realized LONG and SHORT R",
            "activation": "causal score percentile, never an individual outcome bound",
            "uncertainty": "session-block bootstrap of the selected trade cohort",
            "target_and_stop": "1.0 ATR target / 0.75 ATR stop",
            "horizon_minutes": 60,
            "entry": "next finalized 1m open",
            "slippage": "0.5 NIFTY points one way",
            "same_minute_resolution": "STOP_FIRST_CONSERVATIVE",
        },
        "dataset": {
            "complete_trade_paths": len(samples),
            "excluded_trade_paths": exclusions,
            "feature_count": matrix.shape[1],
        },
        "chronology": {
            "walk_forward": NESTED_WALK_FORWARD_CONFIG.to_contract(),
            "model_selection_folds": list(MODEL_SELECTION_FOLDS),
            "score_reference_fold": SCORE_REFERENCE_FOLD,
            "policy_selection_fold": POLICY_SELECTION_FOLD,
            "historical_diagnostic_fold": HISTORICAL_DIAGNOSTIC_FOLD,
            "diagnostic_thresholds_locked_before_diagnostic": True,
            "future_scores_used_in_percentiles": False,
            "final_historical_fold_previously_seen": True,
        },
        "feature_architectures": {
            name: {"feature_count": len(indices)}
            for name, indices in architecture_indices.items()
        },
        "model_comparison": comparisons,
        "selected_models": selected,
        "utility_diagnostics": diagnostics,
        "policy_selection": {
            "candidate_count": len(policy_reports),
            "passing_candidate_count": len(viable_policies),
            "selected": chosen_policy_report,
        },
        "historical_simulated_live_replay": replay,
        "research_gate": {
            "model_gate": MODEL_GATE,
            "policy_gate": POLICY_GATE,
            "passed_before_mandatory_forward_blockers": passed_before_forward,
            "historical_blockers": historical_blockers,
            "blockers": sorted(set(all_blockers)),
        },
        "methodological_correction": {
            "step18d_individual_conformal_retained_as_diagnostic": True,
            "individual_conformal_used_for_activation": False,
            "reason": "bounded stop/target outcomes make individual lower-tail bounds collapse near the stop",
            "replacement": "causal ranking plus cohort-level session bootstrap",
        },
        "known_limitations_not_silently_filled": [
            "HISTORICAL_NEWS_NOT_YET_AVAILABLE",
            "CONTINUOUS_FUTURES_VOLUME_OI_NOT_YET_AVAILABLE",
            "POINT_IN_TIME_CONSTITUENT_BREADTH_NOT_YET_AVAILABLE",
            "HISTORICAL_PERIOD_ALREADY_USED_BY_PRIOR_RESEARCH",
        ],
        "existing_shadow_runtime_modified": False,
        "model_artifact_created": False,
        "approved_for_live_inference": False,
        "precise_probability_display_allowed": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
    }


def architecture_feature_indices(
    architecture: str, names: tuple[str, ...], base_count: int
) -> np.ndarray:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown Step 18E architecture: {architecture}")
    allowed = []
    for index, name in enumerate(names):
        if index < base_count:
            allowed.append(index)
            continue
        bank = name.startswith("context_market__banknifty_spot__")
        vix = name.startswith("context_market__india_vix_spot__")
        bank_cross = name in {
            "cross__bank_minus_nifty_return_1",
            "cross__bank_minus_nifty_return_3",
            "cross__bank_nifty_trend_agreement",
        }
        vix_cross = name in {
            "cross__vix_return_times_nifty_return",
            "cross__vix_shock_abs",
            "cross__risk_off_score",
        }
        if architecture == "NIFTY_BANK" and (bank or bank_cross):
            allowed.append(index)
        elif architecture == "NIFTY_VIX" and (vix or vix_cross):
            allowed.append(index)
        elif architecture == "ALL_CONTEXT":
            allowed.append(index)
    return np.asarray(allowed, dtype=int)


def causal_score_percentiles(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Empirical percentiles using reference and earlier values, never future values."""
    history = sorted(float(item) for item in np.asarray(reference, dtype=float))
    if not history:
        raise ValueError("A non-empty historical score reference is required")
    result = []
    for value in np.asarray(values, dtype=float):
        result.append(bisect_right(history, float(value)) / len(history))
        insort(history, float(value))
    return np.asarray(result, dtype=float)


def simulate_ranked_policy(
    *, samples, long_percentiles, short_percentiles,
    long_paths, short_paths, policy: RankedUtilityPolicy,
) -> dict[str, object]:
    trades: list[TradePath] = []
    waits = Counter[str]()
    active_until: datetime | None = None
    for index, sample in enumerate(samples):
        if active_until is not None and sample.decision_time < active_until:
            waits["ACTIVE_POSITION"] += 1
            continue
        long_rank = float(long_percentiles[index])
        short_rank = float(short_percentiles[index])
        if max(long_rank, short_rank) < policy.minimum_score_percentile:
            waits["SCORE_PERCENTILE_BELOW_THRESHOLD"] += 1
            continue
        if abs(long_rank - short_rank) < policy.minimum_direction_percentile_margin:
            waits["DIRECTION_PERCENTILE_MARGIN_TOO_SMALL"] += 1
            continue
        path = (
            long_paths[sample.sample_id]
            if long_rank > short_rank else short_paths[sample.sample_id]
        )
        trades.append(path)
        active_until = path.exited_at
    return ranked_replay(tuple(trades), len(samples), waits, policy)


def ranked_replay(
    trades: tuple[TradePath, ...], decisions: int, waits: Counter[str],
    policy: RankedUtilityPolicy,
) -> dict[str, object]:
    points = [item.net_points for item in trades]
    r_values = [item.r_multiple for item in trades]
    gains = [item for item in points if item > 0]
    losses = [item for item in points if item < 0]
    cumulative = peak = drawdown = 0.0
    for value in r_values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    return {
        "policy": policy_contract(policy),
        "evaluation_decisions": decisions,
        "trade_count": len(trades),
        "buy_count": sum(item.direction == "LONG" for item in trades),
        "sell_count": sum(item.direction == "SHORT" for item in trades),
        "coverage": len(trades) / decisions if decisions else 0.0,
        "target_hit_count": sum(item.exit_reason == "TARGET" for item in trades),
        "stop_hit_count": sum(item.exit_reason == "STOP" for item in trades),
        "expired_count": sum(item.exit_reason == "HORIZON" for item in trades),
        "win_rate": len(gains) / len(trades) if trades else None,
        "net_points": float(sum(points)),
        "average_points": float(np.mean(points)) if points else None,
        "average_r_multiple": float(np.mean(r_values)) if r_values else None,
        "average_r_session_bootstrap_95": trade_cohort_bootstrap(trades),
        "profit_factor": sum(gains) / abs(sum(losses)) if gains and losses else None,
        "maximum_drawdown_r": drawdown,
        "wait_counts": dict(waits),
        "hypothetical_index_points_only": True,
        "rupee_pnl_available": False,
    }


def trade_cohort_bootstrap(
    trades: tuple[TradePath, ...], iterations: int = 1_000
) -> dict[str, float | int | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        groups[str(trade.decision_time.astimezone(IST).date())].append(trade.r_multiple)
    sessions = tuple(sorted(groups))
    if len(sessions) < 2:
        return {"lower": None, "median": None, "upper": None, "session_count": len(sessions)}
    rng = np.random.default_rng(RANDOM_SEED)
    values = []
    for _ in range(iterations):
        chosen = rng.choice(sessions, len(sessions), replace=True)
        cohort = [value for session in chosen for value in groups[str(session)]]
        values.append(float(np.mean(cohort)))
    lower, median, upper = np.quantile(values, (0.025, 0.5, 0.975))
    return {
        "lower": float(lower), "median": float(median), "upper": float(upper),
        "session_count": len(sessions),
    }


def score_decile_table(percentiles: np.ndarray, actual: np.ndarray) -> list[dict[str, object]]:
    buckets = []
    percentiles = np.asarray(percentiles, dtype=float)
    actual = np.asarray(actual, dtype=float)
    for decile in range(10):
        lower = decile / 10.0
        upper = (decile + 1) / 10.0
        mask = (percentiles >= lower) & (
            percentiles <= upper if decile == 9 else percentiles < upper
        )
        values = actual[mask]
        buckets.append({
            "decile": decile + 1,
            "lower_percentile": lower,
            "upper_percentile": upper,
            "sample_count": int(len(values)),
            "average_realized_r": float(np.mean(values)) if len(values) else None,
        })
    return buckets


def _compact_intersection(
    candidate: str, names: tuple[str, ...], architecture_indices: np.ndarray
) -> np.ndarray:
    compact = set(int(item) for item in _feature_indices(candidate, names))
    return np.asarray(
        [int(item) for item in architecture_indices if int(item) in compact], dtype=int
    )


def _fit_predict(matrix, actual, train_indices, test_indices, indices, candidate):
    train = np.asarray(train_indices)
    test = np.asarray(test_indices)
    model = _candidate(candidate)
    model.fit(matrix[train][:, indices], actual[train])
    return np.asarray(model.predict(matrix[test][:, indices]), dtype=float)


def _model_selection_blockers(metrics, fold_reports) -> list[str]:
    blockers = []
    if metrics["mse_skill_vs_baseline"] <= MODEL_GATE["minimum_selection_mse_skill"]:
        blockers.append("MSE_SKILL_GATE_FAILED")
    if metrics["rank_correlation"] <= MODEL_GATE["minimum_selection_rank_correlation"]:
        blockers.append("RANK_CORRELATION_GATE_FAILED")
    if metrics["predicted_std_r"] < 0.01:
        blockers.append("PREDICTION_COLLAPSE")
    if any(item["metrics"]["rank_correlation"] <= 0 for item in fold_reports):
        blockers.append("NON_POSITIVE_RANK_CORRELATION_IN_SELECTION_FOLD")
    return blockers


def _policy_blockers(metrics: dict[str, object]) -> list[str]:
    blockers = []
    for key, gate, label in (
        ("trade_count", "minimum_trades", "TRADE_SUPPORT_TOO_LOW"),
        ("buy_count", "minimum_buys", "BUY_SUPPORT_TOO_LOW"),
        ("sell_count", "minimum_sells", "SELL_SUPPORT_TOO_LOW"),
    ):
        if metrics[key] < POLICY_GATE[gate]:
            blockers.append(label)
    if metrics["profit_factor"] is None or metrics["profit_factor"] <= POLICY_GATE["minimum_profit_factor"]:
        blockers.append("PROFIT_FACTOR_GATE_FAILED")
    lower = metrics["average_r_session_bootstrap_95"]["lower"]
    if lower is None or lower <= POLICY_GATE["minimum_average_r_lower_95"]:
        blockers.append("EXPECTANCY_LOWER_95_NOT_POSITIVE")
    if metrics["maximum_drawdown_r"] > POLICY_GATE["maximum_drawdown_r"]:
        blockers.append("MAXIMUM_DRAWDOWN_GATE_FAILED")
    return blockers


def _model_rank(item: dict[str, object]) -> tuple[object, ...]:
    metrics = item["selection_metrics"]
    architecture_complexity = ARCHITECTURES.index(str(item["architecture"]))
    return (
        not item["selection_viable"],
        -metrics["rank_correlation"],
        -metrics["mse_skill_vs_baseline"],
        architecture_complexity,
        item["candidate"],
    )


def _policy_rank(item: dict[str, object]) -> tuple[object, ...]:
    metrics = item["metrics"]
    lower = metrics["average_r_session_bootstrap_95"]["lower"]
    return (
        not item["selection_viable"],
        -(lower if lower is not None else -1_000.0),
        -(metrics["average_r_multiple"] if metrics["average_r_multiple"] is not None else -1_000.0),
        -min(metrics["buy_count"], metrics["sell_count"]),
        item["policy"]["minimum_score_percentile"],
    )


def policy_contract(policy: RankedUtilityPolicy) -> dict[str, float]:
    return {
        "minimum_score_percentile": policy.minimum_score_percentile,
        "minimum_direction_percentile_margin": policy.minimum_direction_percentile_margin,
    }
