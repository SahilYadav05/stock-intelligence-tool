"""Step 18D selective utility modelling with conformal downside bounds."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nifty_terminal.calendar.nse import IST
from nifty_terminal.context.features import ContextFeatureBuild
from nifty_terminal.ml.definitions import RANDOM_SEED
from nifty_terminal.ml.split import PurgedWalkForwardSplitter
from nifty_terminal.research.step18b import TradePath, build_trade_paths
from nifty_terminal.research.v2 import NESTED_WALK_FORWARD_CONFIG


STEP18D_VERSION = "selective_utility_research.v1"
MODEL_SELECTION_FOLDS = (0, 1)
CONFORMAL_FIT_FOLD = 2
POLICY_SELECTION_FOLD = 3
HISTORICAL_DIAGNOSTIC_FOLD = 4
CONFORMAL_LOWER_QUANTILE = 0.20
CANDIDATES = (
    "ridge_regularized_all",
    "ridge_regularized_compact",
    "hgb_squared_shallow_all",
    "hgb_absolute_shallow_all",
    "hgb_squared_shallow_compact",
    "extra_trees_regularized_compact",
)
MODEL_GATE = {
    "minimum_mse_skill_vs_train_mean": 0.0,
    "minimum_lower_95_mse_skill": 0.0,
    "minimum_rank_correlation": 0.02,
    "minimum_conformal_coverage": 0.70,
    "maximum_conformal_coverage": 0.95,
    "minimum_context_mse_skill_delta": 0.0,
}
POLICY_GATE = {
    "minimum_trades": 100,
    "minimum_buys": 30,
    "minimum_sells": 30,
    "minimum_profit_factor": 1.05,
    "minimum_lower_95_average_r": 0.0,
    "maximum_drawdown_r": 20.0,
}
RESEARCH_IDENTITY = hashlib.sha256(json.dumps(
    {
        "version": STEP18D_VERSION,
        "candidates": CANDIDATES,
        "model_gate": MODEL_GATE,
        "policy_gate": POLICY_GATE,
        "conformal_quantile": CONFORMAL_LOWER_QUANTILE,
    },
    sort_keys=True,
    separators=(",", ":"),
).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class UtilityPolicy:
    minimum_lower_utility: float
    minimum_direction_margin: float


def run_selective_utility_research(
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
        raise ValueError("Step 18D requires at least 15,000 complete trade paths")
    folds = PurgedWalkForwardSplitter().split(samples, NESTED_WALK_FORWARD_CONFIG)
    targets = {
        "LONG": np.asarray([long_paths[item.sample_id].r_multiple for item in samples]),
        "SHORT": np.asarray([short_paths[item.sample_id].r_multiple for item in samples]),
    }
    candidate_reports: dict[str, list[dict[str, object]]] = {}
    predictions: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    selected: dict[str, str] = {}
    for direction, actual in targets.items():
        reports = []
        predictions[direction] = {}
        for candidate in CANDIDATES:
            by_fold = {}
            fold_reports = []
            indices = _feature_indices(candidate, context.matrix.feature_names)
            for fold in folds:
                train = np.asarray(fold.train_indices)
                test = np.asarray(fold.test_indices)
                model = _candidate(candidate)
                model.fit(matrix[train][:, indices], actual[train])
                predicted = np.asarray(model.predict(matrix[test][:, indices]), dtype=float)
                by_fold[fold.fold_index] = predicted
                baseline = np.full(len(test), float(np.mean(actual[train])))
                fold_reports.append({
                    "fold_index": fold.fold_index,
                    "metrics": regression_metrics(actual[test], predicted, baseline),
                })
            selection_actual = np.concatenate([
                actual[np.asarray(folds[index].test_indices)] for index in MODEL_SELECTION_FOLDS
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
            blockers = _selection_blockers(metrics)
            reports.append({
                "name": candidate,
                "feature_count": len(indices),
                "selection_metrics": metrics,
                "selection_viable": not blockers,
                "selection_blockers": blockers,
                "folds": fold_reports,
            })
            predictions[direction][candidate] = by_fold
        viable = [item for item in reports if item["selection_viable"]]
        pool = viable or reports
        chosen = min(pool, key=_model_rank)
        selected[direction] = str(chosen["name"])
        candidate_reports[direction] = reports

    corrections = {}
    lower_predictions: dict[str, dict[int, np.ndarray]] = {}
    point_predictions: dict[str, dict[int, np.ndarray]] = {}
    diagnostics = {}
    base_count = int(context.diagnostics["base_feature_count"])
    for direction, actual in targets.items():
        candidate = selected[direction]
        by_fold = predictions[direction][candidate]
        fit_indices = np.asarray(folds[CONFORMAL_FIT_FOLD].test_indices)
        residual = actual[fit_indices] - by_fold[CONFORMAL_FIT_FOLD]
        correction = float(np.quantile(residual, CONFORMAL_LOWER_QUANTILE))
        corrections[direction] = correction
        lower_predictions[direction] = {
            index: by_fold[index] + correction for index in range(len(folds))
        }
        point_predictions[direction] = by_fold
        diagnostic_indices = np.asarray(folds[HISTORICAL_DIAGNOSTIC_FOLD].test_indices)
        train_indices = np.asarray(folds[HISTORICAL_DIAGNOSTIC_FOLD].train_indices)
        baseline = np.full(
            len(diagnostic_indices), float(np.mean(actual[train_indices]))
        )
        point = by_fold[HISTORICAL_DIAGNOSTIC_FOLD]
        lower = lower_predictions[direction][HISTORICAL_DIAGNOSTIC_FOLD]
        metrics = regression_metrics(actual[diagnostic_indices], point, baseline)
        interval = session_bootstrap_mse_skill(
            samples=tuple(samples[index] for index in diagnostic_indices),
            actual=actual[diagnostic_indices],
            predicted=point,
            baseline=baseline,
        )
        base_control = _base_control(
            candidate=candidate,
            base_count=base_count,
            feature_names=context.matrix.feature_names,
            matrix=matrix,
            actual=actual,
            fold=folds[HISTORICAL_DIAGNOSTIC_FOLD],
        )
        context_delta = metrics["mse_skill_vs_baseline"] - base_control["mse_skill_vs_baseline"]
        coverage = float(np.mean(actual[diagnostic_indices] >= lower))
        blockers = _diagnostic_model_blockers(
            metrics=metrics,
            interval=interval,
            coverage=coverage,
            context_delta=context_delta,
        )
        diagnostics[direction] = {
            "selected_candidate": candidate,
            "conformal_residual_quantile": correction,
            "point_metrics": metrics,
            "session_block_bootstrap_mse_skill_95": interval,
            "lower_bound_empirical_coverage": coverage,
            "nifty_only_base_control": base_control,
            "context_mse_skill_delta": context_delta,
            "gate_passed": not blockers,
            "blockers": blockers,
        }

    policies = tuple(
        UtilityPolicy(minimum, margin)
        for minimum in (-0.20, -0.10, 0.0, 0.05, 0.10, 0.20)
        for margin in (0.0, 0.05, 0.10, 0.20)
    )
    selection_indices = np.asarray(folds[POLICY_SELECTION_FOLD].test_indices)
    selection_samples = tuple(samples[index] for index in selection_indices)
    policy_reports = []
    for policy in policies:
        metrics = simulate_utility_policy(
            samples=selection_samples,
            long_lower=lower_predictions["LONG"][POLICY_SELECTION_FOLD],
            short_lower=lower_predictions["SHORT"][POLICY_SELECTION_FOLD],
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
    chosen_policy = UtilityPolicy(**chosen_policy_report["policy"])
    diagnostic_indices = np.asarray(folds[HISTORICAL_DIAGNOSTIC_FOLD].test_indices)
    diagnostic_samples = tuple(samples[index] for index in diagnostic_indices)
    replay = simulate_utility_policy(
        samples=diagnostic_samples,
        long_lower=lower_predictions["LONG"][HISTORICAL_DIAGNOSTIC_FOLD],
        short_lower=lower_predictions["SHORT"][HISTORICAL_DIAGNOSTIC_FOLD],
        long_paths=long_paths,
        short_paths=short_paths,
        policy=chosen_policy,
    )
    replay_blockers = _policy_blockers(replay)
    blockers = [
        f"{direction}_{item}"
        for direction, report in diagnostics.items()
        for item in report["blockers"]
    ]
    blockers.extend(replay_blockers)
    if not viable_policies:
        blockers.append("NO_POLICY_PASSED_SELECTION_GATE")
    historical_gate_passed = not blockers
    blockers.extend((
        "HISTORICAL_PERIOD_USED_FOR_MODEL_DEVELOPMENT",
        "FORWARD_CONFIRMATION_NOT_COMPLETED",
    ))
    return {
        "schema_version": 1,
        "step18d_version": STEP18D_VERSION,
        "research_identity": RESEARCH_IDENTITY,
        "dataset_id": context.dataset.dataset_id,
        "context_bundle_sha256": context_bundle_sha256,
        "objective": {
            "outputs": [
                "expected realized LONG R under fixed execution",
                "expected realized SHORT R under fixed execution",
                "conformal 20th-percentile downside estimate for each direction",
            ],
            "target_and_stop": "same 1.0 ATR target / 0.75 ATR stop as Step 18B",
            "horizon_minutes": 60,
            "entry": "next finalized 1m open",
            "slippage": "0.5 NIFTY points one way",
            "same_minute_resolution": "STOP_FIRST_CONSERVATIVE",
        },
        "feature_architecture": context.diagnostics,
        "dataset": {
            "complete_trade_paths": len(samples),
            "excluded_trade_paths": exclusions,
            "feature_count": matrix.shape[1],
        },
        "chronology": {
            "walk_forward": NESTED_WALK_FORWARD_CONFIG.to_contract(),
            "model_selection_folds": list(MODEL_SELECTION_FOLDS),
            "conformal_fit_fold": CONFORMAL_FIT_FOLD,
            "policy_selection_fold": POLICY_SELECTION_FOLD,
            "historical_diagnostic_fold": HISTORICAL_DIAGNOSTIC_FOLD,
            "final_historical_fold_previously_seen": True,
        },
        "candidate_comparison": candidate_reports,
        "selected_candidates": selected,
        "utility_diagnostics": diagnostics,
        "conformal": {
            "lower_quantile": CONFORMAL_LOWER_QUANTILE,
            "corrections": corrections,
            "fit_fold": CONFORMAL_FIT_FOLD,
            "future_labels_used_at_inference": False,
        },
        "policy_selection": {
            "candidate_count": len(policy_reports),
            "passing_candidate_count": len(viable_policies),
            "selected": chosen_policy_report,
        },
        "historical_simulated_live_replay": replay,
        "research_gate": {
            "model_gate": MODEL_GATE,
            "policy_gate": POLICY_GATE,
            "passed_before_mandatory_forward_blockers": historical_gate_passed,
            "blockers": sorted(set(blockers)),
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


def regression_metrics(
    actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    mse = float(np.mean((actual - predicted) ** 2))
    baseline_mse = float(np.mean((actual - baseline) ** 2))
    return {
        "sample_count": len(actual),
        "actual_mean_r": float(np.mean(actual)),
        "predicted_mean_r": float(np.mean(predicted)),
        "predicted_std_r": float(np.std(predicted)),
        "mae": float(np.mean(np.abs(actual - predicted))),
        "mse": mse,
        "baseline_mse": baseline_mse,
        "mse_skill_vs_baseline": 1.0 - mse / baseline_mse if baseline_mse else -1.0,
        "rank_correlation": _rank_correlation(actual, predicted),
    }


def session_bootstrap_mse_skill(
    *, samples, actual, predicted, baseline, iterations: int = 1_000
) -> dict[str, float | int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[str(sample.decision_time.astimezone(IST).date())].append(index)
    days = tuple(sorted(groups))
    if len(days) < 2:
        return {"lower": -1.0, "median": -1.0, "upper": -1.0, "session_count": len(days)}
    rng = np.random.default_rng(RANDOM_SEED)
    values = []
    for _ in range(iterations):
        chosen = rng.choice(days, len(days), replace=True)
        indices = np.asarray([index for day in chosen for index in groups[str(day)]])
        mse = float(np.mean((actual[indices] - predicted[indices]) ** 2))
        base = float(np.mean((actual[indices] - baseline[indices]) ** 2))
        values.append(1.0 - mse / base if base else -1.0)
    lower, median, upper = np.quantile(values, (0.025, 0.5, 0.975))
    return {
        "lower": float(lower), "median": float(median), "upper": float(upper),
        "session_count": len(days),
    }


def simulate_utility_policy(
    *, samples, long_lower, short_lower, long_paths, short_paths, policy: UtilityPolicy
) -> dict[str, object]:
    trades = []
    waits = Counter[str]()
    active_until: datetime | None = None
    for index, sample in enumerate(samples):
        if active_until is not None and sample.decision_time < active_until:
            waits["ACTIVE_POSITION"] += 1
            continue
        long_value = float(long_lower[index])
        short_value = float(short_lower[index])
        best = max(long_value, short_value)
        if best < policy.minimum_lower_utility:
            waits["LOWER_UTILITY_BELOW_THRESHOLD"] += 1
            continue
        if abs(long_value - short_value) < policy.minimum_direction_margin:
            waits["DIRECTION_MARGIN_TOO_SMALL"] += 1
            continue
        path = long_paths[sample.sample_id] if long_value > short_value else short_paths[sample.sample_id]
        trades.append(path)
        active_until = path.exited_at
    return _replay(tuple(trades), len(samples), waits, policy)


def _replay(trades, decisions: int, waits: Counter[str], policy: UtilityPolicy) -> dict[str, object]:
    points = [item.net_points for item in trades]
    r_values = [item.r_multiple for item in trades]
    gains = [item for item in points if item > 0]
    losses = [item for item in points if item < 0]
    cumulative = peak = drawdown = 0.0
    for value in r_values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    interval = _trade_bootstrap(trades)
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
        "average_r_session_bootstrap_95": interval,
        "profit_factor": sum(gains) / abs(sum(losses)) if gains and losses else None,
        "maximum_drawdown_r": drawdown,
        "wait_counts": dict(waits),
        "hypothetical_index_points_only": True,
        "rupee_pnl_available": False,
    }


def _trade_bootstrap(trades, iterations: int = 1_000) -> dict[str, float | int | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        groups[str(trade.decision_time.astimezone(IST).date())].append(trade.r_multiple)
    days = tuple(sorted(groups))
    if len(days) < 2:
        return {"lower": None, "median": None, "upper": None, "session_count": len(days)}
    rng = np.random.default_rng(RANDOM_SEED)
    values = []
    for _ in range(iterations):
        chosen = rng.choice(days, len(days), replace=True)
        selected = [value for day in chosen for value in groups[str(day)]]
        values.append(float(np.mean(selected)))
    lower, median, upper = np.quantile(values, (0.025, 0.5, 0.975))
    return {"lower": float(lower), "median": float(median), "upper": float(upper), "session_count": len(days)}


def _candidate(name: str):
    if name.startswith("ridge_"):
        return Pipeline((("scale", StandardScaler()), ("model", Ridge(alpha=10.0))))
    if name.startswith("hgb_absolute"):
        return HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.03, max_iter=180,
            max_leaf_nodes=7, min_samples_leaf=60, l2_regularization=5.0,
            random_state=RANDOM_SEED,
        )
    if name.startswith("hgb_"):
        return HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.03, max_iter=180,
            max_leaf_nodes=7, min_samples_leaf=60, l2_regularization=5.0,
            random_state=RANDOM_SEED,
        )
    if name.startswith("extra_trees"):
        return ExtraTreesRegressor(
            n_estimators=240, max_depth=8, min_samples_leaf=30,
            max_features=0.5, n_jobs=-1, random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown Step 18D candidate: {name}")


def _feature_indices(name: str, names: tuple[str, ...]) -> np.ndarray:
    if name.endswith("_all"):
        return np.arange(len(names))
    excluded_fragments = (
        "__doji", "__hammer", "__shooting_star", "__bullish_engulfing",
        "__bearish_engulfing", "__inside_bar", "__outside_bar",
    )
    return np.asarray([
        index for index, feature in enumerate(names)
        if not any(fragment in feature for fragment in excluded_fragments)
    ])


def _base_control(*, candidate, base_count, feature_names, matrix, actual, fold) -> dict[str, float]:
    train = np.asarray(fold.train_indices)
    test = np.asarray(fold.test_indices)
    names = feature_names[:base_count]
    indices = _feature_indices(candidate, names)
    model = _candidate(candidate)
    model.fit(matrix[train, :base_count][:, indices], actual[train])
    predicted = model.predict(matrix[test, :base_count][:, indices])
    baseline = np.full(len(test), float(np.mean(actual[train])))
    return regression_metrics(actual[test], predicted, baseline)


def _selection_blockers(metrics: dict[str, float]) -> list[str]:
    blockers = []
    if metrics["mse_skill_vs_baseline"] <= 0:
        blockers.append("MSE_SKILL_GATE_FAILED")
    if metrics["rank_correlation"] <= 0:
        blockers.append("RANK_CORRELATION_GATE_FAILED")
    if metrics["predicted_std_r"] < 0.01:
        blockers.append("PREDICTION_COLLAPSE")
    return blockers


def _diagnostic_model_blockers(*, metrics, interval, coverage, context_delta) -> list[str]:
    blockers = _selection_blockers(metrics)
    if interval["lower"] <= MODEL_GATE["minimum_lower_95_mse_skill"]:
        blockers.append("MSE_SKILL_LOWER_95_NOT_POSITIVE")
    if metrics["rank_correlation"] <= MODEL_GATE["minimum_rank_correlation"]:
        blockers.append("RANK_CORRELATION_TOO_LOW")
    if not MODEL_GATE["minimum_conformal_coverage"] <= coverage <= MODEL_GATE["maximum_conformal_coverage"]:
        blockers.append("CONFORMAL_COVERAGE_OUTSIDE_GATE")
    if context_delta <= MODEL_GATE["minimum_context_mse_skill_delta"]:
        blockers.append("CONTEXT_DID_NOT_IMPROVE_MSE_SKILL")
    return blockers


def _policy_blockers(metrics: dict[str, object]) -> list[str]:
    blockers = []
    for key, gate, label in (
        ("trade_count", "minimum_trades", "TRADE_SUPPORT_TOO_LOW"),
        ("buy_count", "minimum_buys", "BUY_SUPPORT_TOO_LOW"),
        ("sell_count", "minimum_sells", "SELL_SUPPORT_TOO_LOW"),
    ):
        if metrics[key] < POLICY_GATE[gate]: blockers.append(label)
    if metrics["profit_factor"] is None or metrics["profit_factor"] <= POLICY_GATE["minimum_profit_factor"]:
        blockers.append("PROFIT_FACTOR_GATE_FAILED")
    lower = metrics["average_r_session_bootstrap_95"]["lower"]
    if lower is None or lower <= POLICY_GATE["minimum_lower_95_average_r"]:
        blockers.append("EXPECTANCY_LOWER_95_NOT_POSITIVE")
    if metrics["maximum_drawdown_r"] > POLICY_GATE["maximum_drawdown_r"]:
        blockers.append("MAXIMUM_DRAWDOWN_GATE_FAILED")
    return blockers


def _model_rank(item: dict[str, object]) -> tuple[object, ...]:
    metrics = item["selection_metrics"]
    return (not item["selection_viable"], metrics["mse"], -metrics["rank_correlation"], item["name"])


def _policy_rank(item: dict[str, object]) -> tuple[object, ...]:
    metrics = item["metrics"]
    lower = metrics["average_r_session_bootstrap_95"]["lower"]
    return (
        not item["selection_viable"],
        -(lower if lower is not None else -1_000.0),
        -(metrics["average_r_multiple"] if metrics["average_r_multiple"] is not None else -1_000.0),
        -min(metrics["buy_count"], metrics["sell_count"]),
        item["policy"]["minimum_lower_utility"],
    )


def policy_contract(policy: UtilityPolicy) -> dict[str, float]:
    return {
        "minimum_lower_utility": policy.minimum_lower_utility,
        "minimum_direction_margin": policy.minimum_direction_margin,
    }


def _rank_correlation(actual: np.ndarray, predicted: np.ndarray) -> float:
    if np.std(actual) <= 1e-12 or np.std(predicted) <= 1e-12:
        return 0.0
    actual_rank = _ranks(actual)
    predicted_rank = _ranks(predicted)
    return float(np.corrcoef(actual_rank, predicted_rank)[0, 1])


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks
