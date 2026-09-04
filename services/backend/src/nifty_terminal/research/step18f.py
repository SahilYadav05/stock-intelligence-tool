"""Step 18F baseline-controlled, direction-specific utility research.

Step 18E demonstrated why a positive replay is not enough: a broad directional
period effect can look profitable even when the model's rank ordering reverses
between chronological folds.  This module models *incremental* utility above a
causal time/regime baseline and requires each enabled direction to be stable on
every selection fold.  Economic validation also compares the frozen policy with
WAIT, unconditional-direction, technical-trend, and causal time/regime policies.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

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
from nifty_terminal.research.step18d import _rank_correlation, regression_metrics
from nifty_terminal.research.step18e import (
    architecture_feature_indices,
    causal_score_percentiles,
)
from nifty_terminal.research.v2 import NESTED_WALK_FORWARD_CONFIG


STEP18F_VERSION = "baseline_controlled_directional_research.v1"
MODEL_SELECTION_FOLDS = (0, 1)
SCORE_REFERENCE_FOLD = 2
POLICY_SELECTION_FOLD = 3
HISTORICAL_DIAGNOSTIC_FOLD = 4
ARCHITECTURES = ("NIFTY_ONLY", "NIFTY_BANK", "NIFTY_VIX", "ALL_CONTEXT")
MODEL_CANDIDATES = (
    "residual_ridge_compact",
    "residual_hgb_squared_compact",
    "residual_hgb_absolute_compact",
    "residual_extra_trees_compact",
    "residual_robust_ensemble_compact",
)
SCORE_THRESHOLDS = (0.70, 0.75, 0.80, 0.85, 0.90)
DIRECTION_PERCENTILE_MARGIN = 0.05
TOP_COHORT_FRACTION = 0.20
MODEL_GATE = {
    "minimum_fold_mse_skill_vs_regime_baseline": 0.0,
    "minimum_fold_incremental_rank_correlation": 0.0,
    "minimum_pooled_incremental_rank_correlation": 0.02,
    "minimum_fold_top_cohort_excess_r": 0.0,
    "minimum_prediction_std_r": 0.01,
}
POLICY_GATE = {
    "minimum_trades": 100,
    "minimum_trades_per_enabled_direction": 50,
    "minimum_sessions": 25,
    "maximum_session_trade_share": 0.20,
    "minimum_profit_factor": 1.05,
    "minimum_raw_average_r_lower_95": 0.0,
    "minimum_excess_average_r_lower_95": 0.0,
    "minimum_daily_r_uplift_lower_95_vs_each_baseline": 0.0,
    "maximum_drawdown_r": 20.0,
}
BENCHMARKS = (
    "WAIT",
    "ALWAYS_LONG",
    "ALWAYS_SHORT",
    "TECHNICAL_TREND",
    "CAUSAL_TIME_REGIME",
)
RESEARCH_IDENTITY = hashlib.sha256(json.dumps(
    {
        "version": STEP18F_VERSION,
        "architectures": ARCHITECTURES,
        "models": MODEL_CANDIDATES,
        "score_thresholds": SCORE_THRESHOLDS,
        "direction_margin": DIRECTION_PERCENTILE_MARGIN,
        "top_cohort_fraction": TOP_COHORT_FRACTION,
        "model_gate": MODEL_GATE,
        "policy_gate": POLICY_GATE,
        "benchmarks": BENCHMARKS,
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
class BaselineState:
    global_mean: float
    time_effects: dict[int, float]
    regime_effects: dict[tuple[int, int], float]
    adx_threshold: float
    slope_index: int
    adx_index: int


@dataclass(frozen=True, slots=True)
class DirectionalPolicy:
    long_threshold: float | None
    short_threshold: float | None
    minimum_percentile_margin: float = DIRECTION_PERCENTILE_MARGIN


@dataclass(frozen=True, slots=True)
class _SelectedTrade:
    path: TradePath
    baseline_r: float


def run_baseline_controlled_research(
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
        raise ValueError("Step 18F requires at least 15,000 complete trade paths")
    folds = PurgedWalkForwardSplitter().split(samples, NESTED_WALK_FORWARD_CONFIG)
    names = context.matrix.feature_names
    base_count = int(context.diagnostics["base_feature_count"])
    architectures = {
        name: architecture_feature_indices(name, names, base_count)
        for name in ARCHITECTURES
    }
    targets = {
        "LONG": np.asarray([long_paths[item.sample_id].r_multiple for item in samples]),
        "SHORT": np.asarray([short_paths[item.sample_id].r_multiple for item in samples]),
    }

    comparisons: dict[str, list[dict[str, object]]] = {}
    selected: dict[str, dict[str, object]] = {}
    selected_outputs: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for direction, actual in targets.items():
        reports = []
        for architecture, architecture_indices in architectures.items():
            feature_indices = _compact_indices(names, architecture_indices)
            for candidate in MODEL_CANDIDATES:
                fold_reports = []
                for fold_index in MODEL_SELECTION_FOLDS:
                    fold = folds[fold_index]
                    output = _fit_predict_incremental(
                        samples=samples,
                        matrix=matrix,
                        names=names,
                        actual=actual,
                        train_indices=np.asarray(fold.train_indices),
                        test_indices=np.asarray(fold.test_indices),
                        feature_indices=feature_indices,
                        candidate=candidate,
                    )
                    fold_reports.append({
                        "fold_index": fold_index,
                        **_incremental_report(
                            actual=actual[np.asarray(fold.test_indices)],
                            total_prediction=output["total"],
                            incremental_prediction=output["incremental"],
                            baseline_prediction=output["baseline"],
                            train_mean_prediction=output["train_mean"],
                        ),
                    })
                pooled = _pool_incremental_reports(fold_reports)
                blockers = _model_selection_blockers(pooled, fold_reports)
                reports.append({
                    "architecture": architecture,
                    "candidate": candidate,
                    "feature_count": len(feature_indices),
                    "selection_metrics": pooled,
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
            "direction_enabled_for_policy_search": bool(chosen["selection_viable"]),
            "selection_viable": bool(chosen["selection_viable"]),
            "selection_blockers": chosen["selection_blockers"],
        }
        feature_indices = _compact_indices(
            names, architectures[str(chosen["architecture"])]
        )
        selected_outputs[direction] = {}
        for fold_index in (
            SCORE_REFERENCE_FOLD, POLICY_SELECTION_FOLD, HISTORICAL_DIAGNOSTIC_FOLD
        ):
            fold = folds[fold_index]
            selected_outputs[direction][fold_index] = _fit_predict_incremental(
                samples=samples,
                matrix=matrix,
                names=names,
                actual=actual,
                train_indices=np.asarray(fold.train_indices),
                test_indices=np.asarray(fold.test_indices),
                feature_indices=feature_indices,
                candidate=str(chosen["candidate"]),
            )

    percentiles: dict[str, dict[int, np.ndarray]] = {"LONG": {}, "SHORT": {}}
    for direction in ("LONG", "SHORT"):
        reference = selected_outputs[direction][SCORE_REFERENCE_FOLD]["incremental"]
        selection = selected_outputs[direction][POLICY_SELECTION_FOLD]["incremental"]
        diagnostic = selected_outputs[direction][HISTORICAL_DIAGNOSTIC_FOLD]["incremental"]
        percentiles[direction][POLICY_SELECTION_FOLD] = causal_score_percentiles(
            reference, selection
        )
        percentiles[direction][HISTORICAL_DIAGNOSTIC_FOLD] = causal_score_percentiles(
            np.concatenate((reference, selection)), diagnostic
        )

    stable = {
        direction: bool(selected[direction]["selection_viable"])
        for direction in ("LONG", "SHORT")
    }
    selection_indices = np.asarray(folds[POLICY_SELECTION_FOLD].test_indices)
    selection_samples = tuple(samples[index] for index in selection_indices)
    selection_benchmarks = benchmark_suite(
        samples=selection_samples,
        matrix=matrix[selection_indices],
        names=names,
        long_paths=long_paths,
        short_paths=short_paths,
        long_baseline=selected_outputs["LONG"][POLICY_SELECTION_FOLD]["baseline"],
        short_baseline=selected_outputs["SHORT"][POLICY_SELECTION_FOLD]["baseline"],
    )
    policy_reports = []
    options = (None,) + SCORE_THRESHOLDS
    for long_threshold in options:
        for short_threshold in options:
            if long_threshold is None and short_threshold is None:
                continue
            policy = DirectionalPolicy(long_threshold, short_threshold)
            metrics = simulate_directional_policy(
                samples=selection_samples,
                long_percentiles=percentiles["LONG"][POLICY_SELECTION_FOLD],
                short_percentiles=percentiles["SHORT"][POLICY_SELECTION_FOLD],
                long_baseline=selected_outputs["LONG"][POLICY_SELECTION_FOLD]["baseline"],
                short_baseline=selected_outputs["SHORT"][POLICY_SELECTION_FOLD]["baseline"],
                long_paths=long_paths,
                short_paths=short_paths,
                policy=policy,
                benchmarks=selection_benchmarks,
            )
            blockers = _policy_blockers(metrics, policy, stable)
            policy_reports.append({
                "policy": policy_contract(policy),
                "metrics": metrics,
                "selection_viable": not blockers,
                "selection_blockers": blockers,
            })
    viable_policies = [item for item in policy_reports if item["selection_viable"]]
    exploratory = min(policy_reports, key=_policy_rank) if policy_reports else None
    if viable_policies:
        chosen_policy_report = min(viable_policies, key=_policy_rank)
        chosen_policy = DirectionalPolicy(**chosen_policy_report["policy"])
    else:
        chosen_policy = DirectionalPolicy(None, None)
        chosen_policy_report = {
            "policy": policy_contract(chosen_policy),
            "metrics": _empty_replay(len(selection_samples), chosen_policy),
            "selection_viable": False,
            "selection_blockers": ["NO_HISTORICALLY_VIABLE_DIRECTIONAL_POLICY"],
        }

    diagnostic_indices = np.asarray(folds[HISTORICAL_DIAGNOSTIC_FOLD].test_indices)
    diagnostic_samples = tuple(samples[index] for index in diagnostic_indices)
    diagnostic_benchmarks = benchmark_suite(
        samples=diagnostic_samples,
        matrix=matrix[diagnostic_indices],
        names=names,
        long_paths=long_paths,
        short_paths=short_paths,
        long_baseline=selected_outputs["LONG"][HISTORICAL_DIAGNOSTIC_FOLD]["baseline"],
        short_baseline=selected_outputs["SHORT"][HISTORICAL_DIAGNOSTIC_FOLD]["baseline"],
    )
    replay = simulate_directional_policy(
        samples=diagnostic_samples,
        long_percentiles=percentiles["LONG"][HISTORICAL_DIAGNOSTIC_FOLD],
        short_percentiles=percentiles["SHORT"][HISTORICAL_DIAGNOSTIC_FOLD],
        long_baseline=selected_outputs["LONG"][HISTORICAL_DIAGNOSTIC_FOLD]["baseline"],
        short_baseline=selected_outputs["SHORT"][HISTORICAL_DIAGNOSTIC_FOLD]["baseline"],
        long_paths=long_paths,
        short_paths=short_paths,
        policy=chosen_policy,
        benchmarks=diagnostic_benchmarks,
    )

    diagnostics = {}
    model_blockers = []
    for direction, actual in targets.items():
        test = diagnostic_indices
        output = selected_outputs[direction][HISTORICAL_DIAGNOSTIC_FOLD]
        report = _incremental_report(
            actual=actual[test],
            total_prediction=output["total"],
            incremental_prediction=output["incremental"],
            baseline_prediction=output["baseline"],
            train_mean_prediction=output["train_mean"],
        )
        interval = session_bootstrap_incremental_skill(
            samples=diagnostic_samples,
            actual=actual[test],
            total_prediction=output["total"],
            baseline_prediction=output["baseline"],
        )
        blockers = []
        if interval["lower"] <= 0:
            blockers.append("INCREMENTAL_MSE_SKILL_LOWER_95_NOT_POSITIVE")
        if report["incremental_rank_correlation"] <= 0.02:
            blockers.append("INCREMENTAL_RANK_CORRELATION_TOO_LOW")
        if not selected[direction]["selection_viable"]:
            blockers.append("DIRECTION_FAILED_MODEL_SELECTION")
        diagnostics[direction] = {
            "selected": selected[direction],
            **report,
            "session_block_bootstrap_incremental_mse_skill_95": interval,
            "gate_passed": not blockers,
            "blockers": blockers,
        }
        model_blockers.extend(f"{direction}_{item}" for item in blockers)

    replay_blockers = _policy_blockers(replay, chosen_policy, stable)
    selection_blockers = [] if viable_policies else ["NO_POLICY_PASSED_SELECTION_GATE"]
    historical_blockers = sorted(set(model_blockers + replay_blockers + selection_blockers))
    return {
        "schema_version": 1,
        "step18f_version": STEP18F_VERSION,
        "research_identity": RESEARCH_IDENTITY,
        "dataset_id": context.dataset.dataset_id,
        "context_bundle_sha256": context_bundle_sha256,
        "objective": {
            "model_output": "incremental realized R above a causal time/regime baseline",
            "policy_score": "causal percentile of incremental utility",
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
            "future_labels_used_in_scores": False,
            "diagnostic_thresholds_locked_before_diagnostic": True,
            "final_historical_fold_previously_seen": True,
        },
        "baseline_control": {
            "type": "train-only shrunk time-of-day plus trend/ADX regime mean",
            "target_residualized_before_model_fit": True,
            "benchmarks": list(BENCHMARKS),
            "same_execution_and_no_overlap_for_all_benchmarks": True,
            "selection_benchmarks": selection_benchmarks,
            "diagnostic_benchmarks": diagnostic_benchmarks,
        },
        "model_comparison": comparisons,
        "selected_models": selected,
        "utility_diagnostics": diagnostics,
        "policy_selection": {
            "candidate_count": len(policy_reports),
            "passing_candidate_count": len(viable_policies),
            "selected": chosen_policy_report,
            "best_exploratory_rejected": exploratory,
        },
        "historical_simulated_live_replay": replay,
        "research_gate": {
            "model_gate": MODEL_GATE,
            "policy_gate": POLICY_GATE,
            "passed_before_mandatory_forward_blockers": not historical_blockers,
            "historical_blockers": historical_blockers,
            "blockers": sorted(set(historical_blockers + [
                "HISTORICAL_PERIOD_USED_FOR_MODEL_DEVELOPMENT",
                "FORWARD_CONFIRMATION_NOT_COMPLETED",
            ])),
        },
        "uncertainty_controls": {
            "directional_period_bias_subtracted": True,
            "fold_sign_reversal_is_a_hard_failure": True,
            "selected_policy_must_beat_every_locked_baseline": True,
            "session_dependence_preserved_by_block_bootstrap": True,
            "previously_seen_history_is_not_called_out_of_sample": True,
            "uncertainty_eliminated": False,
            "reason_uncertainty_remains": "market outcomes are stochastic; only future evidence can reduce uncertainty",
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


def fit_causal_baseline(
    *, samples, matrix: np.ndarray, actual: np.ndarray, train_indices: np.ndarray,
    names: tuple[str, ...], shrinkage: float = 80.0,
) -> BaselineState:
    slope_index = _required_feature(names, "research_v3__slope_10_atr")
    adx_index = _required_feature(names, "research_v3__adx_14")
    indices = np.asarray(train_indices, dtype=int)
    global_mean = float(np.mean(actual[indices]))
    adx_threshold = float(np.median(matrix[indices, adx_index]))
    time_groups: dict[int, list[float]] = defaultdict(list)
    regime_groups: dict[tuple[int, int], list[float]] = defaultdict(list)
    for index in indices:
        time_groups[_time_bucket(samples[int(index)].decision_time)].append(float(actual[index]))
        regime_groups[_regime_key(matrix[index], slope_index, adx_index, adx_threshold)].append(
            float(actual[index])
        )
    return BaselineState(
        global_mean=global_mean,
        time_effects={
            key: _shrunk_effect(values, global_mean, shrinkage)
            for key, values in time_groups.items()
        },
        regime_effects={
            key: _shrunk_effect(values, global_mean, shrinkage * 1.5)
            for key, values in regime_groups.items()
        },
        adx_threshold=adx_threshold,
        slope_index=slope_index,
        adx_index=adx_index,
    )


def predict_causal_baseline(
    state: BaselineState, *, samples, matrix: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    values = []
    for index in np.asarray(indices, dtype=int):
        time_effect = state.time_effects.get(
            _time_bucket(samples[int(index)].decision_time), 0.0
        )
        regime_effect = state.regime_effects.get(
            _regime_key(
                matrix[index], state.slope_index, state.adx_index, state.adx_threshold
            ),
            0.0,
        )
        values.append(np.clip(state.global_mean + time_effect + regime_effect, -2.0, 2.0))
    return np.asarray(values, dtype=float)


def simulate_directional_policy(
    *, samples, long_percentiles, short_percentiles, long_baseline, short_baseline,
    long_paths, short_paths, policy: DirectionalPolicy, benchmarks,
) -> dict[str, object]:
    selected: list[_SelectedTrade] = []
    waits = Counter[str]()
    active_until: datetime | None = None
    for index, sample in enumerate(samples):
        if active_until is not None and sample.decision_time < active_until:
            waits["ACTIVE_POSITION"] += 1
            continue
        candidates = []
        if policy.long_threshold is not None and long_percentiles[index] >= policy.long_threshold:
            candidates.append((float(long_percentiles[index]), "LONG"))
        if policy.short_threshold is not None and short_percentiles[index] >= policy.short_threshold:
            candidates.append((float(short_percentiles[index]), "SHORT"))
        if not candidates:
            waits["NO_DIRECTION_ABOVE_THRESHOLD"] += 1
            continue
        candidates.sort(reverse=True)
        if len(candidates) == 2 and candidates[0][0] - candidates[1][0] < policy.minimum_percentile_margin:
            waits["DIRECTION_PERCENTILE_MARGIN_TOO_SMALL"] += 1
            continue
        direction = candidates[0][1]
        path = long_paths[sample.sample_id] if direction == "LONG" else short_paths[sample.sample_id]
        baseline = long_baseline[index] if direction == "LONG" else short_baseline[index]
        selected.append(_SelectedTrade(path=path, baseline_r=float(baseline)))
        active_until = path.exited_at
    return directional_replay(
        selected=tuple(selected), decisions=len(samples), waits=waits,
        policy=policy, benchmarks=benchmarks,
    )


def directional_replay(
    *, selected: tuple[_SelectedTrade, ...], decisions: int, waits: Counter[str],
    policy: DirectionalPolicy, benchmarks,
) -> dict[str, object]:
    paths = tuple(item.path for item in selected)
    raw = np.asarray([item.path.r_multiple for item in selected], dtype=float)
    excess = np.asarray(
        [item.path.r_multiple - item.baseline_r for item in selected], dtype=float
    )
    points = [item.net_points for item in paths]
    gains = [item for item in points if item > 0]
    losses = [item for item in points if item < 0]
    cumulative = peak = drawdown = 0.0
    for value in raw:
        cumulative += float(value)
        peak = max(peak, cumulative)
        drawdown = max(drawdown, peak - cumulative)
    session_counts = Counter(_session_key(item.decision_time) for item in paths)
    daily_r = _daily_totals(paths)
    benchmark_uplift = {
        name: daily_uplift_bootstrap(daily_r, report["daily_total_r"])
        for name, report in benchmarks.items()
    }
    direction_metrics = {}
    for direction in ("LONG", "SHORT"):
        subset = tuple(item for item in selected if item.path.direction == direction)
        direction_metrics[direction] = _direction_summary(subset)
    return {
        "policy": policy_contract(policy),
        "evaluation_decisions": decisions,
        "trade_count": len(paths),
        "buy_count": sum(item.direction == "LONG" for item in paths),
        "sell_count": sum(item.direction == "SHORT" for item in paths),
        "coverage": len(paths) / decisions if decisions else 0.0,
        "target_hit_count": sum(item.exit_reason == "TARGET" for item in paths),
        "stop_hit_count": sum(item.exit_reason == "STOP" for item in paths),
        "expired_count": sum(item.exit_reason == "HORIZON" for item in paths),
        "win_rate": len(gains) / len(paths) if paths else None,
        "net_points": float(sum(points)),
        "average_points": float(np.mean(points)) if points else None,
        "average_r_multiple": float(np.mean(raw)) if len(raw) else None,
        "average_excess_r_vs_causal_baseline": float(np.mean(excess)) if len(excess) else None,
        "average_r_session_bootstrap_95": session_bootstrap_values(paths, raw),
        "excess_r_session_bootstrap_95": session_bootstrap_values(paths, excess),
        "profit_factor": sum(gains) / abs(sum(losses)) if gains and losses else None,
        "maximum_drawdown_r": drawdown,
        "session_count": len(session_counts),
        "maximum_session_trade_share": (
            max(session_counts.values()) / len(paths) if paths else None
        ),
        "direction_metrics": direction_metrics,
        "benchmark_daily_r_uplift_95": benchmark_uplift,
        "wait_counts": dict(waits),
        "hypothetical_index_points_only": True,
        "rupee_pnl_available": False,
    }


def benchmark_suite(
    *, samples, matrix, names, long_paths, short_paths, long_baseline, short_baseline,
) -> dict[str, dict[str, object]]:
    slope_index = _required_feature(names, "research_v3__slope_10_atr")
    adx_index = _required_feature(names, "research_v3__adx_14")
    reports = {}
    reports["WAIT"] = _benchmark_report((), len(samples))
    for name, chooser in (
        ("ALWAYS_LONG", lambda index: "LONG"),
        ("ALWAYS_SHORT", lambda index: "SHORT"),
        (
            "TECHNICAL_TREND",
            lambda index: (
                "LONG" if matrix[index, slope_index] > 0.10 and matrix[index, adx_index] >= 0.20
                else "SHORT" if matrix[index, slope_index] < -0.10 and matrix[index, adx_index] >= 0.20
                else None
            ),
        ),
        (
            "CAUSAL_TIME_REGIME",
            lambda index: (
                "LONG" if long_baseline[index] > max(0.0, short_baseline[index])
                else "SHORT" if short_baseline[index] > max(0.0, long_baseline[index])
                else None
            ),
        ),
    ):
        paths = _simulate_benchmark_paths(samples, long_paths, short_paths, chooser)
        reports[name] = _benchmark_report(paths, len(samples))
    return reports


def session_bootstrap_values(
    paths: tuple[TradePath, ...], values: np.ndarray, iterations: int = 2_000
) -> dict[str, float | int | None]:
    groups: dict[str, list[float]] = defaultdict(list)
    for path, value in zip(paths, np.asarray(values, dtype=float)):
        groups[_session_key(path.decision_time)].append(float(value))
    sessions = tuple(sorted(groups))
    if len(sessions) < 2:
        return {"lower": None, "median": None, "upper": None, "session_count": len(sessions)}
    rng = np.random.default_rng(RANDOM_SEED)
    draws = []
    for _ in range(iterations):
        chosen = rng.choice(sessions, len(sessions), replace=True)
        cohort = [value for session in chosen for value in groups[str(session)]]
        draws.append(float(np.mean(cohort)))
    lower, median, upper = np.quantile(draws, (0.025, 0.5, 0.975))
    return {
        "lower": float(lower), "median": float(median), "upper": float(upper),
        "session_count": len(sessions),
    }


def session_bootstrap_incremental_skill(
    *, samples, actual, total_prediction, baseline_prediction, iterations: int = 2_000,
) -> dict[str, float | int]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[_session_key(sample.decision_time)].append(index)
    sessions = tuple(sorted(groups))
    if len(sessions) < 2:
        return {"lower": -1.0, "median": -1.0, "upper": -1.0, "session_count": len(sessions)}
    actual = np.asarray(actual, dtype=float)
    total_prediction = np.asarray(total_prediction, dtype=float)
    baseline_prediction = np.asarray(baseline_prediction, dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    draws = []
    for _ in range(iterations):
        chosen = rng.choice(sessions, len(sessions), replace=True)
        indices = np.asarray([index for session in chosen for index in groups[str(session)]])
        model_mse = float(np.mean((actual[indices] - total_prediction[indices]) ** 2))
        base_mse = float(np.mean((actual[indices] - baseline_prediction[indices]) ** 2))
        draws.append(1.0 - model_mse / base_mse if base_mse else -1.0)
    lower, median, upper = np.quantile(draws, (0.025, 0.5, 0.975))
    return {
        "lower": float(lower), "median": float(median), "upper": float(upper),
        "session_count": len(sessions),
    }


def daily_uplift_bootstrap(
    model_daily: dict[str, float], baseline_daily: dict[str, float], iterations: int = 2_000
) -> dict[str, float | int | None]:
    sessions = tuple(sorted(set(model_daily) | set(baseline_daily)))
    if len(sessions) < 2:
        return {"lower": None, "median": None, "upper": None, "session_count": len(sessions)}
    differences = np.asarray([
        model_daily.get(session, 0.0) - baseline_daily.get(session, 0.0)
        for session in sessions
    ])
    rng = np.random.default_rng(RANDOM_SEED)
    draws = []
    for _ in range(iterations):
        chosen = rng.integers(0, len(differences), len(differences))
        draws.append(float(np.mean(differences[chosen])))
    lower, median, upper = np.quantile(draws, (0.025, 0.5, 0.975))
    return {
        "lower": float(lower), "median": float(median), "upper": float(upper),
        "session_count": len(sessions),
    }


def _fit_predict_incremental(
    *, samples, matrix, names, actual, train_indices, test_indices,
    feature_indices, candidate,
) -> dict[str, np.ndarray]:
    baseline_state = fit_causal_baseline(
        samples=samples, matrix=matrix, actual=actual,
        train_indices=train_indices, names=names,
    )
    train_baseline = predict_causal_baseline(
        baseline_state, samples=samples, matrix=matrix, indices=train_indices
    )
    test_baseline = predict_causal_baseline(
        baseline_state, samples=samples, matrix=matrix, indices=test_indices
    )
    residual_target = actual[train_indices] - train_baseline
    weights = _session_balanced_weights(samples, train_indices)
    incremental = _fit_candidate(
        candidate=candidate,
        train_x=matrix[train_indices][:, feature_indices],
        train_y=residual_target,
        test_x=matrix[test_indices][:, feature_indices],
        sample_weight=weights,
    )
    return {
        "baseline": test_baseline,
        "incremental": incremental,
        "total": test_baseline + incremental,
        "train_mean": np.full(len(test_indices), baseline_state.global_mean),
    }


def _fit_candidate(*, candidate, train_x, train_y, test_x, sample_weight) -> np.ndarray:
    names = (
        "residual_hgb_squared_compact",
        "residual_hgb_absolute_compact",
        "residual_extra_trees_compact",
    ) if candidate == "residual_robust_ensemble_compact" else (candidate,)
    predictions = []
    for name in names:
        model = _candidate(name)
        if isinstance(model, Pipeline):
            model.fit(train_x, train_y, model__sample_weight=sample_weight)
        else:
            model.fit(train_x, train_y, sample_weight=sample_weight)
        predictions.append(np.asarray(model.predict(test_x), dtype=float))
    return np.median(np.vstack(predictions), axis=0)


def _candidate(name: str):
    if name == "residual_ridge_compact":
        return Pipeline((
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=20.0)),
        ))
    if name == "residual_hgb_squared_compact":
        return HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.025, max_iter=220,
            max_leaf_nodes=7, min_samples_leaf=80, l2_regularization=10.0,
            random_state=RANDOM_SEED,
        )
    if name == "residual_hgb_absolute_compact":
        return HistGradientBoostingRegressor(
            loss="absolute_error", learning_rate=0.025, max_iter=220,
            max_leaf_nodes=7, min_samples_leaf=80, l2_regularization=10.0,
            random_state=RANDOM_SEED,
        )
    if name == "residual_extra_trees_compact":
        return ExtraTreesRegressor(
            n_estimators=320, max_depth=7, min_samples_leaf=45,
            max_features=0.45, n_jobs=-1, random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown Step 18F candidate: {name}")


def _incremental_report(
    *, actual, total_prediction, incremental_prediction, baseline_prediction,
    train_mean_prediction,
) -> dict[str, object]:
    actual = np.asarray(actual, dtype=float)
    total_prediction = np.asarray(total_prediction, dtype=float)
    incremental_prediction = np.asarray(incremental_prediction, dtype=float)
    baseline_prediction = np.asarray(baseline_prediction, dtype=float)
    train_mean_prediction = np.asarray(train_mean_prediction, dtype=float)
    residual_actual = actual - baseline_prediction
    model_metrics = regression_metrics(actual, total_prediction, baseline_prediction)
    baseline_vs_mean = regression_metrics(
        actual, baseline_prediction, train_mean_prediction
    )
    cutoff = float(np.quantile(incremental_prediction, 1.0 - TOP_COHORT_FRACTION))
    top = residual_actual[incremental_prediction >= cutoff]
    return {
        "model_vs_causal_baseline": model_metrics,
        "causal_baseline_vs_train_mean": baseline_vs_mean,
        "incremental_rank_correlation": _rank_correlation(
            residual_actual, incremental_prediction
        ),
        "incremental_prediction_std_r": float(np.std(incremental_prediction)),
        "top_cohort_fraction": TOP_COHORT_FRACTION,
        "top_cohort_support": int(len(top)),
        "top_cohort_average_excess_r": float(np.mean(top)) if len(top) else None,
    }


def _pool_incremental_reports(folds: list[dict[str, object]]) -> dict[str, float]:
    weights = np.asarray([
        item["model_vs_causal_baseline"]["sample_count"] for item in folds
    ], dtype=float)
    weights /= np.sum(weights)
    return {
        "mse_skill_vs_regime_baseline": float(sum(
            weight * item["model_vs_causal_baseline"]["mse_skill_vs_baseline"]
            for weight, item in zip(weights, folds)
        )),
        "incremental_rank_correlation": float(sum(
            weight * item["incremental_rank_correlation"]
            for weight, item in zip(weights, folds)
        )),
        "top_cohort_average_excess_r": float(sum(
            weight * item["top_cohort_average_excess_r"]
            for weight, item in zip(weights, folds)
        )),
        "incremental_prediction_std_r": float(sum(
            weight * item["incremental_prediction_std_r"]
            for weight, item in zip(weights, folds)
        )),
    }


def _model_selection_blockers(pooled, folds) -> list[str]:
    blockers = []
    if pooled["mse_skill_vs_regime_baseline"] <= 0:
        blockers.append("POOLED_MSE_SKILL_VS_REGIME_BASELINE_NOT_POSITIVE")
    if pooled["incremental_rank_correlation"] <= MODEL_GATE["minimum_pooled_incremental_rank_correlation"]:
        blockers.append("POOLED_INCREMENTAL_RANK_CORRELATION_TOO_LOW")
    if pooled["incremental_prediction_std_r"] < MODEL_GATE["minimum_prediction_std_r"]:
        blockers.append("PREDICTION_COLLAPSE")
    if any(
        item["model_vs_causal_baseline"]["mse_skill_vs_baseline"]
        <= MODEL_GATE["minimum_fold_mse_skill_vs_regime_baseline"]
        for item in folds
    ):
        blockers.append("NON_POSITIVE_MSE_SKILL_IN_SELECTION_FOLD")
    if any(
        item["incremental_rank_correlation"]
        <= MODEL_GATE["minimum_fold_incremental_rank_correlation"]
        for item in folds
    ):
        blockers.append("RANK_SIGN_REVERSAL_OR_ZERO_IN_SELECTION_FOLD")
    if any(
        item["top_cohort_average_excess_r"]
        <= MODEL_GATE["minimum_fold_top_cohort_excess_r"]
        for item in folds
    ):
        blockers.append("TOP_COHORT_EXCESS_NOT_POSITIVE_IN_SELECTION_FOLD")
    return blockers


def _policy_blockers(metrics, policy: DirectionalPolicy, stable) -> list[str]:
    blockers = []
    enabled = {
        "LONG": policy.long_threshold is not None,
        "SHORT": policy.short_threshold is not None,
    }
    if not any(enabled.values()):
        return ["NO_DIRECTION_ENABLED"]
    for direction in ("LONG", "SHORT"):
        if enabled[direction] and not stable[direction]:
            blockers.append(f"{direction}_MODEL_NOT_SELECTION_VIABLE")
        if not enabled[direction]:
            continue
        report = metrics["direction_metrics"][direction]
        if report["trade_count"] < POLICY_GATE["minimum_trades_per_enabled_direction"]:
            blockers.append(f"{direction}_SUPPORT_TOO_LOW")
        if report["profit_factor"] is None or report["profit_factor"] <= POLICY_GATE["minimum_profit_factor"]:
            blockers.append(f"{direction}_PROFIT_FACTOR_GATE_FAILED")
        if _lower(report["average_r_session_bootstrap_95"]) <= 0:
            blockers.append(f"{direction}_RAW_EXPECTANCY_LOWER_95_NOT_POSITIVE")
        if _lower(report["excess_r_session_bootstrap_95"]) <= 0:
            blockers.append(f"{direction}_EXCESS_EXPECTANCY_LOWER_95_NOT_POSITIVE")
    if metrics["trade_count"] < POLICY_GATE["minimum_trades"]:
        blockers.append("TRADE_SUPPORT_TOO_LOW")
    if metrics["session_count"] < POLICY_GATE["minimum_sessions"]:
        blockers.append("SESSION_SUPPORT_TOO_LOW")
    share = metrics["maximum_session_trade_share"]
    if share is None or share > POLICY_GATE["maximum_session_trade_share"]:
        blockers.append("SESSION_CONCENTRATION_TOO_HIGH")
    if metrics["profit_factor"] is None or metrics["profit_factor"] <= POLICY_GATE["minimum_profit_factor"]:
        blockers.append("PROFIT_FACTOR_GATE_FAILED")
    if _lower(metrics["average_r_session_bootstrap_95"]) <= POLICY_GATE["minimum_raw_average_r_lower_95"]:
        blockers.append("RAW_EXPECTANCY_LOWER_95_NOT_POSITIVE")
    if _lower(metrics["excess_r_session_bootstrap_95"]) <= POLICY_GATE["minimum_excess_average_r_lower_95"]:
        blockers.append("EXCESS_EXPECTANCY_LOWER_95_NOT_POSITIVE")
    if metrics["maximum_drawdown_r"] > POLICY_GATE["maximum_drawdown_r"]:
        blockers.append("MAXIMUM_DRAWDOWN_GATE_FAILED")
    for name, interval in metrics["benchmark_daily_r_uplift_95"].items():
        if _lower(interval) <= POLICY_GATE["minimum_daily_r_uplift_lower_95_vs_each_baseline"]:
            blockers.append(f"DAILY_R_UPLIFT_NOT_POSITIVE_VS_{name}")
    return sorted(set(blockers))


def _model_rank(item) -> tuple[object, ...]:
    metrics = item["selection_metrics"]
    return (
        not item["selection_viable"],
        -metrics["top_cohort_average_excess_r"],
        -metrics["incremental_rank_correlation"],
        -metrics["mse_skill_vs_regime_baseline"],
        ARCHITECTURES.index(str(item["architecture"])),
        str(item["candidate"]),
    )


def _policy_rank(item) -> tuple[object, ...]:
    metrics = item["metrics"]
    return (
        not item["selection_viable"],
        -_lower(metrics["average_r_session_bootstrap_95"]),
        -_lower(metrics["excess_r_session_bootstrap_95"]),
        -(metrics["average_r_multiple"] if metrics["average_r_multiple"] is not None else -1_000.0),
        -metrics["trade_count"],
    )


def policy_contract(policy: DirectionalPolicy) -> dict[str, float | None]:
    return {
        "long_threshold": policy.long_threshold,
        "short_threshold": policy.short_threshold,
        "minimum_percentile_margin": policy.minimum_percentile_margin,
    }


def _compact_indices(names: tuple[str, ...], architecture_indices: np.ndarray) -> np.ndarray:
    excluded = (
        "__doji", "__hammer", "__shooting_star", "__bullish_engulfing",
        "__bearish_engulfing", "__inside_bar", "__outside_bar",
    )
    return np.asarray([
        int(index) for index in architecture_indices
        if not any(fragment in names[int(index)] for fragment in excluded)
    ], dtype=int)


def _session_balanced_weights(samples, indices: np.ndarray) -> np.ndarray:
    counts = Counter(_session_key(samples[int(index)].decision_time) for index in indices)
    raw = np.asarray([
        1.0 / counts[_session_key(samples[int(index)].decision_time)] for index in indices
    ])
    return raw * len(raw) / np.sum(raw)


def _required_feature(names: tuple[str, ...], name: str) -> int:
    try:
        return names.index(name)
    except ValueError as error:
        raise ValueError(f"Step 18F requires feature: {name}") from error


def _time_bucket(value: datetime) -> int:
    local = value.astimezone(IST)
    minutes = local.hour * 60 + local.minute - (9 * 60 + 15)
    return max(0, minutes // 30)


def _regime_key(row, slope_index: int, adx_index: int, adx_threshold: float) -> tuple[int, int]:
    slope = float(row[slope_index])
    trend = 1 if slope > 0.10 else -1 if slope < -0.10 else 0
    strength = 1 if float(row[adx_index]) >= adx_threshold else 0
    return trend, strength


def _shrunk_effect(values: list[float], global_mean: float, shrinkage: float) -> float:
    weight = len(values) / (len(values) + shrinkage)
    return weight * (float(np.mean(values)) - global_mean)


def _simulate_benchmark_paths(samples, long_paths, short_paths, chooser) -> tuple[TradePath, ...]:
    paths = []
    active_until: datetime | None = None
    for index, sample in enumerate(samples):
        if active_until is not None and sample.decision_time < active_until:
            continue
        direction = chooser(index)
        if direction is None:
            continue
        path = long_paths[sample.sample_id] if direction == "LONG" else short_paths[sample.sample_id]
        paths.append(path)
        active_until = path.exited_at
    return tuple(paths)


def _benchmark_report(paths: tuple[TradePath, ...], decisions: int) -> dict[str, object]:
    values = [item.r_multiple for item in paths]
    return {
        "trade_count": len(paths),
        "buy_count": sum(item.direction == "LONG" for item in paths),
        "sell_count": sum(item.direction == "SHORT" for item in paths),
        "average_r": float(np.mean(values)) if values else None,
        "total_r": float(sum(values)),
        "coverage": len(paths) / decisions if decisions else 0.0,
        "daily_total_r": _daily_totals(paths),
    }


def _daily_totals(paths: tuple[TradePath, ...]) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for path in paths:
        result[_session_key(path.decision_time)] += float(path.r_multiple)
    return dict(result)


def _direction_summary(selected: tuple[_SelectedTrade, ...]) -> dict[str, object]:
    paths = tuple(item.path for item in selected)
    raw = np.asarray([item.path.r_multiple for item in selected], dtype=float)
    excess = np.asarray([item.path.r_multiple - item.baseline_r for item in selected])
    points = [item.net_points for item in paths]
    gains = [item for item in points if item > 0]
    losses = [item for item in points if item < 0]
    return {
        "trade_count": len(paths),
        "average_r": float(np.mean(raw)) if len(raw) else None,
        "average_excess_r": float(np.mean(excess)) if len(excess) else None,
        "profit_factor": sum(gains) / abs(sum(losses)) if gains and losses else None,
        "average_r_session_bootstrap_95": session_bootstrap_values(paths, raw),
        "excess_r_session_bootstrap_95": session_bootstrap_values(paths, excess),
    }


def _empty_replay(decisions: int, policy: DirectionalPolicy) -> dict[str, object]:
    empty_interval = {"lower": None, "median": None, "upper": None, "session_count": 0}
    return {
        "policy": policy_contract(policy), "evaluation_decisions": decisions,
        "trade_count": 0, "buy_count": 0, "sell_count": 0, "coverage": 0.0,
        "target_hit_count": 0, "stop_hit_count": 0, "expired_count": 0,
        "win_rate": None, "net_points": 0.0, "average_points": None,
        "average_r_multiple": None, "average_excess_r_vs_causal_baseline": None,
        "average_r_session_bootstrap_95": empty_interval,
        "excess_r_session_bootstrap_95": empty_interval,
        "profit_factor": None, "maximum_drawdown_r": 0.0, "session_count": 0,
        "maximum_session_trade_share": None,
        "direction_metrics": {
            "LONG": _direction_summary(()), "SHORT": _direction_summary(()),
        },
        "benchmark_daily_r_uplift_95": {}, "wait_counts": {"NO_DIRECTION_ENABLED": decisions},
        "hypothetical_index_points_only": True, "rupee_pnl_available": False,
    }


def _lower(interval) -> float:
    value = interval.get("lower") if interval else None
    return float(value) if value is not None else -1_000.0


def _session_key(value: datetime) -> str:
    return str(value.astimezone(IST).date())
