"""Step 18C controlled comparison: NIFTY-only versus canonical cross-market context."""

from __future__ import annotations

import hashlib
import json

from nifty_terminal.context.features import ContextFeatureBuild
from nifty_terminal.features.research_v3 import ResearchFeatureMatrix
from nifty_terminal.research.step18b import run_trade_aligned_research


STEP18C_VERSION = "cross_market_model_research.v1"
INCREMENTAL_GATE = {
    "minimum_auc_delta_each_direction": 0.0,
    "minimum_brier_skill_delta_each_direction": 0.0,
    "minimum_probability_brier_skill_lower_95": 0.0,
    "minimum_policy_trades": 100,
    "minimum_policy_buys": 30,
    "minimum_policy_sells": 30,
    "minimum_policy_average_r_lower_95": 0.0,
}
RESEARCH_IDENTITY = hashlib.sha256(json.dumps(
    {"version": STEP18C_VERSION, "gate": INCREMENTAL_GATE},
    sort_keys=True, separators=(",", ":")
).encode()).hexdigest()


def run_cross_market_research(
    *,
    context: ContextFeatureBuild,
    minute_candles,
    primary_candles,
    context_bundle_sha256: str,
) -> dict[str, object]:
    base_count = int(context.diagnostics["base_feature_count"])
    base_matrix = ResearchFeatureMatrix(
        feature_names=context.matrix.feature_names[:base_count],
        rows=tuple(row[:base_count] for row in context.matrix.rows),
        sample_ids=context.matrix.sample_ids,
    )
    base = run_trade_aligned_research(
        dataset=context.dataset,
        minute_candles=minute_candles,
        primary_candles=primary_candles,
        feature_matrix=base_matrix,
    )
    enhanced = run_trade_aligned_research(
        dataset=context.dataset,
        minute_candles=minute_candles,
        primary_candles=primary_candles,
        feature_matrix=context.matrix,
    )
    enhanced["feature_architecture"] = {
        **enhanced["feature_architecture"],
        "version": context.diagnostics["feature_version"],
        "feature_set_hash": context.diagnostics["feature_set_hash"],
        "feature_count": context.diagnostics["total_feature_count"],
        "bank_nifty_used": True,
        "india_vix_used": True,
        "exact_point_in_time_join": True,
    }
    enhanced["known_data_limitations"] = {
        **enhanced["known_data_limitations"],
        "cross_market_context_used": True,
        "india_vix_used": True,
        "reason": "Bank Nifty and India VIX are available; news, breadth and continuous futures/OI remain unavailable",
    }
    enhanced["research_gate"]["blockers"] = [
        item for item in enhanced["research_gate"]["blockers"]
        if item != "CROSS_MARKET_HISTORY_NOT_AVAILABLE"
    ]
    deltas = {}
    blockers = []
    for direction in ("LONG", "SHORT"):
        old = base["probability_diagnostics"][direction]
        new = enhanced["probability_diagnostics"][direction]
        auc_delta = _number(new["calibrated_metrics"]["roc_auc"]) - _number(old["calibrated_metrics"]["roc_auc"])
        skill_delta = float(new["brier_skill_vs_prior"]) - float(old["brier_skill_vs_prior"])
        brier_delta = float(new["calibrated_metrics"]["brier"]) - float(old["calibrated_metrics"]["brier"])
        deltas[direction] = {
            "roc_auc_delta": auc_delta,
            "brier_skill_vs_prior_delta": skill_delta,
            "brier_loss_delta_lower_is_better": brier_delta,
            "base": {
                "roc_auc": old["calibrated_metrics"]["roc_auc"],
                "brier_skill_vs_prior": old["brier_skill_vs_prior"],
            },
            "context": {
                "roc_auc": new["calibrated_metrics"]["roc_auc"],
                "brier_skill_vs_prior": new["brier_skill_vs_prior"],
                "brier_skill_lower_95": new["session_block_bootstrap_brier_skill_95"]["lower"],
                "gate_passed": new["gate_passed"],
                "blockers": new["blockers"],
            },
        }
        if auc_delta <= INCREMENTAL_GATE["minimum_auc_delta_each_direction"]:
            blockers.append(f"{direction}_AUC_DID_NOT_IMPROVE")
        if skill_delta <= INCREMENTAL_GATE["minimum_brier_skill_delta_each_direction"]:
            blockers.append(f"{direction}_BRIER_SKILL_DID_NOT_IMPROVE")
        if float(new["session_block_bootstrap_brier_skill_95"]["lower"]) <= 0:
            blockers.append(f"{direction}_BRIER_SKILL_LOWER_95_NOT_POSITIVE")
        if not new["gate_passed"]:
            blockers.append(f"{direction}_PROBABILITY_GATE_FAILED")
    replay = enhanced["historical_simulated_live_replay"]
    lower = replay["average_r_session_bootstrap_95"]["lower"]
    if replay["trade_count"] < INCREMENTAL_GATE["minimum_policy_trades"]:
        blockers.append("POLICY_TRADE_SUPPORT_TOO_LOW")
    if replay["buy_count"] < INCREMENTAL_GATE["minimum_policy_buys"]:
        blockers.append("POLICY_BUY_SUPPORT_TOO_LOW")
    if replay["sell_count"] < INCREMENTAL_GATE["minimum_policy_sells"]:
        blockers.append("POLICY_SELL_SUPPORT_TOO_LOW")
    if lower is None or lower <= INCREMENTAL_GATE["minimum_policy_average_r_lower_95"]:
        blockers.append("POLICY_EXPECTANCY_LOWER_95_NOT_POSITIVE")
    historical_gate_passed = not blockers
    blockers.extend((
        "HISTORICAL_PERIOD_USED_FOR_MODEL_DEVELOPMENT",
        "FORWARD_CONFIRMATION_NOT_COMPLETED",
    ))
    return {
        "schema_version": 1,
        "step18c_version": STEP18C_VERSION,
        "research_identity": RESEARCH_IDENTITY,
        "dataset_id": context.dataset.dataset_id,
        "context_bundle_sha256": context_bundle_sha256,
        "context_feature_build": context.diagnostics,
        "controlled_comparison": {
            "same_samples": True,
            "same_targets": True,
            "same_folds": True,
            "same_execution_and_costs": True,
            "only_difference": "canonical Bank Nifty and India VIX features",
            "directional_deltas": deltas,
            "base_nifty_only": _compact(base),
            "context_aware": _compact(enhanced),
        },
        "research_gate": {
            "definition": INCREMENTAL_GATE,
            "passed_before_mandatory_forward_and_missing_context_blockers": historical_gate_passed,
            "blockers": sorted(set(blockers)),
        },
        "data_scope": {
            "bank_nifty_used": True,
            "india_vix_used": True,
            "nifty_spot_volume_used": False,
            "developing_candles_used": False,
            "historical_news_used": False,
            "continuous_futures_oi_used": False,
            "constituent_breadth_used": False,
            "chart_image_used": False,
        },
        "known_limitations_not_silently_filled": [
            "HISTORICAL_NEWS_NOT_YET_AVAILABLE",
            "FUTURES_OI_CONTINUOUS_HISTORY_NOT_YET_AVAILABLE",
            "POINT_IN_TIME_CONSTITUENT_BREADTH_NOT_YET_AVAILABLE",
            "USDINR_AND_CRUDE_CONTINUOUS_POINT_IN_TIME_HISTORY_NOT_YET_AVAILABLE",
        ],
        "full_context_result": enhanced,
        "existing_shadow_runtime_modified": False,
        "model_artifact_created": False,
        "approved_for_live_inference": False,
        "precise_probability_display_allowed": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
    }


def _compact(report: dict[str, object]) -> dict[str, object]:
    return {
        "selected_candidates": report["selected_candidates"],
        "selected_calibrations": report["selected_calibrations"],
        "probability_diagnostics": report["probability_diagnostics"],
        "policy_selection": report["policy_selection"],
        "historical_simulated_live_replay": report["historical_simulated_live_replay"],
        "research_gate": report["research_gate"],
    }


def _number(value: object) -> float:
    return float(value) if value is not None else 0.0
