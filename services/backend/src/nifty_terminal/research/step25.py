"""Step 25: compact price-action-first feature-family audit.

This audit chooses a feature architecture, not a trading policy.  It uses only
early chronological folds for selection and one later fold for confirmation.
The already-observed final historical folds are not reused to tune another
backtest.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nifty_terminal.context.features import ContextFeatureBuild
from nifty_terminal.ml.definitions import RANDOM_SEED
from nifty_terminal.ml.models import WalkForwardConfig
from nifty_terminal.ml.split import PurgedWalkForwardSplitter
from nifty_terminal.research.step18b import binary_metrics, build_trade_paths
from nifty_terminal.research.step20 import _session_balanced_weights
from nifty_terminal.research.step23 import conditional_labels


STEP25_VERSION = "compact_price_action_feature_audit.v1"
WALK_FORWARD_CONFIG = WalkForwardConfig(
    n_splits=7,
    minimum_train_samples=10_000,
    test_samples=2_000,
    purge_bars=12,
    embargo_bars=12,
    minimum_train_class_samples=25,
)
SELECTION_FOLDS = (0, 1, 2)
CONFIRMATION_FOLD = 3
FAMILY_NAMES = (
    "PRICE_ACTION_12",
    "STRUCTURE_LEVELS_COMPACT",
    "STRUCTURE_PLUS_CROSS_MARKET",
    "STRUCTURE_PLUS_HIGHER_TIMEFRAME",
    "LEGACY_STRUCTURE_45",
)
DIRECTION_GATE = {
    "minimum_pooled_auc": 0.51,
    "minimum_pooled_brier_skill": 0.0,
    "minimum_folds_above_random_auc": 2,
    "minimum_worst_fold_auc": 0.48,
}
STRUCTURE_LEVEL_FEATURES = (
    "research_v3__adx_14",
    "research_v3__di_spread",
    "research_v3__slope_10_atr",
    "research_v3__support_distance_20_atr",
    "research_v3__resistance_distance_20_atr",
    "research_v3__session_return_atr",
    "research_v3__opening_range_ready",
    "research_v3__opening_range_position",
    "research_v3__previous_high_distance_atr",
    "research_v3__previous_low_distance_atr",
    "research_v3__previous_close_distance_atr",
    "research_v3__three_bar_return_atr",
    "research_v3__consecutive_direction",
)
HIGHER_TIMEFRAME_SUFFIXES = (
    "__distance_ema20_atr",
    "__trend_ema20_above_ema50",
)

RESEARCH_IDENTITY = hashlib.sha256(
    json.dumps(
        {
            "version": STEP25_VERSION,
            "walk_forward": WALK_FORWARD_CONFIG.to_contract(),
            "families": FAMILY_NAMES,
            "structure_level_features": STRUCTURE_LEVEL_FEATURES,
            "higher_timeframe_suffixes": HIGHER_TIMEFRAME_SUFFIXES,
            "direction_gate": DIRECTION_GATE,
            "model": "logistic_l2_c0p02",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
).hexdigest()


def run_compact_feature_audit(
    *, context: ContextFeatureBuild, minute_candles, context_bundle_sha256: str
) -> dict[str, object]:
    samples, matrix, long_paths, short_paths, exclusions = build_trade_paths(
        dataset=context.dataset,
        features=context.matrix,
        minute_candles=minute_candles,
    )
    if len(samples) < 24_000:
        raise ValueError("Step 25 requires at least 24,000 complete trade paths")
    folds = PurgedWalkForwardSplitter().split(samples, WALK_FORWARD_CONFIG)
    labels = conditional_labels(samples, long_paths, short_paths)
    names = context.matrix.feature_names
    comparisons = []
    for family in FAMILY_NAMES:
        indices = feature_family_indices(family, names)
        reports = []
        probabilities = []
        actuals = []
        priors = []
        for fold_index in SELECTION_FOLDS:
            fold = folds[fold_index]
            result = _fit_direction(
                samples=samples,
                matrix=matrix,
                labels=labels,
                feature_indices=indices,
                train_indices=np.asarray(fold.train_indices),
                test_indices=np.asarray(fold.test_indices),
            )
            metrics = binary_metrics(result["actual"], result["probability"])
            prior_metrics = binary_metrics(result["actual"], result["prior"])
            reports.append(
                {
                    "fold_index": fold_index,
                    "metrics": metrics,
                    "brier_skill_vs_prior": _brier_skill(metrics, prior_metrics),
                }
            )
            probabilities.append(result["probability"])
            actuals.append(result["actual"])
            priors.append(result["prior"])
        pooled_metrics = binary_metrics(
            np.concatenate(actuals), np.concatenate(probabilities)
        )
        pooled_prior = binary_metrics(
            np.concatenate(actuals), np.concatenate(priors)
        )
        pooled_skill = _brier_skill(pooled_metrics, pooled_prior)
        blockers = _blockers(pooled_metrics, pooled_skill, reports)
        comparisons.append(
            {
                "family": family,
                "feature_count": len(indices),
                "features": [names[index] for index in indices],
                "selection_metrics": pooled_metrics,
                "selection_brier_skill_vs_prior": pooled_skill,
                "folds": reports,
                "selection_viable": not blockers,
                "selection_blockers": blockers,
            }
        )
    viable = [item for item in comparisons if item["selection_viable"]]
    selected = min(viable or comparisons, key=_rank)
    selected_indices = feature_family_indices(str(selected["family"]), names)
    confirmation = _fit_direction(
        samples=samples,
        matrix=matrix,
        labels=labels,
        feature_indices=selected_indices,
        train_indices=np.asarray(folds[CONFIRMATION_FOLD].train_indices),
        test_indices=np.asarray(folds[CONFIRMATION_FOLD].test_indices),
    )
    confirmation_metrics = binary_metrics(
        confirmation["actual"], confirmation["probability"]
    )
    confirmation_prior = binary_metrics(confirmation["actual"], confirmation["prior"])
    confirmation_skill = _brier_skill(confirmation_metrics, confirmation_prior)
    confirmation_blockers = []
    if confirmation_metrics["roc_auc"] is None or confirmation_metrics["roc_auc"] <= 0.50:
        confirmation_blockers.append("CONFIRMATION_DIRECTION_AUC_NOT_ABOVE_RANDOM")
    if confirmation_skill <= 0:
        confirmation_blockers.append("CONFIRMATION_DIRECTION_BRIER_SKILL_NOT_POSITIVE")
    blockers = sorted(
        set(
            ([] if selected["selection_viable"] else ["NO_FEATURE_FAMILY_PASSED_SELECTION"])
            + confirmation_blockers
            + [
                "HISTORICAL_PERIOD_USED_FOR_MODEL_DEVELOPMENT",
                "DERIVATIVES_FORWARD_DATA_NOT_READY",
                "FORWARD_CONFIRMATION_NOT_COMPLETED",
            ]
        )
    )
    return {
        "schema_version": 1,
        "step25_version": STEP25_VERSION,
        "research_identity": RESEARCH_IDENTITY,
        "dataset_id": context.dataset.dataset_id,
        "context_bundle_sha256": context_bundle_sha256,
        "purpose": "select the smallest stable price-action-first feature family before derivatives augmentation",
        "dataset": {
            "complete_trade_paths": len(samples),
            "direction_training_rows": int(np.sum(labels.opportunity)),
            "excluded_trade_paths": exclusions,
        },
        "chronology": {
            "walk_forward": WALK_FORWARD_CONFIG.to_contract(),
            "selection_folds": list(SELECTION_FOLDS),
            "confirmation_fold": CONFIRMATION_FOLD,
            "final_historical_folds_reused_for_tuning": False,
            "future_labels_used_in_features": False,
        },
        "model": {
            "type": "logistic_regression",
            "regularization": "L2",
            "c": 0.02,
            "session_balanced_training_weights": True,
        },
        "feature_family_comparison": comparisons,
        "selected_family": {
            "family": selected["family"],
            "feature_count": selected["feature_count"],
            "features": selected["features"],
            "selection_viable": selected["selection_viable"],
            "selection_blockers": selected["selection_blockers"],
        },
        "confirmation": {
            "fold_index": CONFIRMATION_FOLD,
            "metrics": confirmation_metrics,
            "brier_skill_vs_prior": confirmation_skill,
            "blockers": confirmation_blockers,
            "passed": not confirmation_blockers,
        },
        "research_gate": {
            "direction_gate": DIRECTION_GATE,
            "blockers": blockers,
            "passed": False,
        },
        "model_artifact_created": False,
        "approved_for_live_inference": False,
        "official_signal_available": False,
        "automatic_trading_enabled": False,
    }


def feature_family_indices(family: str, names: tuple[str, ...]) -> np.ndarray:
    price_action = {name for name in names if name.startswith("price_action__")}
    structure = price_action | set(STRUCTURE_LEVEL_FEATURES)
    if family == "PRICE_ACTION_12":
        selected = price_action
    elif family == "STRUCTURE_LEVELS_COMPACT":
        selected = structure
    elif family == "STRUCTURE_PLUS_CROSS_MARKET":
        selected = structure | {name for name in names if name.startswith("cross__")}
    elif family == "STRUCTURE_PLUS_HIGHER_TIMEFRAME":
        selected = (
            structure
            | {name for name in names if name.startswith("cross__")}
            | {
                name
                for name in names
                if name.startswith(("context_15m__", "context_1h__"))
                and name.endswith(HIGHER_TIMEFRAME_SUFFIXES)
            }
        )
    elif family == "LEGACY_STRUCTURE_45":
        selected = {
            name
            for name in names
            if name.startswith(("price_action__", "research_v3__", "cross__"))
        }
    else:
        raise ValueError(f"Unknown Step 25 feature family: {family}")
    missing = set(STRUCTURE_LEVEL_FEATURES) - set(names) if family != "PRICE_ACTION_12" else set()
    if missing:
        raise ValueError("Required compact structure features are missing: " + ", ".join(sorted(missing)))
    indices = np.asarray([index for index, name in enumerate(names) if name in selected], dtype=int)
    if len(indices) == 0:
        raise ValueError(f"Step 25 family selected no features: {family}")
    return indices


def _fit_direction(
    *, samples, matrix, labels, feature_indices, train_indices, test_indices
):
    train_indices = train_indices[labels.opportunity[train_indices] == 1]
    test_indices = test_indices[labels.opportunity[test_indices] == 1]
    weights = _session_balanced_weights(samples, train_indices)
    model = Pipeline(
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
    model.fit(
        matrix[train_indices][:, feature_indices],
        labels.long_when_opportunity[train_indices],
        model__sample_weight=weights,
    )
    probability = model.predict_proba(matrix[test_indices][:, feature_indices])[:, 1]
    prior = float(
        np.average(labels.long_when_opportunity[train_indices], weights=weights)
    )
    return {
        "actual": labels.long_when_opportunity[test_indices],
        "probability": np.asarray(probability, dtype=float),
        "prior": np.full(len(test_indices), prior),
    }


def _blockers(metrics, skill, folds):
    blockers = []
    if metrics["roc_auc"] is None or metrics["roc_auc"] <= DIRECTION_GATE["minimum_pooled_auc"]:
        blockers.append("POOLED_DIRECTION_AUC_GATE_FAILED")
    if skill <= DIRECTION_GATE["minimum_pooled_brier_skill"]:
        blockers.append("POOLED_DIRECTION_BRIER_SKILL_GATE_FAILED")
    fold_aucs = [float(item["metrics"]["roc_auc"] or 0.0) for item in folds]
    if sum(value > 0.5 for value in fold_aucs) < DIRECTION_GATE["minimum_folds_above_random_auc"]:
        blockers.append("INSUFFICIENT_FOLDS_ABOVE_RANDOM_DIRECTION_AUC")
    if min(fold_aucs) < DIRECTION_GATE["minimum_worst_fold_auc"]:
        blockers.append("DIRECTION_FOLD_INSTABILITY")
    return blockers


def _rank(item):
    fold_aucs = [float(fold["metrics"]["roc_auc"] or 0.0) for fold in item["folds"]]
    return (
        not item["selection_viable"],
        -min(fold_aucs),
        -float(item["selection_brier_skill_vs_prior"]),
        int(item["feature_count"]),
        str(item["family"]),
    )


def _brier_skill(metrics, prior_metrics):
    return 1.0 - float(metrics["brier"]) / float(prior_metrics["brier"])
