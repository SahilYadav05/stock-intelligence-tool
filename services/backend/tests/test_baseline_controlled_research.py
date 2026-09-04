from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest import TestCase

import numpy as np

from nifty_terminal.research.step18b import TradePath
from nifty_terminal.ml.models import TargetOutcome, TrainingSample
from nifty_terminal.research.step18f import (
    RESEARCH_IDENTITY,
    STEP18F_VERSION,
    DirectionalPolicy,
    _model_selection_blockers,
    daily_uplift_bootstrap,
    fit_causal_baseline,
    predict_causal_baseline,
    simulate_directional_policy,
)


class BaselineControlledResearchTests(TestCase):
    def test_causal_baseline_does_not_read_test_outcomes(self) -> None:
        samples = _samples(8)
        names = ("research_v3__slope_10_atr", "research_v3__adx_14")
        matrix = np.asarray([
            (-0.2, 0.1), (-0.1, 0.2), (0.1, 0.3), (0.2, 0.4),
            (0.3, 0.5), (-0.3, 0.2), (0.4, 0.6), (-0.4, 0.1),
        ])
        first = np.asarray([-1.0, -0.5, 0.5, 1.0, 999.0, 999.0, 999.0, 999.0])
        changed = first.copy()
        changed[4:] = -999.0
        train = np.arange(4)
        test = np.arange(4, 8)
        state_a = fit_causal_baseline(
            samples=samples, matrix=matrix, actual=first,
            train_indices=train, names=names,
        )
        state_b = fit_causal_baseline(
            samples=samples, matrix=matrix, actual=changed,
            train_indices=train, names=names,
        )
        predicted_a = predict_causal_baseline(
            state_a, samples=samples, matrix=matrix, indices=test
        )
        predicted_b = predict_causal_baseline(
            state_b, samples=samples, matrix=matrix, indices=test
        )
        np.testing.assert_allclose(predicted_a, predicted_b)

    def test_fold_sign_reversal_is_a_hard_model_failure(self) -> None:
        pooled = {
            "mse_skill_vs_regime_baseline": 0.02,
            "incremental_rank_correlation": 0.10,
            "top_cohort_average_excess_r": 0.10,
            "incremental_prediction_std_r": 0.10,
        }
        good = _fold_report(0.02, 0.12, 0.10)
        reversed_fold = _fold_report(0.02, -0.01, 0.10)
        blockers = _model_selection_blockers(pooled, [good, reversed_fold])
        self.assertIn("RANK_SIGN_REVERSAL_OR_ZERO_IN_SELECTION_FOLD", blockers)

    def test_disabled_direction_cannot_be_selected(self) -> None:
        samples = _samples(5)
        long_paths = {
            item.sample_id: _path(item.sample_id, item.decision_time, "LONG", 1.0)
            for item in samples
        }
        short_paths = {
            item.sample_id: _path(item.sample_id, item.decision_time, "SHORT", 1.0)
            for item in samples
        }
        empty_benchmarks = {
            "WAIT": {"daily_total_r": {}},
        }
        result = simulate_directional_policy(
            samples=samples,
            long_percentiles=np.asarray([0.95, 0.1, 0.95, 0.1, 0.95]),
            short_percentiles=np.ones(5),
            long_baseline=np.zeros(5),
            short_baseline=np.zeros(5),
            long_paths=long_paths,
            short_paths=short_paths,
            policy=DirectionalPolicy(0.90, None),
            benchmarks=empty_benchmarks,
        )
        self.assertGreater(result["buy_count"], 0)
        self.assertEqual(result["sell_count"], 0)

    def test_daily_uplift_bootstrap_is_session_blocked(self) -> None:
        model = {f"2026-01-{day:02d}": 2.0 for day in range(1, 11)}
        baseline = {f"2026-01-{day:02d}": 0.0 for day in range(1, 11)}
        interval = daily_uplift_bootstrap(model, baseline, iterations=200)
        self.assertGreater(interval["lower"], 0.0)
        self.assertEqual(interval["session_count"], 10)

    def test_contract_is_permanently_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[3]
        with (root / "contracts" / "baseline-controlled-research.v1.schema.json").open(
            "r", encoding="utf-8"
        ) as file:
            schema = json.load(file)
        self.assertEqual(STEP18F_VERSION, "baseline_controlled_directional_research.v1")
        self.assertEqual(len(RESEARCH_IDENTITY), 64)
        for name in (
            "model_artifact_created",
            "approved_for_live_inference",
            "precise_probability_display_allowed",
            "official_signal_available",
            "automatic_trading_enabled",
        ):
            self.assertFalse(schema["properties"][name]["const"])
        self.assertFalse(
            schema["properties"]["uncertainty_controls"]["properties"]
            ["uncertainty_eliminated"]["const"]
        )


def _samples(count):
    base = _sample()
    return tuple(
        replace(
            base,
            sample_id=f"sample-{index}",
            decision_time=base.decision_time + timedelta(minutes=65 * index),
        )
        for index in range(count)
    )


def _sample() -> TrainingSample:
    decision = datetime(2026, 8, 3, 4, 0, tzinfo=timezone.utc)
    return TrainingSample(
        sample_id="sample",
        dataset_id="dataset",
        instrument_id="NIFTY50_SPOT",
        decision_time=decision,
        label_window_end=decision + timedelta(minutes=60),
        label_id="label",
        outcome=TargetOutcome.UP,
        primary_candle_id="5m",
        context_15m_candle_id="15m",
        context_1h_candle_id="1h",
        input_revision_checksum="checksum",
        feature_names=("primary_5m__return_1",),
        feature_values=(0.0,),
    )


def _path(sample_id, decision_time, direction, r_multiple) -> TradePath:
    return TradePath(
        sample_id=sample_id,
        decision_time=decision_time,
        direction=direction,
        success=1,
        exit_reason="TARGET",
        entered_at=decision_time,
        exited_at=decision_time + timedelta(minutes=6),
        entry=24000.0,
        stop=23990.0,
        target=24010.0,
        exit=24010.0,
        net_points=10.0,
        r_multiple=r_multiple,
    )


def _fold_report(mse_skill, rank, top_excess):
    return {
        "model_vs_causal_baseline": {
            "sample_count": 100,
            "mse_skill_vs_baseline": mse_skill,
        },
        "incremental_rank_correlation": rank,
        "incremental_prediction_std_r": 0.10,
        "top_cohort_average_excess_r": top_excess,
    }
