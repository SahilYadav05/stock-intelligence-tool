from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest import TestCase

import numpy as np

from nifty_terminal.research.step18b import TradePath
from nifty_terminal.research.step18d import (
    RESEARCH_IDENTITY,
    STEP18D_VERSION,
    UtilityPolicy,
    _rank_correlation,
    regression_metrics,
    session_bootstrap_mse_skill,
    simulate_utility_policy,
)
from test_model_v2_research import _sample


class SelectiveUtilityResearchTests(TestCase):
    def test_regression_metrics_reward_real_skill_not_constant_confidence(self) -> None:
        actual = np.asarray([-1.0, -0.5, 0.5, 1.0] * 50)
        perfect = actual.copy()
        baseline = np.full(len(actual), float(np.mean(actual)))
        metrics = regression_metrics(actual, perfect, baseline)
        self.assertAlmostEqual(metrics["mse_skill_vs_baseline"], 1.0)
        self.assertAlmostEqual(metrics["rank_correlation"], 1.0)

    def test_rank_correlation_averages_ties(self) -> None:
        actual = np.asarray([-1.0, -1.0, 1.0, 1.0])
        predicted = np.asarray([-0.8, -0.8, 0.7, 0.7])
        self.assertAlmostEqual(_rank_correlation(actual, predicted), 1.0)

    def test_session_bootstrap_exposes_uncertain_skill(self) -> None:
        base = _sample()
        samples = tuple(
            replace(
                base,
                sample_id=f"sample-{index}",
                decision_time=base.decision_time + timedelta(days=index // 20, minutes=index),
            )
            for index in range(200)
        )
        actual = np.asarray([(-1.0 if index % 2 else 1.0) for index in range(200)])
        predicted = np.zeros(200)
        baseline = np.zeros(200)
        interval = session_bootstrap_mse_skill(
            samples=samples,
            actual=actual,
            predicted=predicted,
            baseline=baseline,
            iterations=100,
        )
        self.assertLessEqual(interval["lower"], 0.0)
        self.assertEqual(interval["session_count"], 10)

    def test_utility_policy_supports_both_directions_and_no_overlap(self) -> None:
        base = _sample()
        samples = tuple(
            replace(
                base,
                sample_id=f"sample-{index}",
                decision_time=base.decision_time + timedelta(minutes=5 * index),
            )
            for index in range(4)
        )
        long_paths = {
            item.sample_id: _path(item.sample_id, item.decision_time, "LONG", 1.0)
            for item in samples
        }
        short_paths = {
            item.sample_id: _path(item.sample_id, item.decision_time, "SHORT", 1.0)
            for item in samples
        }
        result = simulate_utility_policy(
            samples=samples,
            long_lower=np.asarray([0.4, 0.4, -0.2, -0.2]),
            short_lower=np.asarray([-0.2, -0.2, 0.5, 0.5]),
            long_paths=long_paths,
            short_paths=short_paths,
            policy=UtilityPolicy(0.1, 0.1),
        )
        self.assertEqual(result["buy_count"], 1)
        self.assertEqual(result["sell_count"], 1)
        self.assertEqual(result["trade_count"], 2)
        self.assertGreater(result["wait_counts"]["ACTIVE_POSITION"], 0)

    def test_contract_is_permanently_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[3]
        import json
        with (root / "contracts" / "selective-utility-research.v1.schema.json").open(
            "r", encoding="utf-8"
        ) as file:
            schema = json.load(file)
        self.assertEqual(STEP18D_VERSION, "selective_utility_research.v1")
        self.assertEqual(len(RESEARCH_IDENTITY), 64)
        for name in (
            "model_artifact_created",
            "approved_for_live_inference",
            "precise_probability_display_allowed",
            "official_signal_available",
            "automatic_trading_enabled",
        ):
            self.assertFalse(schema["properties"][name]["const"])


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
