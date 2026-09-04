from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest import TestCase

import numpy as np

from nifty_terminal.research.step18b import TradePath
from nifty_terminal.research.step18e import (
    RESEARCH_IDENTITY,
    STEP18E_VERSION,
    RankedUtilityPolicy,
    architecture_feature_indices,
    causal_score_percentiles,
    score_decile_table,
    simulate_ranked_policy,
)
from test_model_v2_research import _sample


class RankedUtilityResearchTests(TestCase):
    def test_causal_percentile_never_uses_later_scores(self) -> None:
        reference = np.asarray([0.0, 1.0, 2.0])
        first = causal_score_percentiles(reference, np.asarray([1.5, 100.0, -100.0]))
        changed = causal_score_percentiles(reference, np.asarray([1.5, 100.0, 999.0]))
        self.assertAlmostEqual(first[0], changed[0])
        self.assertAlmostEqual(first[1], changed[1])
        self.assertAlmostEqual(first[0], 2.0 / 3.0)

    def test_architectures_do_not_silently_mix_context(self) -> None:
        names = (
            "nifty__return",
            "context_market__banknifty_spot__return_1",
            "context_market__india_vix_spot__return_1",
            "cross__bank_minus_nifty_return_1",
            "cross__vix_shock_abs",
        )
        self.assertEqual(
            architecture_feature_indices("NIFTY_ONLY", names, 1).tolist(), [0]
        )
        self.assertEqual(
            architecture_feature_indices("NIFTY_BANK", names, 1).tolist(), [0, 1, 3]
        )
        self.assertEqual(
            architecture_feature_indices("NIFTY_VIX", names, 1).tolist(), [0, 2, 4]
        )
        self.assertEqual(
            architecture_feature_indices("ALL_CONTEXT", names, 1).tolist(),
            [0, 1, 2, 3, 4],
        )

    def test_ranked_policy_supports_both_directions_and_no_overlap(self) -> None:
        base = _sample()
        samples = tuple(
            replace(
                base,
                sample_id=f"sample-{index}",
                decision_time=base.decision_time + timedelta(minutes=5 * index),
            )
            for index in range(5)
        )
        long_paths = {
            item.sample_id: _path(item.sample_id, item.decision_time, "LONG", 1.0)
            for item in samples
        }
        short_paths = {
            item.sample_id: _path(item.sample_id, item.decision_time, "SHORT", 1.0)
            for item in samples
        }
        result = simulate_ranked_policy(
            samples=samples,
            long_percentiles=np.asarray([0.95, 0.90, 0.1, 0.1, 0.1]),
            short_percentiles=np.asarray([0.1, 0.1, 0.95, 0.90, 0.1]),
            long_paths=long_paths,
            short_paths=short_paths,
            policy=RankedUtilityPolicy(0.8, 0.1),
        )
        self.assertEqual(result["buy_count"], 1)
        self.assertEqual(result["sell_count"], 1)
        self.assertEqual(result["trade_count"], 2)
        self.assertGreater(result["wait_counts"]["ACTIVE_POSITION"], 0)

    def test_score_deciles_preserve_support(self) -> None:
        percentiles = np.linspace(0.0, 1.0, 101)
        actual = np.arange(101, dtype=float)
        rows = score_decile_table(percentiles, actual)
        self.assertEqual(len(rows), 10)
        self.assertEqual(sum(item["sample_count"] for item in rows), 101)

    def test_contract_is_permanently_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[3]
        import json
        with (root / "contracts" / "ranked-utility-research.v1.schema.json").open(
            "r", encoding="utf-8"
        ) as file:
            schema = json.load(file)
        self.assertEqual(STEP18E_VERSION, "ranked_utility_cohort_research.v1")
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
