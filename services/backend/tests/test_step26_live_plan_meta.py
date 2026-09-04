from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from unittest import TestCase

import numpy as np

from history_fixture import SESSION_OPEN
from nifty_terminal.price_action.replay import PriceActionPathResult
from nifty_terminal.research.step26 import (
    LivePlanCandidate,
    MetaPolicy,
    STEP26_VERSION,
    _meta_design,
    _wilson_interval,
    replay_candidates,
)


def _candidate(index: int, *, direction: str = "BUY") -> LivePlanCandidate:
    decision = SESSION_OPEN + timedelta(minutes=index * 5)
    result = PriceActionPathResult(
        status="TARGET1_REACHED",
        direction=direction,
        entered_at=decision,
        exited_at=decision + timedelta(minutes=1),
        entry_price=Decimal("100.5"),
        exit_price=Decimal("102.5"),
        maximum_target_reached=1,
        stop_hit=False,
        net_points=Decimal("1.5"),
        r_multiple=Decimal("0.75"),
        realized_allocations=(Decimal("1"), Decimal("0"), Decimal("0")),
        remaining_allocation=Decimal("0"),
    )
    return LivePlanCandidate(
        sample_index=index,
        sample_id=f"sample-{index}",
        decision_time=decision,
        direction=direction,
        confluence_score=70,
        evidence_grade="STRONG",
        volatility_regime="NORMAL",
        risk_atr=1.0,
        paths=(("FULL_TARGET1", result),),
    )


class Step26LivePlanMetaTests(TestCase):
    def test_meta_model_can_filter_but_not_reverse_price_action_direction(self) -> None:
        candidates = (_candidate(0, direction="BUY"), _candidate(1, direction="SELL"))
        matrix = np.asarray([[1.0, 2.0], [3.0, 4.0]])

        design = _meta_design(candidates, matrix, np.asarray([0, 1]))

        np.testing.assert_array_equal(design[0], [1.0, 2.0, 1.0, 2.0, 1.0])
        np.testing.assert_array_equal(design[1], [3.0, 4.0, -3.0, -4.0, -1.0])

    def test_policy_uses_score_threshold_and_keeps_meaningful_trade_count(self) -> None:
        candidates = tuple(_candidate(index) for index in range(6))
        metrics = replay_candidates(
            candidates=candidates,
            policy_name="FULL_TARGET1",
            scores=np.asarray([0.1, 0.8, 0.2, 0.9, 0.3, 1.0]),
            meta_policy=MetaPolicy(0.5),
        )

        self.assertEqual(metrics["trade_count"], 3)
        self.assertEqual(metrics["win_rate"], 1.0)
        self.assertEqual(metrics["wait_counts"]["META_SCORE_BELOW_THRESHOLD"], 3)

    def test_wilson_interval_is_reported_instead_of_point_estimate_only(self) -> None:
        interval = _wilson_interval(70, 100)

        self.assertLess(interval["lower"], 0.70)
        self.assertGreater(interval["upper"], 0.70)
        self.assertEqual(STEP26_VERSION, "live_plan_meta_label_research.v1")
