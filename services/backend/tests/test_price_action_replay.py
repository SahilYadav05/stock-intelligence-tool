from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from unittest import TestCase

from history_fixture import SESSION_OPEN
from nifty_terminal.price_action.models import ConditionalTradePlan
from nifty_terminal.price_action.replay import (
    FULL_TARGET1_POLICY,
    SCALE_PROTECTED_POLICY,
    replay_price_action_plan,
)
from test_ml_labels import _minute_candles


def _plan() -> ConditionalTradePlan:
    return ConditionalTradePlan(
        direction="BUY",
        trigger=Decimal("100"),
        entry_low=Decimal("99.50"),
        entry_high=Decimal("100.50"),
        stop=Decimal("98"),
        invalidation=Decimal("98"),
        target1=Decimal("102.50"),
        target2=Decimal("104"),
        target3=Decimal("106"),
        risk_points=Decimal("2"),
        target1_reward_risk=1.25,
        target2_reward_risk=2.0,
        target3_reward_risk=3.0,
        expiry_bars=12,
        blockers=(),
    )


class PriceActionReplayTests(TestCase):
    def test_full_target1_policy_accounts_for_entry_and_exit_slippage(self) -> None:
        candles = list(_minute_candles(SESSION_OPEN, 60))
        candles[1] = replace(candles[1], high=Decimal("103"))

        result = replay_price_action_plan(
            plan=_plan(), minute_candles=tuple(candles), policy=FULL_TARGET1_POLICY
        )

        self.assertEqual(result.status, "TARGET1_REACHED")
        self.assertEqual(result.maximum_target_reached, 1)
        self.assertEqual(result.net_points, Decimal("1.50"))
        self.assertEqual(result.r_multiple, Decimal("0.75"))

    def test_stop_is_resolved_before_target_in_same_minute(self) -> None:
        candles = list(_minute_candles(SESSION_OPEN, 60))
        candles[0] = replace(
            candles[0], high=Decimal("103"), low=Decimal("97")
        )

        result = replay_price_action_plan(
            plan=_plan(), minute_candles=tuple(candles), policy=FULL_TARGET1_POLICY
        )

        self.assertEqual(result.status, "STOPPED")
        self.assertTrue(result.stop_hit)
        self.assertEqual(result.maximum_target_reached, 0)

    def test_protected_scale_moves_stop_only_for_following_minute(self) -> None:
        candles = list(_minute_candles(SESSION_OPEN, 60))
        candles[0] = replace(candles[0], high=Decimal("103"), low=Decimal("99"))
        candles[1] = replace(candles[1], high=Decimal("101"), low=Decimal("99.5"))

        result = replay_price_action_plan(
            plan=_plan(),
            minute_candles=tuple(candles),
            policy=SCALE_PROTECTED_POLICY,
        )

        self.assertEqual(result.status, "STOPPED_AFTER_TARGET")
        self.assertEqual(result.maximum_target_reached, 1)
        self.assertEqual(result.realized_allocations[0], Decimal("0.50"))

    def test_gap_beyond_entry_zone_is_not_chased(self) -> None:
        candles = list(_minute_candles(SESSION_OPEN, 60))
        candles[0] = replace(
            candles[0], open=Decimal("101"), high=Decimal("103"), low=Decimal("100.8")
        )

        result = replay_price_action_plan(
            plan=_plan(), minute_candles=tuple(candles), policy=FULL_TARGET1_POLICY
        )

        self.assertEqual(result.status, "MISSED_GAP_BEYOND_ENTRY_ZONE")
        self.assertFalse(result.entered)
