from __future__ import annotations

from unittest import TestCase

from nifty_terminal.research.step25 import (
    FAMILY_NAMES,
    STEP25_VERSION,
    feature_family_indices,
)


class CompactFeatureAuditTests(TestCase):
    def test_feature_families_are_nested_and_price_action_first(self) -> None:
        names = (
            "price_action__structure_score",
            "price_action__break_of_structure_up",
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
            "research_v3__macd_line_atr",
            "cross__bank_minus_nifty_return_1",
            "context_15m__distance_ema20_atr",
            "context_1h__trend_ema20_above_ema50",
        )

        price = feature_family_indices("PRICE_ACTION_12", names)
        compact = feature_family_indices("STRUCTURE_LEVELS_COMPACT", names)
        cross = feature_family_indices("STRUCTURE_PLUS_CROSS_MARKET", names)
        higher = feature_family_indices("STRUCTURE_PLUS_HIGHER_TIMEFRAME", names)
        legacy = feature_family_indices("LEGACY_STRUCTURE_45", names)

        self.assertEqual(STEP25_VERSION, "compact_price_action_feature_audit.v1")
        self.assertEqual(FAMILY_NAMES[0], "PRICE_ACTION_12")
        self.assertLess(len(price), len(compact))
        self.assertLess(len(compact), len(cross))
        self.assertLess(len(cross), len(higher))
        self.assertGreater(len(legacy), len(compact))
        self.assertNotIn(names.index("research_v3__macd_line_atr"), compact)

    def test_unknown_family_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown Step 25"):
            feature_family_indices("UNKNOWN", ("price_action__structure_score",))
