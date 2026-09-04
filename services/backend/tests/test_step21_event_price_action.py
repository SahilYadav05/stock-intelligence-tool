from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from nifty_terminal.research.step21 import (
    MAX_TRADES_PER_SESSION,
    RESEARCH_IDENTITY,
    SETUP_ORDER,
    STEP21_VERSION,
    build_event_candidates,
)


NAMES = (
    "price_action__break_of_structure_up",
    "price_action__break_of_structure_down",
    "price_action__swing_low_liquidity_sweep",
    "price_action__swing_high_liquidity_sweep",
    "price_action__range_compression_3_to_12",
    "price_action__close_location",
    "price_action__body_efficiency",
    "price_action__structure_score",
    "primary_5m__distance_ema20_atr",
)


class Step21EventPriceActionTests(TestCase):
    def test_events_use_only_current_and_same_session_previous_rows(self) -> None:
        start = datetime(2026, 8, 24, 4, 0, tzinfo=timezone.utc)
        samples = tuple(
            SimpleNamespace(decision_time=start + timedelta(minutes=5 * index))
            for index in range(3)
        )
        rows = np.zeros((3, len(NAMES)), dtype=float)
        rows[:, NAMES.index("price_action__range_compression_3_to_12")] = 1.0
        rows[1, NAMES.index("price_action__break_of_structure_up")] = 1.0
        rows[1, NAMES.index("price_action__close_location")] = 0.8
        rows[1, NAMES.index("price_action__body_efficiency")] = 0.6
        rows[2, NAMES.index("price_action__swing_high_liquidity_sweep")] = 1.0
        rows[2, NAMES.index("price_action__close_location")] = 0.2
        rows[2, NAMES.index("price_action__body_efficiency")] = -0.5

        events, diagnostics = build_event_candidates(samples, rows, NAMES)

        self.assertEqual(
            tuple((item.sample_index, item.direction, item.setup) for item in events),
            (
                (1, "LONG", "CONFIRMED_STRUCTURE_BREAK"),
                (2, "SHORT", "LIQUIDITY_SWEEP_REVERSAL"),
            ),
        )
        self.assertEqual(diagnostics["NO_EVENT"], 1)

    def test_methodology_identity_and_trade_cap_are_locked(self) -> None:
        self.assertEqual(STEP21_VERSION, "event_price_action_research.v1")
        self.assertEqual(len(RESEARCH_IDENTITY), 64)
        self.assertEqual(MAX_TRADES_PER_SESSION, 5)
        self.assertEqual(len(SETUP_ORDER), 4)

