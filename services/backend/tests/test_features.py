from __future__ import annotations

from dataclasses import replace
from unittest import TestCase

from history_fixture import finalized_candles
from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.domain.candle import CandleStatus
from nifty_terminal.features.definitions import FEATURE_SET_HASH, FEATURE_VERSION
from nifty_terminal.features.engine import PriceFeatureEngine


class PriceFeatureEngineTests(TestCase):
    def setUp(self) -> None:
        self.engine = PriceFeatureEngine(NseSessionCalendar())

    def test_warmup_is_explicit_and_row_fifty_is_ready(self) -> None:
        rows = self.engine.calculate(finalized_candles(60))

        self.assertFalse(rows[48].is_ready)
        self.assertTrue(rows[49].is_ready)
        self.assertEqual(rows[49].feature_version, FEATURE_VERSION)
        self.assertEqual(rows[49].feature_set_hash, FEATURE_SET_HASH)
        self.assertIsNotNone(rows[49].get("atr_14"))
        self.assertIsNotNone(rows[49].get("rsi_14"))
        self.assertIsNotNone(rows[49].get("ema_50"))

    def test_batch_and_point_in_time_prefix_results_are_identical(self) -> None:
        candles = finalized_candles(60)
        batch_row = self.engine.calculate(candles)[54]
        point_in_time_row = self.engine.calculate(candles[:55])[-1]

        self.assertEqual(batch_row, point_in_time_row)

    def test_future_candle_change_cannot_change_prior_features(self) -> None:
        candles = finalized_candles(60)
        baseline = self.engine.calculate(candles)[54]
        changed_future = candles[:-1] + (
            replace(candles[-1], close=candles[-1].close + 100),
        )

        self.assertEqual(self.engine.calculate(changed_future)[54], baseline)

    def test_developing_candle_is_rejected(self) -> None:
        candles = finalized_candles(50)
        developing = replace(candles[-1], status=CandleStatus.DEVELOPING, finalized_at=None)
        with self.assertRaisesRegex(ValueError, "Developing"):
            self.engine.calculate(candles[:-1] + (developing,))

    def test_intraday_gap_blocks_affected_feature_window(self) -> None:
        rows = self.engine.calculate(finalized_candles(60, skip_index=25))
        self.assertFalse(rows[-1].is_ready)
        self.assertIn("INTRADAY_GAP_IN_FEATURE_WINDOW", rows[-1].blockers)

    def test_nifty_feature_set_contains_no_fabricated_volume_feature(self) -> None:
        names = {name for name, _ in self.engine.calculate(finalized_candles(50))[-1].values}
        self.assertFalse(any("volume" in name or "vwap" in name for name in names))
