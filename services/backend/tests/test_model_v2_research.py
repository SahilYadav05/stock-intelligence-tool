from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest import TestCase

import numpy as np

from nifty_terminal.features.enhanced import (
    ENHANCED_FEATURE_DEFINITIONS,
    ENHANCED_FEATURE_SET_HASH,
    ENHANCED_FEATURE_VERSION,
    enhance_sample,
)
from nifty_terminal.ml.models import TargetOutcome, TrainingSample
from nifty_terminal.research.step18 import (
    COLLAPSE_GATE,
    FINAL_RESEARCH_GATE,
    combine_hierarchical_probabilities,
    collapse_blockers,
    collapse_diagnostics,
)


class ModelV2ResearchTests(TestCase):
    def test_hierarchical_probability_composition_uses_canonical_class_order(self) -> None:
        result = combine_hierarchical_probabilities(
            np.asarray([0.8, 0.4]),
            np.asarray([0.75, 0.25]),
        )

        np.testing.assert_allclose(
            result,
            np.asarray(
                [
                    [0.20, 0.20, 0.60],
                    [0.30, 0.60, 0.10],
                ]
            ),
        )
        np.testing.assert_allclose(result.sum(axis=1), np.ones(2))

    def test_directional_collapse_is_a_hard_research_blocker(self) -> None:
        actual = ("DOWN", "DOWN", "NEITHER", "UP", "UP")
        probabilities = np.asarray(
            [
                [0.70, 0.20, 0.10],
                [0.60, 0.30, 0.10],
                [0.55, 0.35, 0.10],
                [0.65, 0.20, 0.15],
                [0.60, 0.25, 0.15],
            ]
        )

        diagnostics = collapse_diagnostics(actual, probabilities)
        blockers = collapse_blockers(diagnostics)

        self.assertTrue(diagnostics["directional_collapse_detected"])
        self.assertIn("UP_PREDICTION_SHARE_TOO_LOW", blockers)
        self.assertIn("UP_RECALL_TOO_LOW", blockers)

    def test_enhanced_features_are_deterministic_and_do_not_replace_base_features(self) -> None:
        sample = _sample()

        first = enhance_sample(sample)
        second = enhance_sample(sample)

        self.assertEqual(first, second)
        self.assertEqual(first.feature_names[: len(sample.feature_names)], sample.feature_names)
        self.assertEqual(
            len(first.feature_names),
            len(sample.feature_names) + len(ENHANCED_FEATURE_DEFINITIONS),
        )
        self.assertNotEqual(first.input_revision_checksum, sample.input_revision_checksum)
        self.assertEqual(ENHANCED_FEATURE_VERSION, "price_features.v2")
        self.assertEqual(len(ENHANCED_FEATURE_SET_HASH), 64)

    def test_step18_contract_is_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[3]
        with (root / "contracts" / "model-v2-research.v1.schema.json").open(
            "r", encoding="utf-8"
        ) as file:
            schema = json.load(file)

        properties = schema["properties"]
        self.assertFalse(properties["approved_for_live_inference"]["const"])
        self.assertFalse(properties["official_signal_available"]["const"])
        self.assertFalse(properties["automatic_trading_enabled"]["const"])
        self.assertFalse(properties["existing_step17_runtime_modified"]["const"])
        self.assertFalse(properties["model_artifact_created"]["const"])
        self.assertGreater(COLLAPSE_GATE["minimum_up_prediction_share"], 0)
        self.assertFalse(FINAL_RESEARCH_GATE["live_inference_approval"])


def _sample() -> TrainingSample:
    prefixes = ("primary_5m", "context_15m", "context_1h")
    names = []
    values = []
    defaults = {
        "return_1": 0.001,
        "return_5": 0.002,
        "log_return_1": 0.001,
        "range_pct": 0.003,
        "body_pct": 0.001,
        "upper_wick_pct": 0.0005,
        "lower_wick_pct": 0.0008,
        "sma_20": 100.0,
        "sma_50": 99.0,
        "ema_20": 101.0,
        "ema_50": 99.0,
        "atr_14": 2.0,
        "atr_pct": 0.02,
        "rsi_14": 55.0,
        "rolling_vol_20": 0.01,
        "bollinger_z_20": 0.3,
        "roc_5": 0.01,
        "roc_12": 0.02,
        "distance_ema20_atr": 0.5,
        "range_atr": 1.1,
        "trend_ema20_above_ema50": 1.0,
        "breakout_up_20": 0.0,
        "breakout_down_20": 0.0,
        "minute_of_session": 120.0,
        "minutes_to_session_close": 255.0,
        "day_of_week": 2.0,
    }
    for prefix in prefixes:
        for suffix, value in defaults.items():
            names.append(f"{prefix}__{suffix}")
            values.append(value)
    decision = datetime(2026, 8, 25, 5, 30, tzinfo=timezone.utc)
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
        input_revision_checksum="base-checksum",
        feature_names=tuple(names),
        feature_values=tuple(values),
    )
