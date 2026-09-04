from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
from unittest import TestCase

import numpy as np

from nifty_terminal.domain.candle import Candle, CandleSource, CandleStatus, Timeframe
from nifty_terminal.features.research_v3 import (
    RESEARCH_FEATURE_SET_HASH,
    RESEARCH_FEATURE_VERSION,
    build_research_feature_matrix,
)
from nifty_terminal.ml.models import DatasetBuildReport
from nifty_terminal.research.step18b import (
    BinaryCalibrationArtifact,
    _trade_path,
    apply_binary_calibrator,
    block_bootstrap_brier_skill,
    fit_binary_calibrator,
    _fit_with_convergence_check,
)
from sklearn.linear_model import LogisticRegression

from test_model_v2_research import _sample


class TradeAlignedResearchTests(TestCase):
    def test_feature_v3_removes_absolute_levels_and_is_future_invariant(self) -> None:
        candles = _candles(Timeframe.M5, 90, minutes=5)
        sample = replace(
            _sample(),
            primary_candle_id=candles[60].candle_id,
            decision_time=candles[60].closes_at,
            label_window_end=candles[60].closes_at + timedelta(minutes=60),
        )
        dataset = _dataset(sample)

        original = build_research_feature_matrix(dataset, candles)
        changed_future = list(candles)
        changed_future[80] = replace(
            changed_future[80],
            high=changed_future[80].high + Decimal("500"),
            close=changed_future[80].close + Decimal("400"),
        )
        after_future_change = build_research_feature_matrix(dataset, tuple(changed_future))

        self.assertEqual(original.rows, after_future_change.rows)
        self.assertFalse(any(name.endswith("__sma_20") for name in original.feature_names))
        self.assertFalse(any(name.endswith("__ema_50") for name in original.feature_names))
        self.assertIn("research_v3__macd_histogram_atr", original.feature_names)
        self.assertIn("research_v3__bullish_engulfing", original.feature_names)
        self.assertEqual(RESEARCH_FEATURE_VERSION, "stationary_price_features.v3")
        self.assertEqual(len(RESEARCH_FEATURE_SET_HASH), 64)

    def test_trade_target_and_stop_match_the_model_label(self) -> None:
        minutes = _candles(Timeframe.M1, 60, minutes=1)
        entry = minutes[0]
        sample = replace(
            _sample(),
            decision_time=entry.opens_at,
            label_window_end=entry.opens_at + timedelta(minutes=60),
        )
        target = entry.open + Decimal("10")
        winning_window = list(minutes)
        winning_window[2] = replace(winning_window[2], high=target)

        path = _trade_path(
            sample,
            "LONG",
            entry,
            tuple(winning_window),
            Decimal("10"),
        )

        self.assertEqual(path.exit_reason, "TARGET")
        self.assertEqual(path.success, 1)
        self.assertEqual(path.target, float(target))
        self.assertEqual(path.stop, float(entry.open - Decimal("7.5")))

    def test_same_minute_target_and_stop_is_resolved_stop_first(self) -> None:
        minutes = list(_candles(Timeframe.M1, 60, minutes=1))
        entry = minutes[0]
        sample = replace(_sample(), decision_time=entry.opens_at)
        minutes[0] = replace(
            minutes[0],
            high=entry.open + Decimal("10"),
            low=entry.open - Decimal("8"),
        )

        path = _trade_path(sample, "LONG", entry, tuple(minutes), Decimal("10"))

        self.assertEqual(path.exit_reason, "STOP")
        self.assertEqual(path.success, 0)

    def test_binary_calibrators_are_bounded_and_deterministic(self) -> None:
        probabilities = np.asarray([0.1, 0.2, 0.4, 0.6, 0.8, 0.9] * 20)
        actual = np.asarray([0, 0, 0, 1, 1, 1] * 20)
        for method in ("identity", "platt", "isotonic", "beta", "prior_shrinkage"):
            artifact = fit_binary_calibrator(
                method=method,
                probabilities=probabilities,
                actual=actual,
                prior=0.5,
            )
            first = apply_binary_calibrator(artifact, probabilities)
            second = apply_binary_calibrator(artifact, probabilities)
            np.testing.assert_allclose(first, second)
            self.assertTrue(np.all(first >= 0))
            self.assertTrue(np.all(first <= 1))

    def test_non_converged_optimizer_is_detected_and_cannot_be_silent(self) -> None:
        rng = np.random.default_rng(1701)
        matrix = rng.normal(size=(500, 100))
        labels = np.asarray([index % 2 for index in range(500)])
        model = LogisticRegression(max_iter=1, solver="saga", random_state=1701)

        converged = _fit_with_convergence_check(model, matrix, labels)

        self.assertFalse(converged)

    def test_session_bootstrap_requires_a_positive_lower_bound_not_point_skill(self) -> None:
        base = _sample()
        samples = tuple(
            replace(
                base,
                sample_id=f"sample-{index}",
                decision_time=base.decision_time + timedelta(days=index // 10, minutes=5 * index),
            )
            for index in range(100)
        )
        actual = np.asarray([index % 2 for index in range(100)])
        weak = np.full(100, 0.49)
        prior = np.full(100, 0.50)

        interval = block_bootstrap_brier_skill(
            samples=samples,
            actual=actual,
            probabilities=weak,
            prior=prior,
            iterations=100,
        )

        self.assertLessEqual(interval["lower"], 0)
        self.assertEqual(interval["session_count"], 10)

    def test_step18b_contract_is_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[3]
        with (root / "contracts" / "trade-aligned-research.v1.schema.json").open(
            "r", encoding="utf-8"
        ) as file:
            schema = json.load(file)
        properties = schema["properties"]
        for name in (
            "existing_step17_runtime_modified",
            "existing_step18_report_modified",
            "model_artifact_created",
            "approved_for_live_inference",
            "precise_probability_display_allowed",
            "official_signal_available",
            "automatic_trading_enabled",
        ):
            self.assertFalse(properties[name]["const"])


def _dataset(sample) -> DatasetBuildReport:
    return DatasetBuildReport(
        dataset_id="dataset",
        candidate_decisions=1,
        eligible_samples=1,
        outcome_support=(("DOWN", 0), ("NEITHER", 0), ("UP", 1)),
        ambiguous_labels=0,
        unavailable_labels=0,
        excluded_feature_rows=0,
        exclusion_counts=(),
        feature_names=sample.feature_names,
        labels=(),
        samples=(sample,),
    )


def _candles(
    timeframe: Timeframe, count: int, *, minutes: int
) -> tuple[Candle, ...]:
    start = datetime(2026, 8, 24, 3, 45, tzinfo=timezone.utc)
    result = []
    price = Decimal("24000")
    for index in range(count):
        opens_at = start + timedelta(minutes=index * minutes)
        change = Decimal(str((index % 7 - 3) * 0.5))
        close = price + change
        result.append(
            Candle(
                schema_version=1,
                candle_id=f"{timeframe.value}-{index}",
                instrument_id="NIFTY50_SPOT",
                timeframe=timeframe,
                opens_at=opens_at,
                closes_at=opens_at + timedelta(minutes=minutes),
                open=price,
                high=max(price, close) + Decimal("2"),
                low=min(price, close) - Decimal("2"),
                close=close,
                volume=None,
                status=CandleStatus.FINALIZED,
                revision=1,
                source=CandleSource.AGGREGATED,
                provider="test",
                source_revision=1,
                finalized_at=opens_at + timedelta(minutes=minutes),
                component_candle_ids=(),
                source_watermark=f"watermark-{index}",
            )
        )
        price = close
    return tuple(result)
