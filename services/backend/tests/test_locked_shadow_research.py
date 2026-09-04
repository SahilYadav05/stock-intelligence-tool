from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from unittest import TestCase

import numpy as np

from history_fixture import SESSION_OPEN
from nifty_terminal.ml.models import (
    DatasetBuildReport,
    MetricSummary,
    TargetOutcome,
    TrainingSample,
)
from nifty_terminal.research.step16 import (
    CALIBRATION_METHODS,
    LOCKED_ATR_MULTIPLIER,
    LOCKED_CANDIDATE,
    _calibration_blockers,
    _direction,
    _simulate_trade,
    apply_calibrator,
    build_shadow_artifact,
    fit_calibrator,
)

from test_ml_labels import _minute_candles


class LockedShadowResearchTests(TestCase):
    def test_locked_candidate_and_target_match_corrected_step15_result(self) -> None:
        self.assertEqual(format(LOCKED_ATR_MULTIPLIER, "f"), "1.5")
        self.assertEqual(LOCKED_CANDIDATE, "multinomial_logistic_unweighted")

    def test_all_calibrators_return_valid_ordered_probabilities(self) -> None:
        probabilities = np.asarray(
            [
                [0.60, 0.10, 0.30],
                [0.20, 0.20, 0.60],
                [0.10, 0.70, 0.20],
                [0.55, 0.15, 0.30],
                [0.25, 0.20, 0.55],
                [0.20, 0.60, 0.20],
                [0.50, 0.20, 0.30],
                [0.20, 0.30, 0.50],
                [0.30, 0.50, 0.20],
            ],
            dtype=float,
        )
        actual = (
            "DOWN", "UP", "NEITHER", "DOWN", "UP", "NEITHER", "DOWN", "UP", "NEITHER"
        )
        prior = np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=float)

        for method in CALIBRATION_METHODS:
            with self.subTest(method=method):
                artifact = fit_calibrator(
                    method=method,
                    probabilities=probabilities,
                    actual=actual,
                    prior=prior,
                )
                transformed = apply_calibrator(artifact, probabilities)
                self.assertEqual(transformed.shape, probabilities.shape)
                self.assertTrue(np.isfinite(transformed).all())
                self.assertTrue((transformed >= 0).all())
                np.testing.assert_allclose(transformed.sum(axis=1), 1.0)
                json.dumps(artifact.to_contract())

    def test_signal_policy_waits_until_directional_evidence_is_material(self) -> None:
        self.assertIsNone(_direction({"UP": 0.59, "DOWN": 0.20, "NEITHER": 0.21}))
        self.assertIsNone(_direction({"UP": 0.61, "DOWN": 0.48, "NEITHER": 0.00}))
        self.assertEqual(
            _direction({"UP": 0.66, "DOWN": 0.20, "NEITHER": 0.14}),
            "BUY",
        )
        self.assertEqual(
            _direction({"UP": 0.18, "DOWN": 0.68, "NEITHER": 0.14}),
            "SELL",
        )

    def test_same_minute_stop_and_target_is_resolved_stop_first(self) -> None:
        minutes = list(_minute_candles(_samples()[0].decision_time, 60))
        minutes[0] = replace(
            minutes[0], high=Decimal("102.5"), low=Decimal("97.5")
        )
        trade = _simulate_trade(
            sample=_samples()[0],
            direction="BUY",
            probability=0.70,
            entry=minutes[0],
            atr=Decimal("2"),
            minute_by_open={item.opens_at: item for item in minutes},
        )

        self.assertIsNotNone(trade)
        self.assertEqual(trade["exit_reason"], "STOP")
        self.assertLess(trade["net_points"], 0)

    def test_shadow_artifact_is_json_only_hash_verified_and_not_releasable(self) -> None:
        dataset = _dataset_report()
        calibrator = fit_calibrator(
            method="identity",
            probabilities=np.asarray([[0.4, 0.2, 0.4]] * 9),
            actual=tuple(item.outcome.value for item in dataset.samples),
            prior=np.asarray([1 / 3, 1 / 3, 1 / 3]),
        )
        artifact = build_shadow_artifact(dataset=dataset, calibrator=calibrator)
        expected = artifact.pop("sha256")
        actual = hashlib.sha256(
            json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        self.assertEqual(actual, expected)
        self.assertTrue(artifact["shadow_only"])
        self.assertFalse(artifact["approved_for_live_inference"])
        self.assertEqual(artifact["serialization"], "SAFE_JSON_PARAMETERS_ONLY")
        json.dumps(artifact)

    def test_historical_success_cannot_remove_forward_confirmation_blockers(self) -> None:
        metrics = _metrics(brier=0.55, log_loss=0.90, balanced_accuracy=0.40, ece=0.02)
        prior = _metrics(brier=0.60, log_loss=1.00, balanced_accuracy=1 / 3, ece=0.01)
        blockers = _calibration_blockers(
            raw=metrics,
            calibrated=metrics,
            prior=prior,
            brier_skill=1 - 0.55 / 0.60,
        )

        self.assertIn("TARGET_SELECTED_USING_THIS_HISTORICAL_PERIOD", blockers)
        self.assertIn("FORWARD_CONFIRMATION_NOT_COMPLETED", blockers)

    def test_step16_contracts_are_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[3]
        for name in (
            "locked-shadow-research.v1.schema.json",
            "shadow-model-artifact.v1.schema.json",
        ):
            with (root / "contracts" / name).open("r", encoding="utf-8") as file:
                schema = json.load(file)
            self.assertFalse(schema["properties"]["approved_for_live_inference"]["const"])


def _samples() -> tuple[TrainingSample, ...]:
    start = SESSION_OPEN + timedelta(hours=1)
    outcomes = (TargetOutcome.DOWN, TargetOutcome.NEITHER, TargetOutcome.UP) * 3
    return tuple(
        TrainingSample(
            sample_id=f"sample-{index}",
            dataset_id="dataset-step16",
            instrument_id="NIFTY50_SPOT",
            decision_time=start + timedelta(minutes=index * 65),
            label_window_end=start + timedelta(minutes=index * 65 + 60),
            label_id=f"label-{index}",
            outcome=outcome,
            primary_candle_id=f"primary-{index}",
            context_15m_candle_id=f"context15-{index}",
            context_1h_candle_id=f"context1h-{index}",
            input_revision_checksum=f"checksum-{index}",
            feature_names=("feature_a", "feature_b"),
            feature_values=(float(index), float((index * index) % 7)),
        )
        for index, outcome in enumerate(outcomes)
    )


def _dataset_report() -> DatasetBuildReport:
    samples = _samples()
    return DatasetBuildReport(
        dataset_id="dataset-step16",
        candidate_decisions=len(samples),
        eligible_samples=len(samples),
        outcome_support=(("DOWN", 3), ("NEITHER", 3), ("UP", 3)),
        ambiguous_labels=0,
        unavailable_labels=0,
        excluded_feature_rows=0,
        exclusion_counts=(),
        feature_names=("feature_a", "feature_b"),
        labels=(),
        samples=samples,
    )


def _metrics(
    *, brier: float, log_loss: float, balanced_accuracy: float, ece: float
) -> MetricSummary:
    return MetricSummary(
        sample_count=2_000,
        accuracy=0.45,
        balanced_accuracy=balanced_accuracy,
        multiclass_brier=brier,
        log_loss=log_loss,
        raw_ece_10_bin=ece,
        class_support=(("DOWN", 700), ("NEITHER", 600), ("UP", 700)),
        class_recall=(("DOWN", 0.40), ("NEITHER", 0.40), ("UP", 0.40)),
    )
