from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest import TestCase

import numpy as np

from nifty_terminal.calibration.models import CalibrationConfig, CalibrationObservation
from nifty_terminal.calibration.pipeline import CalibrationPipeline
from nifty_terminal.calibration.temperature import apply_temperature, fit_temperature
from nifty_terminal.ml.models import TargetOutcome


class CalibrationTests(TestCase):
    def test_temperature_scaling_preserves_valid_multiclass_probabilities(self) -> None:
        raw = np.asarray([[0.10, 0.20, 0.70], [0.70, 0.20, 0.10]])
        calibrated = apply_temperature(raw, 1.8)

        self.assertTrue(np.allclose(calibrated.sum(axis=1), 1.0))
        self.assertTrue(np.all(calibrated >= 0.0))
        self.assertLess(calibrated[0, 2], raw[0, 2])

    def test_temperature_is_fitted_only_from_earlier_partition(self) -> None:
        observations = _observations(90)
        report = CalibrationPipeline().run(
            observations=observations,
            config=_permissive_config(),
            created_at=datetime(2026, 8, 24, 12, tzinfo=timezone.utc),
        )

        self.assertEqual(report.fit_observations, 54)
        self.assertEqual(report.evaluation_observations, 36)
        self.assertLess(report.artifact.fit_ends_at, report.artifact.evaluation_starts_at)
        self.assertEqual(len(report.predictions), 36)
        self.assertTrue(all(item.evaluation_partition for item in report.predictions))
        self.assertTrue(report.release_gate_passed)
        self.assertGreater(report.artifact.temperature, 0.0)

    def test_default_release_gate_blocks_small_research_sample(self) -> None:
        report = CalibrationPipeline().run(observations=_observations(90))

        self.assertFalse(report.release_gate_passed)
        self.assertIn("INSUFFICIENT_TOTAL_OOS_PREDICTIONS", report.blockers)
        self.assertFalse(report.to_contract()["precise_probability_display_allowed"])

    def test_duplicate_source_predictions_are_rejected(self) -> None:
        rows = _observations(6)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            CalibrationPipeline().run(observations=rows + (rows[-1],))

    def test_temperature_fit_is_deterministic(self) -> None:
        raw = np.asarray([[0.05, 0.10, 0.85], [0.80, 0.10, 0.10], [0.10, 0.80, 0.10]])
        actual = np.asarray([2, 0, 1])
        self.assertAlmostEqual(fit_temperature(raw, actual), fit_temperature(raw, actual), places=12)


def _observations(count: int) -> tuple[CalibrationObservation, ...]:
    starts_at = datetime(2026, 1, 1, 4, tzinfo=timezone.utc)
    outcomes = (TargetOutcome.DOWN, TargetOutcome.NEITHER, TargetOutcome.UP)
    rows = []
    for index in range(count):
        actual = outcomes[index % 3]
        probabilities = {
            "DOWN": 0.10,
            "NEITHER": 0.10,
            "UP": 0.10,
        }
        probabilities[actual.value] = 0.80
        rows.append(
            CalibrationObservation(
                prediction_id=f"prediction-{index}",
                run_id="run-1",
                candidate_name="multinomial_logistic",
                fold_index=1 if index < count // 2 else 2,
                decision_time=starts_at + timedelta(minutes=5 * index),
                raw_probabilities=tuple(sorted(probabilities.items())),
                actual_outcome=actual,
            )
        )
    return tuple(rows)


def _permissive_config() -> CalibrationConfig:
    return CalibrationConfig(
        minimum_total_predictions=30,
        minimum_fit_class_support=5,
        minimum_evaluation_class_support=5,
        minimum_supported_probability_bin=1,
        minimum_supported_probability_bins=1,
        maximum_ece=1.0,
        minimum_brier_skill=-10.0,
        maximum_slice_ece=1.0,
        minimum_slice_brier_skill=-10.0,
        minimum_slice_samples=1,
    )
