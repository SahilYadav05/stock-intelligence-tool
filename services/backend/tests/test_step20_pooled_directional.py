from datetime import datetime, timedelta
from types import SimpleNamespace
import unittest
import numpy as np

from nifty_terminal.calendar.nse import IST
from nifty_terminal.research.step18b import TradePath
from nifty_terminal.research.step18b import BinaryCalibrationArtifact
from nifty_terminal.research.step20 import (
    CoveragePolicy,
    _calibration_blockers,
    _directional_design,
    _reference_percentiles,
    simulate_coverage_policy,
)


class Step20PooledDirectionalTests(unittest.TestCase):
    def test_directional_design_keeps_raw_values_and_adds_side_interactions(self):
        design = _directional_design(np.asarray([[1.0, 2.0], [3.0, 4.0]]))
        self.assertEqual(design.shape, (4, 5))
        np.testing.assert_array_equal(design[0], [1.0, 2.0, 1.0, 2.0, 1.0])
        np.testing.assert_array_equal(design[2], [1.0, 2.0, -1.0, -2.0, -1.0])

    def test_percentiles_use_only_the_supplied_earlier_reference(self):
        observed = _reference_percentiles(
            np.asarray([0.20, 0.40, 0.60, 0.80]),
            np.asarray([0.10, 0.40, 0.90]),
        )
        np.testing.assert_array_equal(observed, [0.0, 0.5, 1.0])

    def test_calibration_cannot_reverse_model_rank(self):
        artifact = BinaryCalibrationArtifact(
            "platt", {"coefficient": -0.25, "intercept": 0.1}
        )
        self.assertEqual(_calibration_blockers(artifact), ["CALIBRATION_RANK_REVERSAL"])

    def test_policy_enforces_non_overlapping_positions(self):
        start = datetime(2026, 1, 2, 9, 20, tzinfo=IST)
        samples = tuple(
            SimpleNamespace(sample_id=f"sample-{index}", decision_time=start + timedelta(minutes=5 * index))
            for index in range(3)
        )
        long_paths = {
            sample.sample_id: self._path(
                sample.sample_id,
                sample.decision_time,
                sample.decision_time + timedelta(minutes=8),
                "LONG",
            )
            for sample in samples
        }
        short_paths = {
            sample.sample_id: self._path(
                sample.sample_id,
                sample.decision_time,
                sample.decision_time + timedelta(minutes=8),
                "SHORT",
            )
            for sample in samples
        }
        result = simulate_coverage_policy(
            samples=samples,
            score_percentiles=np.ones(3),
            long_probabilities=np.asarray([0.70, 0.70, 0.70]),
            short_probabilities=np.asarray([0.30, 0.30, 0.30]),
            long_paths=long_paths,
            short_paths=short_paths,
            policy=CoveragePolicy(0.45, 0.05),
            benchmarks={"WAIT": {"daily_total_r": {}}},
        )
        self.assertEqual(result["trade_count"], 2)
        self.assertEqual(result["wait_counts"], {"ACTIVE_POSITION": 1})
        self.assertEqual(result["buy_count"], 2)

    @staticmethod
    def _path(sample_id, entered_at, exited_at, direction):
        return TradePath(
            sample_id=sample_id,
            decision_time=entered_at,
            direction=direction,
            success=1,
            exit_reason="TARGET",
            entered_at=entered_at,
            exited_at=exited_at,
            entry=100.0,
            stop=99.0,
            target=101.0,
            exit=101.0,
            net_points=1.0,
            r_multiple=1.0,
        )


if __name__ == "__main__":
    unittest.main()
