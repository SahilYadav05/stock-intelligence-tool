from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from nifty_terminal.research.step23 import (
    STEP23_VERSION,
    _calibrated_scores,
    _chosen_wins,
    conditional_labels,
)
from nifty_terminal.research.step18b import BinaryCalibrationArtifact


class ConditionalDirectionResearchTests(TestCase):
    def test_conditional_labels_separate_opportunity_and_side(self) -> None:
        samples = tuple(SimpleNamespace(sample_id=value) for value in ("a", "b", "c"))
        long_paths = {
            "a": SimpleNamespace(success=1),
            "b": SimpleNamespace(success=0),
            "c": SimpleNamespace(success=0),
        }
        short_paths = {
            "a": SimpleNamespace(success=0),
            "b": SimpleNamespace(success=1),
            "c": SimpleNamespace(success=0),
        }

        labels = conditional_labels(samples, long_paths, short_paths)

        self.assertEqual(STEP23_VERSION, "conditional_direction_research.v1")
        self.assertEqual(labels.opportunity.tolist(), [1, 1, 0])
        self.assertEqual(labels.long_when_opportunity.tolist(), [1, 0, 0])

    def test_two_successful_sides_fail_closed(self) -> None:
        samples = (SimpleNamespace(sample_id="a"),)
        paths = {"a": SimpleNamespace(success=1)}

        with self.assertRaisesRegex(ValueError, "two successful sides"):
            conditional_labels(samples, paths, paths)

    def test_joint_scores_factor_opportunity_and_direction(self) -> None:
        identity = BinaryCalibrationArtifact(method="identity", parameters={})
        scores = _calibrated_scores(
            {
                "opportunity": np.asarray([0.8, 0.6]),
                "direction": np.asarray([0.75, 0.25]),
            },
            opportunity_calibration=identity,
            direction_calibration=identity,
        )

        np.testing.assert_allclose(scores["LONG"], [0.6, 0.15])
        np.testing.assert_allclose(scores["SHORT"], [0.2, 0.45])
        np.testing.assert_allclose(scores["LONG"] + scores["SHORT"], [0.8, 0.6])

    def test_chosen_wins_require_opportunity_and_correct_side(self) -> None:
        wins = _chosen_wins(
            np.asarray([1, 1, 0]),
            np.asarray([1, 0, 0]),
            np.asarray([0.7, 0.2, 0.1]),
        )

        self.assertEqual(wins.tolist(), [True, True, False])
