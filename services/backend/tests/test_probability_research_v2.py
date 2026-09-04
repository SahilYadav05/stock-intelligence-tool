from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

from nifty_terminal.calendar.nse import NseSessionCalendar
from nifty_terminal.ml.labels import FirstTouchLabeler, symmetric_first_touch_config
from nifty_terminal.ml.models import MetricSummary
from nifty_terminal.research.v2 import (
    BARRIER_MULTIPLIERS,
    CANDIDATE_NAMES,
    SCREENING_GATE_DEFINITION,
    _screening_blockers,
    _target_ranking_key,
)

from test_ml_labels import _atr_feature, _five_minute_candles


class ProbabilityResearchV2Tests(TestCase):
    def test_symmetric_research_target_changes_barriers_without_changing_horizon(self) -> None:
        candles = _five_minute_candles(13)
        config = symmetric_first_touch_config(Decimal("1.5"))
        label = FirstTouchLabeler(NseSessionCalendar(), config).build(
            dataset_id="dataset-v2",
            primary_candles=candles,
            primary_features=(_atr_feature(candles[0], Decimal("2")),),
        )[0]

        self.assertEqual(label.up_barrier, Decimal("103.0"))
        self.assertEqual(label.down_barrier, Decimal("97.0"))
        self.assertEqual(label.window_ends_at, candles[0].closes_at + timedelta(minutes=60))
        self.assertIn("symmetric_1.5", label.label_version)

    def test_experiment_definitions_include_probability_safe_candidates(self) -> None:
        self.assertEqual(BARRIER_MULTIPLIERS, (Decimal("1.0"), Decimal("1.25"), Decimal("1.5")))
        self.assertIn("multinomial_logistic_unweighted", CANDIDATE_NAMES)
        self.assertIn("hist_gradient_boosting_unweighted", CANDIDATE_NAMES)
        self.assertEqual(SCREENING_GATE_DEFINITION["final_screening_fold"], 4)
        self.assertFalse(SCREENING_GATE_DEFINITION["live_inference_approval"])

    def test_proper_score_failure_blocks_screening_even_with_balanced_accuracy(self) -> None:
        prior = _metrics(brier=0.50, log_loss=0.80, balanced_accuracy=1 / 3, ece=0.01)
        raw = _metrics(brier=0.55, log_loss=0.90, balanced_accuracy=0.44, ece=0.02)
        calibrated = _metrics(brier=0.56, log_loss=0.91, balanced_accuracy=0.44, ece=0.01)

        blockers = _screening_blockers(
            full_support={"DOWN": 10_000, "NEITHER": 2_000, "UP": 10_000},
            final_support={"DOWN": 900, "NEITHER": 200, "UP": 900},
            raw=raw,
            calibrated=calibrated,
            prior=prior,
            brier_skill=-0.12,
        )

        self.assertIn("FINAL_BRIER_SKILL_GATE_FAILED", blockers)
        self.assertIn("FINAL_LOG_LOSS_GATE_FAILED", blockers)
        self.assertIn("OUT_OF_TIME_CALIBRATION_DEGRADATION", blockers)

    def test_different_targets_are_ranked_by_normalized_skill_not_absolute_brier(self) -> None:
        low_absolute_brier_but_negative_skill = SimpleNamespace(
            atr_multiplier=Decimal("1.0"),
            brier_skill_vs_prior=-0.004,
            calibrated_final_metrics=_metrics(
                brier=0.54,
                log_loss=0.83,
                balanced_accuracy=0.34,
                ece=0.03,
            ),
        )
        higher_absolute_brier_but_positive_skill = SimpleNamespace(
            atr_multiplier=Decimal("1.5"),
            brier_skill_vs_prior=0.010,
            calibrated_final_metrics=_metrics(
                brier=0.65,
                log_loss=1.08,
                balanced_accuracy=0.39,
                ece=0.03,
            ),
        )

        leader = min(
            (low_absolute_brier_but_negative_skill, higher_absolute_brier_but_positive_skill),
            key=_target_ranking_key,
        )

        self.assertEqual(leader.atr_multiplier, Decimal("1.5"))

    def test_step15_contract_is_fail_closed(self) -> None:
        root = Path(__file__).resolve().parents[3]
        with (root / "contracts" / "probability-research-v2.v1.schema.json").open(
            "r", encoding="utf-8"
        ) as file:
            schema = json.load(file)
        self.assertFalse(schema["properties"]["approved_for_live_inference"]["const"])


def _metrics(
    *,
    brier: float,
    log_loss: float,
    balanced_accuracy: float,
    ece: float,
) -> MetricSummary:
    return MetricSummary(
        sample_count=2_000,
        accuracy=0.48,
        balanced_accuracy=balanced_accuracy,
        multiclass_brier=brier,
        log_loss=log_loss,
        raw_ece_10_bin=ece,
        class_support=(("DOWN", 900), ("NEITHER", 200), ("UP", 900)),
        class_recall=(("DOWN", 0.45), ("NEITHER", 0.42), ("UP", 0.45)),
    )
