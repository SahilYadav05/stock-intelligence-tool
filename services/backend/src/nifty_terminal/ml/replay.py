"""Immutable assessment events for historical simulated-live predictions."""

from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.ml.models import ReplayAssessment, ResearchPrediction


def build_replay_assessments(
    predictions: tuple[ResearchPrediction, ...],
) -> tuple[ReplayAssessment, ...]:
    return tuple(
        ReplayAssessment(
            assessment_id=str(
                uuid5(
                    NAMESPACE_URL,
                    f"replay-assessment:{prediction.prediction_id}:{prediction.label_id}",
                )
            ),
            prediction_id=prediction.prediction_id,
            label_id=prediction.label_id,
            assessed_at=prediction.label_window_end,
            actual_outcome=prediction.actual_outcome,
            predicted_outcome=prediction.predicted_outcome,
            correct=prediction.actual_outcome is prediction.predicted_outcome,
        )
        for prediction in predictions
    )
