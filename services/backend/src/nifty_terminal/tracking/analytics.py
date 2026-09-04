"""Evidence-gated prediction and paper-trade analytics."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from nifty_terminal.tracking.models import (
    EvidenceStatus,
    PaperTrade,
    PaperTradeEvent,
    PaperTradeStatus,
    PredictionAnalytics,
    PredictionAssessment,
    TrackedPrediction,
)


MINIMUM_REPORTING_SAMPLE = 30


def build_prediction_analytics(
    *,
    instrument_id: str,
    generated_at: datetime,
    predictions: tuple[TrackedPrediction, ...],
    assessments: tuple[PredictionAssessment, ...],
    paper_trades: tuple[PaperTrade, ...],
    paper_events: tuple[PaperTradeEvent, ...],
    minimum_sample: int = MINIMUM_REPORTING_SAMPLE,
) -> PredictionAnalytics:
    if minimum_sample < 1:
        raise ValueError("minimum_sample must be positive")
    relevant_assessments = tuple(
        item for item in assessments if item.instrument_id == instrument_id
    )
    relevant_predictions = tuple(
        item for item in predictions if item.instrument_id == instrument_id
    )
    assessed = len(relevant_assessments)
    metrics_status = (
        EvidenceStatus.UNAVAILABLE
        if not relevant_predictions
        else EvidenceStatus.READY
        if assessed >= minimum_sample
        else EvidenceStatus.INSUFFICIENT_SAMPLE
    )
    outcome_counts = {"DOWN": 0, "NEITHER": 0, "UP": 0}
    for item in relevant_assessments:
        outcome_counts[item.actual_outcome.value] += 1

    accuracy = brier = ece = None
    if metrics_status is EvidenceStatus.READY:
        accuracy = sum(item.correct for item in relevant_assessments) / assessed
        brier = sum(_brier(item) for item in relevant_assessments) / assessed
        ece = _expected_calibration_error(relevant_assessments)

    relevant_trades = tuple(item for item in paper_trades if item.instrument_id == instrument_id)
    terminal_events = tuple(
        item
        for item in paper_events
        if item.paper_trade_id in {trade.paper_trade_id for trade in relevant_trades}
        and item.status
        in {
            PaperTradeStatus.TARGET_1_HIT,
            PaperTradeStatus.STOP_HIT,
            PaperTradeStatus.EXPIRED,
        }
        and item.pnl_points is not None
    )
    paper_status = (
        EvidenceStatus.UNAVAILABLE
        if not relevant_trades
        else EvidenceStatus.READY
        if len(terminal_events) >= minimum_sample
        else EvidenceStatus.INSUFFICIENT_SAMPLE
    )
    paper_win_rate = paper_total = None
    if paper_status is EvidenceStatus.READY:
        paper_win_rate = sum(item.pnl_points > 0 for item in terminal_events) / len(terminal_events)
        paper_total = sum((item.pnl_points for item in terminal_events), Decimal("0"))

    blockers: list[str] = []
    if metrics_status is not EvidenceStatus.READY:
        blockers.append(f"PREDICTION_SAMPLE:{assessed}/{minimum_sample}")
    if paper_status is not EvidenceStatus.READY:
        blockers.append(f"CLOSED_PAPER_SAMPLE:{len(terminal_events)}/{minimum_sample}")
    return PredictionAnalytics(
        instrument_id=instrument_id,
        generated_at=generated_at,
        minimum_sample=minimum_sample,
        tracked_predictions=len(relevant_predictions),
        assessed_predictions=assessed,
        pending_predictions=max(0, len(relevant_predictions) - assessed),
        actual_outcome_counts=tuple(sorted(outcome_counts.items())),
        metrics_status=metrics_status,
        accuracy=accuracy,
        multiclass_brier_score=brier,
        expected_calibration_error=ece,
        paper_trades=len(relevant_trades),
        closed_paper_trades=len(terminal_events),
        paper_metrics_status=paper_status,
        paper_win_rate=paper_win_rate,
        paper_total_points=paper_total,
        blockers=tuple(blockers),
    )


def _brier(item: PredictionAssessment) -> float:
    actual = item.actual_outcome.value
    return sum(
        (probability - (1.0 if outcome == actual else 0.0)) ** 2
        for outcome, probability in item.probabilities
    )


def _expected_calibration_error(
    assessments: tuple[PredictionAssessment, ...],
    bins: int = 10,
) -> float:
    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for item in assessments:
        confidence = max(value for _, value in item.probabilities)
        index = min(bins - 1, int(confidence * bins))
        grouped[index].append((confidence, item.correct))
    total = len(assessments)
    return sum(
        (len(bucket) / total)
        * abs(
            sum(confidence for confidence, _ in bucket) / len(bucket)
            - sum(correct for _, correct in bucket) / len(bucket)
        )
        for bucket in grouped
        if bucket
    )
