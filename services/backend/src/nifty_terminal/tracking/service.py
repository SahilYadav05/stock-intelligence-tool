"""Event-driven registration, assessment, paper simulation, and overview service."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid5

from nifty_terminal.dashboard.models import AnalysisView
from nifty_terminal.delivery.models import MarketStateView
from nifty_terminal.ml.definitions import LABEL_VERSION
from nifty_terminal.ml.models import TargetOutcome
from nifty_terminal.hardening.models import DriftEvidence
from nifty_terminal.tracking.analytics import build_prediction_analytics
from nifty_terminal.tracking.models import PredictionAssessment, TrackedPrediction, TrackingOverview
from nifty_terminal.tracking.monitoring import build_monitoring_view
from nifty_terminal.tracking.paper import assess_paper_trade, create_paper_trade
from nifty_terminal.tracking.read_model import InMemoryTrackingReadModel


class TrackingService:
    def __init__(
        self,
        read_model: InMemoryTrackingReadModel | None = None,
        *,
        drift_evidence: DriftEvidence | None = None,
    ) -> None:
        self.read_model = read_model or InMemoryTrackingReadModel()
        self.drift_evidence = drift_evidence

    def register_analysis(self, analysis: AnalysisView) -> TrackedPrediction:
        probabilities = analysis.signal.probabilities
        predicted = None
        if probabilities:
            predicted = TargetOutcome(max(probabilities, key=lambda item: item[1])[0])
        item = TrackedPrediction(
            prediction_id=analysis.signal.prediction_id,
            signal_id=analysis.signal.signal_id,
            snapshot_id=analysis.snapshot_id,
            instrument_id=analysis.instrument_id,
            decision_time=analysis.decision_time,
            registered_at=analysis.generated_at,
            direction=analysis.signal.direction,
            predicted_outcome=predicted,
            probabilities=probabilities,
            model_version=analysis.model_version,
            calibration_version=analysis.calibration_version,
            feature_version=analysis.feature_version,
            signal_policy_version=analysis.signal.signal_policy_version,
            input_revision_checksum=analysis.candle_revision_checksum,
        )
        self.read_model.put_prediction(item)
        trade = create_paper_trade(analysis)
        if trade is not None:
            self.read_model.put_trade(trade)
        return item

    def assess_prediction(
        self,
        prediction_id: str,
        *,
        actual_outcome: TargetOutcome,
        assessed_at: datetime,
        first_touch_at: datetime | None,
    ) -> PredictionAssessment:
        prediction = self.read_model.get_prediction(prediction_id)
        if prediction is None:
            raise ValueError("prediction is not registered")
        if prediction.predicted_outcome is None or prediction.probabilities is None:
            raise ValueError("prediction has no calibrated probabilities to assess")
        identity = f"prediction-assessment:{prediction_id}:{LABEL_VERSION}"
        item = PredictionAssessment(
            assessment_id=str(uuid5(NAMESPACE_URL, identity)),
            prediction_id=prediction.prediction_id,
            signal_id=prediction.signal_id,
            snapshot_id=prediction.snapshot_id,
            instrument_id=prediction.instrument_id,
            decision_time=prediction.decision_time,
            assessed_at=assessed_at,
            predicted_outcome=prediction.predicted_outcome,
            actual_outcome=actual_outcome,
            probabilities=prediction.probabilities,
            correct=prediction.predicted_outcome is actual_outcome,
            first_touch_at=first_touch_at,
            label_version=LABEL_VERSION,
            input_revision_checksum=prediction.input_revision_checksum,
        )
        self.read_model.put_assessment(item)
        return item

    def assess_paper_trades(
        self,
        instrument_id: str,
        *,
        observed_at: datetime,
        high: Decimal,
        low: Decimal,
        close: Decimal,
    ) -> int:
        inserted = 0
        for trade in self.read_model.trades(instrument_id):
            if self.read_model.terminal_event(trade.paper_trade_id) is not None:
                continue
            event = assess_paper_trade(
                trade,
                observed_at=observed_at,
                high=high,
                low=low,
                close=close,
                opened_price=self.read_model.opened_price(trade.paper_trade_id),
            )
            if event is not None and self.read_model.put_event(event):
                inserted += 1
        return inserted

    def overview(
        self,
        instrument_id: str,
        *,
        generated_at: datetime,
        market_view: MarketStateView | None,
        analysis: AnalysisView | None,
    ) -> TrackingOverview:
        predictions = self.read_model.predictions(instrument_id)
        assessments = self.read_model.assessments(instrument_id)
        trades = self.read_model.trades(instrument_id)
        events = self.read_model.events(instrument_id)
        analytics = build_prediction_analytics(
            instrument_id=instrument_id,
            generated_at=generated_at,
            predictions=predictions,
            assessments=assessments,
            paper_trades=trades,
            paper_events=events,
        )
        monitoring = build_monitoring_view(
            instrument_id=instrument_id,
            generated_at=generated_at,
            market_view=market_view,
            analysis=analysis,
            analytics=analytics,
            drift_evidence=self.drift_evidence,
        )
        return TrackingOverview(
            instrument_id=instrument_id,
            generated_at=generated_at,
            analytics=analytics,
            monitoring=monitoring,
            paper_trades=trades[:100],
            recent_paper_events=events[:200],
        )
