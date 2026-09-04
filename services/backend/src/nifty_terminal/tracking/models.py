"""Point-in-time records for paper research and production monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from math import isclose

from nifty_terminal.ml.models import TargetOutcome
from nifty_terminal.signals.models import SignalDirection


class PaperTradeStatus(StrEnum):
    PLANNED = "PLANNED"
    OPEN = "OPEN"
    TARGET_1_HIT = "TARGET_1_HIT"
    STOP_HIT = "STOP_HIT"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


class PaperTradeEventType(StrEnum):
    CREATED = "CREATED"
    OPENED = "OPENED"
    TARGET_1_HIT = "TARGET_1_HIT"
    STOP_HIT = "STOP_HIT"
    EXPIRED = "EXPIRED"
    INVALIDATED = "INVALIDATED"


class EvidenceStatus(StrEnum):
    READY = "READY"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    UNAVAILABLE = "UNAVAILABLE"
    BREACHED = "BREACHED"


class MonitorStatus(StrEnum):
    OK = "OK"
    WARN = "WARN"
    CRITICAL = "CRITICAL"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class TrackedPrediction:
    prediction_id: str
    signal_id: str
    snapshot_id: str
    instrument_id: str
    decision_time: datetime
    registered_at: datetime
    direction: SignalDirection
    predicted_outcome: TargetOutcome | None
    probabilities: tuple[tuple[str, float], ...] | None
    model_version: str
    calibration_version: str
    feature_version: str
    signal_policy_version: str
    input_revision_checksum: str

    def __post_init__(self) -> None:
        _aware(self.decision_time, "decision_time")
        _aware(self.registered_at, "registered_at")
        if self.registered_at < self.decision_time:
            raise ValueError("registered_at cannot precede decision_time")
        if len(self.input_revision_checksum) != 64:
            raise ValueError("input_revision_checksum must be a SHA-256 hex digest")
        if self.probabilities is None:
            if self.predicted_outcome is not None:
                raise ValueError("predicted_outcome requires probabilities")
        else:
            values = dict(self.probabilities)
            if set(values) != {"DOWN", "NEITHER", "UP"}:
                raise ValueError("probabilities must contain DOWN, NEITHER, and UP")
            if any(value < 0.0 or value > 1.0 for value in values.values()):
                raise ValueError("probabilities must be in [0, 1]")
            if not isclose(sum(values.values()), 1.0, abs_tol=1e-9):
                raise ValueError("probabilities must sum to 1")

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "prediction_id": self.prediction_id,
            "signal_id": self.signal_id,
            "snapshot_id": self.snapshot_id,
            "instrument_id": self.instrument_id,
            "decision_time": _time(self.decision_time),
            "registered_at": _time(self.registered_at),
            "direction": self.direction.value,
            "predicted_outcome": self.predicted_outcome.value if self.predicted_outcome else None,
            "probabilities": dict(self.probabilities) if self.probabilities else None,
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "feature_version": self.feature_version,
            "signal_policy_version": self.signal_policy_version,
            "input_revision_checksum": self.input_revision_checksum,
        }


@dataclass(frozen=True, slots=True)
class PredictionAssessment:
    assessment_id: str
    prediction_id: str
    signal_id: str
    snapshot_id: str
    instrument_id: str
    decision_time: datetime
    assessed_at: datetime
    predicted_outcome: TargetOutcome
    actual_outcome: TargetOutcome
    probabilities: tuple[tuple[str, float], ...]
    correct: bool
    first_touch_at: datetime | None
    label_version: str
    input_revision_checksum: str

    def __post_init__(self) -> None:
        _aware(self.decision_time, "decision_time")
        _aware(self.assessed_at, "assessed_at")
        if self.assessed_at <= self.decision_time:
            raise ValueError("assessment must occur after the prediction decision")
        if self.first_touch_at is not None:
            _aware(self.first_touch_at, "first_touch_at")
            if not self.decision_time < self.first_touch_at <= self.assessed_at:
                raise ValueError("first_touch_at must be inside the assessment window")
        if self.correct is not (self.predicted_outcome is self.actual_outcome):
            raise ValueError("correct must match predicted and actual outcome")

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "assessment_id": self.assessment_id,
            "prediction_id": self.prediction_id,
            "signal_id": self.signal_id,
            "snapshot_id": self.snapshot_id,
            "instrument_id": self.instrument_id,
            "decision_time": _time(self.decision_time),
            "assessed_at": _time(self.assessed_at),
            "predicted_outcome": self.predicted_outcome.value,
            "actual_outcome": self.actual_outcome.value,
            "probabilities": dict(self.probabilities),
            "correct": self.correct,
            "first_touch_at": _time(self.first_touch_at),
            "label_version": self.label_version,
            "input_revision_checksum": self.input_revision_checksum,
        }


@dataclass(frozen=True, slots=True)
class PaperTrade:
    paper_trade_id: str
    signal_id: str
    prediction_id: str
    snapshot_id: str
    instrument_id: str
    created_at: datetime
    expires_at: datetime
    direction: SignalDirection
    entry_low: Decimal
    entry_high: Decimal
    stop: Decimal
    target1: Decimal
    target2: Decimal
    target3: Decimal
    model_version: str
    calibration_version: str
    signal_policy_version: str
    input_revision_checksum: str
    status: PaperTradeStatus = PaperTradeStatus.PLANNED

    def __post_init__(self) -> None:
        _aware(self.created_at, "created_at")
        _aware(self.expires_at, "expires_at")
        if self.expires_at <= self.created_at:
            raise ValueError("paper trade expiry must follow creation")
        if self.direction is SignalDirection.WAIT:
            raise ValueError("WAIT cannot create a paper trade")
        if self.entry_low > self.entry_high:
            raise ValueError("entry_low cannot exceed entry_high")
        if self.direction is SignalDirection.BUY:
            valid = self.stop < self.entry_low <= self.entry_high < self.target1 < self.target2 < self.target3
        else:
            valid = self.stop > self.entry_high >= self.entry_low > self.target1 > self.target2 > self.target3
        if not valid:
            raise ValueError("paper trade levels are inconsistent with direction")

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "paper_trade_id": self.paper_trade_id,
            "signal_id": self.signal_id,
            "prediction_id": self.prediction_id,
            "snapshot_id": self.snapshot_id,
            "instrument_id": self.instrument_id,
            "created_at": _time(self.created_at),
            "expires_at": _time(self.expires_at),
            "direction": self.direction.value,
            "status": self.status.value,
            "entry_low": _decimal(self.entry_low),
            "entry_high": _decimal(self.entry_high),
            "stop": _decimal(self.stop),
            "target1": _decimal(self.target1),
            "target2": _decimal(self.target2),
            "target3": _decimal(self.target3),
            "model_version": self.model_version,
            "calibration_version": self.calibration_version,
            "signal_policy_version": self.signal_policy_version,
            "input_revision_checksum": self.input_revision_checksum,
            "unit": "NIFTY_INDEX_POINTS",
            "automatic_execution": False,
        }


@dataclass(frozen=True, slots=True)
class PaperTradeEvent:
    event_id: str
    paper_trade_id: str
    signal_id: str
    event_type: PaperTradeEventType
    status: PaperTradeStatus
    occurred_at: datetime
    observed_price: Decimal | None
    pnl_points: Decimal | None
    reason: str

    def __post_init__(self) -> None:
        _aware(self.occurred_at, "occurred_at")
        if not self.reason:
            raise ValueError("paper trade event reason is required")
        terminal = self.status in {
            PaperTradeStatus.TARGET_1_HIT,
            PaperTradeStatus.STOP_HIT,
            PaperTradeStatus.EXPIRED,
            PaperTradeStatus.INVALIDATED,
        }
        if self.pnl_points is not None and not terminal:
            raise ValueError("P&L points are permitted only on terminal events")

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "event_id": self.event_id,
            "paper_trade_id": self.paper_trade_id,
            "signal_id": self.signal_id,
            "event_type": self.event_type.value,
            "status": self.status.value,
            "occurred_at": _time(self.occurred_at),
            "observed_price": _decimal(self.observed_price),
            "pnl_points": _decimal(self.pnl_points),
            "reason": self.reason,
            "unit": "NIFTY_INDEX_POINTS",
        }


@dataclass(frozen=True, slots=True)
class PredictionAnalytics:
    instrument_id: str
    generated_at: datetime
    minimum_sample: int
    tracked_predictions: int
    assessed_predictions: int
    pending_predictions: int
    actual_outcome_counts: tuple[tuple[str, int], ...]
    metrics_status: EvidenceStatus
    accuracy: float | None
    multiclass_brier_score: float | None
    expected_calibration_error: float | None
    paper_trades: int
    closed_paper_trades: int
    paper_metrics_status: EvidenceStatus
    paper_win_rate: float | None
    paper_total_points: Decimal | None
    blockers: tuple[str, ...]

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "instrument_id": self.instrument_id,
            "generated_at": _time(self.generated_at),
            "minimum_sample": self.minimum_sample,
            "tracked_predictions": self.tracked_predictions,
            "assessed_predictions": self.assessed_predictions,
            "pending_predictions": self.pending_predictions,
            "actual_outcome_counts": dict(self.actual_outcome_counts),
            "metrics_status": self.metrics_status.value,
            "accuracy": self.accuracy,
            "multiclass_brier_score": self.multiclass_brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "paper_trades": self.paper_trades,
            "closed_paper_trades": self.closed_paper_trades,
            "paper_metrics_status": self.paper_metrics_status.value,
            "paper_win_rate": self.paper_win_rate,
            "paper_total_points": _decimal(self.paper_total_points),
            "blockers": list(self.blockers),
            "performance_claim_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class MonitoringCheck:
    key: str
    status: MonitorStatus
    observed_at: datetime
    detail: str

    def to_contract(self) -> dict[str, object]:
        return {
            "key": self.key,
            "status": self.status.value,
            "observed_at": _time(self.observed_at),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class MonitoringView:
    instrument_id: str
    generated_at: datetime
    overall_status: MonitorStatus
    checks: tuple[MonitoringCheck, ...]
    model_drift_status: EvidenceStatus
    probability_drift_status: EvidenceStatus
    alerting_enabled: bool

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "instrument_id": self.instrument_id,
            "generated_at": _time(self.generated_at),
            "overall_status": self.overall_status.value,
            "checks": [item.to_contract() for item in self.checks],
            "model_drift_status": self.model_drift_status.value,
            "probability_drift_status": self.probability_drift_status.value,
            "alerting_enabled": self.alerting_enabled,
        }


@dataclass(frozen=True, slots=True)
class TrackingOverview:
    instrument_id: str
    generated_at: datetime
    analytics: PredictionAnalytics
    monitoring: MonitoringView
    paper_trades: tuple[PaperTrade, ...]
    recent_paper_events: tuple[PaperTradeEvent, ...]

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "instrument_id": self.instrument_id,
            "generated_at": _time(self.generated_at),
            "analytics": self.analytics.to_contract(),
            "monitoring": self.monitoring.to_contract(),
            "paper_trades": [item.to_contract() for item in self.paper_trades],
            "recent_paper_events": [item.to_contract() for item in self.recent_paper_events],
            "paper_only": True,
            "automatic_execution": False,
        }


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


def _time(value: datetime | None) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def _decimal(value: Decimal | None) -> str | None:
    return format(value, "f") if value is not None else None
