"""Thread-safe append-only process projection for Step 9 tracking APIs."""

from __future__ import annotations

from threading import RLock

from nifty_terminal.tracking.models import (
    PaperTrade,
    PaperTradeEvent,
    PredictionAssessment,
    TrackedPrediction,
)


class InMemoryTrackingReadModel:
    def __init__(self) -> None:
        self._lock = RLock()
        self._predictions: dict[str, TrackedPrediction] = {}
        self._assessments: dict[str, PredictionAssessment] = {}
        self._trades: dict[str, PaperTrade] = {}
        self._events: dict[str, PaperTradeEvent] = {}

    def put_prediction(self, item: TrackedPrediction) -> bool:
        return self._put(self._predictions, item.prediction_id, item, "prediction")

    def put_assessment(self, item: PredictionAssessment) -> bool:
        with self._lock:
            if item.prediction_id not in self._predictions:
                raise ValueError("assessment requires a registered prediction")
            if any(
                existing.prediction_id == item.prediction_id
                for existing in self._assessments.values()
            ):
                existing = next(
                    value
                    for value in self._assessments.values()
                    if value.prediction_id == item.prediction_id
                )
                if existing == item:
                    return False
                raise ValueError("prediction already has an immutable assessment")
            return self._put(self._assessments, item.assessment_id, item, "assessment")

    def put_trade(self, item: PaperTrade) -> bool:
        with self._lock:
            if item.prediction_id not in self._predictions:
                raise ValueError("paper trade requires a registered prediction")
            return self._put(self._trades, item.paper_trade_id, item, "paper trade")

    def put_event(self, item: PaperTradeEvent) -> bool:
        with self._lock:
            if item.paper_trade_id not in self._trades:
                raise ValueError("paper event requires a registered paper trade")
            terminal = self.terminal_event(item.paper_trade_id)
            if terminal is not None:
                if terminal == item:
                    return False
                raise ValueError("paper trade already has an immutable terminal event")
            return self._put(self._events, item.event_id, item, "paper event")

    def get_prediction(self, prediction_id: str) -> TrackedPrediction | None:
        with self._lock:
            return self._predictions.get(prediction_id)

    def predictions(self, instrument_id: str) -> tuple[TrackedPrediction, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (item for item in self._predictions.values() if item.instrument_id == instrument_id),
                    key=lambda item: (item.decision_time, item.prediction_id),
                )
            )

    def assessments(self, instrument_id: str) -> tuple[PredictionAssessment, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (item for item in self._assessments.values() if item.instrument_id == instrument_id),
                    key=lambda item: (item.assessed_at, item.assessment_id),
                )
            )

    def trades(self, instrument_id: str) -> tuple[PaperTrade, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (item for item in self._trades.values() if item.instrument_id == instrument_id),
                    key=lambda item: (item.created_at, item.paper_trade_id),
                    reverse=True,
                )
            )

    def events(self, instrument_id: str) -> tuple[PaperTradeEvent, ...]:
        trade_ids = {item.paper_trade_id for item in self.trades(instrument_id)}
        with self._lock:
            return tuple(
                sorted(
                    (item for item in self._events.values() if item.paper_trade_id in trade_ids),
                    key=lambda item: (item.occurred_at, item.event_id),
                    reverse=True,
                )
            )

    def opened_price(self, paper_trade_id: str):
        with self._lock:
            opened = [
                item
                for item in self._events.values()
                if item.paper_trade_id == paper_trade_id and item.status.value == "OPEN"
            ]
        return opened[0].observed_price if opened else None

    def terminal_event(self, paper_trade_id: str) -> PaperTradeEvent | None:
        terminal_statuses = {"TARGET_1_HIT", "STOP_HIT", "EXPIRED", "INVALIDATED"}
        with self._lock:
            events = [
                item
                for item in self._events.values()
                if item.paper_trade_id == paper_trade_id and item.status.value in terminal_statuses
            ]
        return sorted(events, key=lambda item: item.occurred_at)[-1] if events else None

    def _put(self, store: dict, key: str, item, label: str) -> bool:
        existing = store.get(key)
        if existing is not None:
            if existing == item:
                return False
            raise ValueError(f"{label} identity cannot be replaced")
        store[key] = item
        return True
