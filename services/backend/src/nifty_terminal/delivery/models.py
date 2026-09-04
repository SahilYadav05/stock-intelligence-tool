"""Validated chart-ready view of one canonical market-state snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import hashlib

from nifty_terminal.domain.candle import Candle, CandleStatus, Timeframe
from nifty_terminal.snapshots.models import MarketStateSnapshot


class SyncState(StrEnum):
    SYNCED = "SYNCED"
    SYNCING = "SYNCING"


@dataclass(frozen=True, slots=True)
class MarketStateView:
    """One atomic payload rendered by the chart and consumed by later analysis.

    Construction fails if the supplied candle revisions cannot satisfy every
    candle ID named by the snapshot. A mismatched payload is never published.
    """

    schema_version: int
    snapshot: MarketStateSnapshot
    finalized_candles: tuple[Candle, ...]
    developing_candle: Candle | None
    published_at: datetime
    sync_state: SyncState = SyncState.SYNCED

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("MarketStateView schema_version must be 1")
        if self.published_at.tzinfo is None or self.published_at.utcoffset() is None:
            raise ValueError("published_at must be timezone-aware")
        if self.sync_state is not SyncState.SYNCED:
            raise ValueError("Only a fully synchronized view may be published")

        candle_by_id = {item.candle_id: item for item in self.finalized_candles}
        if len(candle_by_id) != len(self.finalized_candles):
            raise ValueError("Finalized candle IDs must be unique")
        if any(item.status is not CandleStatus.FINALIZED for item in self.finalized_candles):
            raise ValueError("finalized_candles cannot contain a developing candle")
        if any(
            item.instrument_id != self.snapshot.instrument_id
            for item in self.finalized_candles
        ):
            raise ValueError("Every candle must match the snapshot instrument")
        if any(item.closes_at > self.snapshot.decision_time for item in self.finalized_candles):
            raise ValueError("Finalized view cannot contain data after the snapshot decision time")
        bucket_keys = {(item.timeframe, item.opens_at) for item in self.finalized_candles}
        if len(bucket_keys) != len(self.finalized_candles):
            raise ValueError("View cannot contain multiple revisions for one candle bucket")

        required_ids = set(self.snapshot.model_input_candle_ids)
        missing = sorted(required_ids.difference(candle_by_id))
        if missing:
            raise ValueError(f"Snapshot candle revisions are missing from view: {missing}")
        primary = candle_by_id.get(self.snapshot.primary_candle_id)
        if primary is None or primary.timeframe is not Timeframe.M5:
            raise ValueError("Snapshot primary candle must be present as finalized 5m data")
        if primary.closes_at != self.snapshot.decision_time:
            raise ValueError("Snapshot decision time must equal the primary candle close")
        expected_checksum = hashlib.sha256(
            "|".join(self.snapshot.model_input_candle_ids).encode("utf-8")
        ).hexdigest()
        if expected_checksum != self.snapshot.candle_revision_checksum:
            raise ValueError("Snapshot candle revision checksum is invalid")

        if self.developing_candle is None:
            if self.snapshot.developing_candle_id is not None:
                raise ValueError("Snapshot names a developing candle not present in the view")
        else:
            if self.developing_candle.status is not CandleStatus.DEVELOPING:
                raise ValueError("developing_candle must have DEVELOPING status")
            if self.developing_candle.candle_id != self.snapshot.developing_candle_id:
                raise ValueError("Developing candle revision does not match the snapshot")
            if self.developing_candle.candle_id in required_ids:
                raise ValueError("Developing candle cannot be a model input")

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sync_state": self.sync_state.value,
            "published_at": _datetime_text(self.published_at),
            "snapshot": self.snapshot.to_contract(),
            "finalized_candles": [item.to_contract() for item in self.finalized_candles],
            "developing_candle": (
                self.developing_candle.to_contract() if self.developing_candle else None
            ),
        }


def _datetime_text(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
