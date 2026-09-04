"""Immutable market-state snapshot contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from nifty_terminal.domain.candle import Timeframe
from nifty_terminal.domain.enums import ConnectionState


class DataMode(StrEnum):
    REPLAY = "REPLAY"
    LIVE = "LIVE"


@dataclass(frozen=True, slots=True)
class MarketStateSnapshot:
    schema_version: int
    snapshot_id: str
    instrument_id: str
    decision_time: datetime
    created_at: datetime
    data_as_of: datetime
    data_mode: DataMode
    data_status: ConnectionState
    primary_timeframe: Timeframe
    primary_candle_id: str
    context_15m_candle_id: str | None
    context_1h_candle_id: str | None
    recent_primary_candle_ids: tuple[str, ...]
    developing_candle_id: str | None
    model_input_candle_ids: tuple[str, ...]
    source_watermark: str
    candle_revision_checksum: str
    live_inference_eligible: bool
    blockers: tuple[str, ...]

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "instrument_id": self.instrument_id,
            "decision_time": _datetime_text(self.decision_time),
            "created_at": _datetime_text(self.created_at),
            "data_as_of": _datetime_text(self.data_as_of),
            "data_mode": self.data_mode.value,
            "data_status": self.data_status.value,
            "primary_timeframe": self.primary_timeframe.value,
            "primary_candle_id": self.primary_candle_id,
            "context_15m_candle_id": self.context_15m_candle_id,
            "context_1h_candle_id": self.context_1h_candle_id,
            "recent_primary_candle_ids": list(self.recent_primary_candle_ids),
            "developing_candle_id": self.developing_candle_id,
            "model_input_candle_ids": list(self.model_input_candle_ids),
            "source_watermark": self.source_watermark,
            "candle_revision_checksum": self.candle_revision_checksum,
            "live_inference_eligible": self.live_inference_eligible,
            "blockers": list(self.blockers),
        }


def _datetime_text(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
