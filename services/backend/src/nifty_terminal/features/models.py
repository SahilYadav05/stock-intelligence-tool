"""Immutable feature rows and snapshot-level feature vectors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from nifty_terminal.domain.candle import Timeframe


FeatureValue = Decimal | int | bool | None


@dataclass(frozen=True, slots=True)
class PriceFeatureRow:
    schema_version: int
    feature_version: str
    feature_set_hash: str
    source_candle_id: str
    instrument_id: str
    timeframe: Timeframe
    decision_time: datetime
    values: tuple[tuple[str, FeatureValue], ...]
    is_ready: bool
    blockers: tuple[str, ...]

    def get(self, name: str) -> FeatureValue:
        return dict(self.values).get(name)

    def contract_values(self) -> dict[str, object]:
        return {name: _contract_value(value) for name, value in self.values}

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "feature_version": self.feature_version,
            "feature_set_hash": self.feature_set_hash,
            "source_candle_id": self.source_candle_id,
            "instrument_id": self.instrument_id,
            "timeframe": self.timeframe.value,
            "decision_time": _time(self.decision_time),
            "values": self.contract_values(),
            "is_ready": self.is_ready,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True, slots=True)
class FeatureSnapshot:
    schema_version: int
    feature_snapshot_id: str
    feature_version: str
    feature_set_hash: str
    market_snapshot_id: str
    instrument_id: str
    decision_time: datetime
    input_revision_checksum: str
    values: tuple[tuple[str, FeatureValue], ...]
    is_ready: bool
    blockers: tuple[str, ...]

    def to_contract(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "feature_snapshot_id": self.feature_snapshot_id,
            "feature_version": self.feature_version,
            "feature_set_hash": self.feature_set_hash,
            "market_snapshot_id": self.market_snapshot_id,
            "instrument_id": self.instrument_id,
            "decision_time": _time(self.decision_time),
            "input_revision_checksum": self.input_revision_checksum,
            "values": {name: _contract_value(value) for name, value in self.values},
            "is_ready": self.is_ready,
            "blockers": list(self.blockers),
        }


def _contract_value(value: FeatureValue) -> object:
    return format(value, "f") if isinstance(value, Decimal) else value


def _time(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
