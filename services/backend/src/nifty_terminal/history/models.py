"""Immutable historical-data acquisition and audit records."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from nifty_terminal.domain.candle import FinalizedMinuteBarInput, Timeframe


class QualityStatus(StrEnum):
    PASS = "PASS"
    DEGRADED = "DEGRADED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class HistoricalRequest:
    instrument_id: str
    timeframe: Timeframe
    starts_at: datetime
    ends_at: datetime

    def __post_init__(self) -> None:
        if self.timeframe is not Timeframe.M1:
            raise ValueError("Step 5 historical acquisition requires authoritative 1m bars")
        for value in (self.starts_at, self.ends_at):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("Historical request timestamps must be timezone-aware")
        if self.starts_at >= self.ends_at:
            raise ValueError("Historical request starts_at must precede ends_at")


@dataclass(frozen=True, slots=True)
class HistoricalBatch:
    provider: str
    source_label: str
    source_sha256: str
    acquired_at: datetime
    request: HistoricalRequest
    rows: tuple[FinalizedMinuteBarInput, ...]


@dataclass(frozen=True, slots=True)
class HistoricalQualityReport:
    status: QualityStatus
    total_rows: int
    unique_minute_buckets: int
    correction_rows: int
    missing_minutes: int
    duplicate_provider_ids: int
    out_of_order_rows: int
    out_of_request_rows: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_contract(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "total_rows": self.total_rows,
            "unique_minute_buckets": self.unique_minute_buckets,
            "correction_rows": self.correction_rows,
            "missing_minutes": self.missing_minutes,
            "duplicate_provider_ids": self.duplicate_provider_ids,
            "out_of_order_rows": self.out_of_order_rows,
            "out_of_request_rows": self.out_of_request_rows,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class HistoricalImportResult:
    dataset_id: str
    imported: bool
    candle_revision_count: int
    quality: HistoricalQualityReport
