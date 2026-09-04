"""Strict adapter for a provider-authorized historical CSV export."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
from pathlib import Path

from nifty_terminal.domain.candle import FinalizedMinuteBarInput
from nifty_terminal.history.models import HistoricalBatch, HistoricalRequest
from nifty_terminal.history.sources.base import HistoricalDataSource


REQUIRED_COLUMNS = frozenset(
    {
        "provider_bar_id",
        "provider_revision",
        "opens_at",
        "closes_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "finalized_at",
        "source_watermark",
    }
)


class CsvHistoricalDataSource(HistoricalDataSource):
    def __init__(self, *, path: Path, provider: str) -> None:
        if not provider.strip():
            raise ValueError("Historical CSV provider name is required")
        self._path = path
        self._provider = provider.casefold().strip()

    def fetch(self, request: HistoricalRequest) -> HistoricalBatch:
        payload = self._path.read_bytes()
        rows: list[FinalizedMinuteBarInput] = []
        with self._path.open("r", encoding="utf-8-sig", newline="") as source_file:
            reader = csv.DictReader(source_file)
            columns = frozenset(reader.fieldnames or ())
            missing = sorted(REQUIRED_COLUMNS.difference(columns))
            if missing:
                raise ValueError(f"Historical CSV missing required columns: {missing}")
            for line_number, row in enumerate(reader, start=2):
                try:
                    rows.append(
                        FinalizedMinuteBarInput(
                            provider_bar_id=_required(row, "provider_bar_id"),
                            provider=self._provider,
                            instrument_id=request.instrument_id,
                            opens_at=_timestamp(row, "opens_at"),
                            closes_at=_timestamp(row, "closes_at"),
                            open=_decimal(row, "open"),
                            high=_decimal(row, "high"),
                            low=_decimal(row, "low"),
                            close=_decimal(row, "close"),
                            volume=_optional_decimal(row, "volume"),
                            provider_revision=int(_required(row, "provider_revision")),
                            finalized_at=_timestamp(row, "finalized_at"),
                            source_watermark=_required(row, "source_watermark"),
                        )
                    )
                except (ValueError, InvalidOperation) as error:
                    raise ValueError(f"Invalid historical CSV row {line_number}: {error}") from error

        return HistoricalBatch(
            provider=self._provider,
            source_label=self._path.name,
            source_sha256=hashlib.sha256(payload).hexdigest(),
            acquired_at=datetime.now(timezone.utc),
            request=request,
            rows=tuple(rows),
        )


def _required(row: dict[str, str | None], key: str) -> str:
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _timestamp(row: dict[str, str | None], key: str) -> datetime:
    text = _required(row, key)
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{key} must include a timezone")
    return value.astimezone(timezone.utc)


def _decimal(row: dict[str, str | None], key: str) -> Decimal:
    return Decimal(_required(row, key))


def _optional_decimal(row: dict[str, str | None], key: str) -> Decimal | None:
    text = (row.get(key) or "").strip()
    return Decimal(text) if text else None
